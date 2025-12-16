"""
MLO Landmark Detection Training

Trains CRAUNet model to detect pectoral muscle line and nipple position
in MLO (Medio-Lateral Oblique) mammography images.

Usage:
    python main.py --config configs/example_config.json

Output:
    - 6 coordinates: pec1_x, pec1_y, pec2_x, pec2_y, nipple_x, nipple_y
"""

import argparse
import json
import os
import gc
import traceback

import pandas as pd
import torch
from tqdm import tqdm

from utils.dataloader import create_dataloaders, preprocess_data
from utils.train import Trainer
from utils.validate import Validator
from utils.models import CRAUNet
from utils.early_stopping import EarlyStopping


# Output features mapping for different tasks
OUT_FEATURES_MAP = {
    'pectoralis': 4,  # pec1(x,y), pec2(x,y)
    'nipple': 2,      # nipple(x,y)
    'all': 6          # pec1(x,y), pec2(x,y), nipple(x,y)
}


def load_config(config_path: str) -> dict:
    """
    Load training configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Configuration dictionary with device set
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['device'] = torch.device(
        config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    )
    return config


def create_model(config: dict, out_features: int) -> CRAUNet:
    """
    Create CRAUNet model for MLO landmark detection.
    
    Args:
        config: Training configuration
        out_features: Number of output coordinates
        
    Returns:
        Initialized CRAUNet model
    """
    dropout_rate = config.get('dropout_rate', 0.3)
    
    model = CRAUNet(
        in_channels=1,
        out_features=out_features,
        dropout_rate=dropout_rate
    )
    
    return model


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: CRAUNet,
    trainer: Trainer,
    train_loss: float,
    val_loss: float,
    best_val_loss: float
) -> None:
    """Save training checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    try:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'warmup_scheduler_state_dict': trainer.warmup_scheduler.state_dict(),
            'cosine_scheduler_state_dict': trainer.cosine_scheduler.state_dict(),
            'current_step': trainer.current_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}")


def main(config: dict) -> None:
    """
    Main training function for MLO landmark detection.
    
    Args:
        config: Training configuration dictionary
    """
    print("=" * 60)
    print("MLO Landmark Detection Training - CRAUNet")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Model: CRAUNet (Coordinate-aware Residual Attention UNet)")
    print("=" * 60)

    # Load and preprocess data
    unified_df = preprocess_data(config)
    dataloaders = create_dataloaders(unified_df, config)
    train_loader, val_loader = dataloaders['Train'], dataloaders['Validation']

    # Determine output features based on target task
    target_task = config.get('target_task', 'all')
    if target_task not in OUT_FEATURES_MAP:
        raise ValueError(f"Invalid target task: {target_task}. Must be one of {list(OUT_FEATURES_MAP.keys())}")
    
    out_features = OUT_FEATURES_MAP[target_task]
    print(f"\nTarget task: {target_task} ({out_features} outputs)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    model = create_model(config, out_features)
    model = model.to(config['device'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Initialize trainer and validator
    trainer = Trainer(config, train_loader, model)
    validator = Validator(config, val_loader, model)

    # Training state
    best_model_state = None
    best_val_loss = float('inf')
    metrics = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 15),
        verbose=True,
        delta=config.get('early_stopping_delta', 0.001),
        path=config['best_model_path']
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(config['best_model_path']), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    epoch_pbar = tqdm(range(config['num_epochs']), desc='Training', unit='epoch')
    
    for epoch in epoch_pbar:
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Train and validate
            train_loss, train_pec1, train_pec2, train_nipple = trainer.train(epoch)
            val_loss, val_pec1, val_pec2, val_nipple, is_best = validator.validate()
            
            current_lr = trainer.optimizer.param_groups[0]['lr']

            # Check for NaN/Inf
            if train_loss != train_loss or val_loss != val_loss:
                print(f"\nERROR: NaN detected at epoch {epoch + 1}! Stopping.")
                break
            
            if abs(train_loss) == float('inf') or abs(val_loss) == float('inf'):
                print(f"\nERROR: Inf detected at epoch {epoch + 1}! Stopping.")
                break

            # Log progress
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print(f"  Train - pec1: {train_pec1:.5f}, pec2: {train_pec2:.5f}, nipple: {train_nipple:.5f}, Total: {train_loss:.5f}")
            print(f"  Val   - pec1: {val_pec1:.5f}, pec2: {val_pec2:.5f}, nipple: {val_nipple:.5f}, Total: {val_loss:.5f}")
            print(f"  LR: {current_lr:.2e}")

            # Save metrics
            metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_pec1': train_pec1,
                'train_pec2': train_pec2,
                'train_nipple': train_nipple,
                'validation_loss': val_loss,
                'val_pec1': val_pec1,
                'val_pec2': val_pec2,
                'val_nipple': val_nipple,
                'learning_rate': current_lr
            }) 

            # Save best model
            if is_best:
                best_model_state = model.state_dict().copy()
                best_val_loss = val_loss
                try:
                    os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
                    torch.save(best_model_state, config['best_model_path'])
                    print(f"  [BEST] Model saved")
                except Exception as e:
                    print(f"  Warning: Failed to save best model: {e}")
            
            # Early stopping check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best validation loss: {early_stopping.val_loss_min:.6f}")
                break
                
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                save_checkpoint(checkpoint_dir, epoch, model, trainer,
                              train_loss, val_loss, best_val_loss)
            
            # Save metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    pd.DataFrame(metrics).to_csv('training_metrics.csv', index=False)
                except Exception as e:
                    print(f"  Warning: Failed to save metrics: {e}")
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"\nCUDA OUT OF MEMORY at epoch {epoch + 1}!")
                if torch.cuda.is_available():
                    print(f"Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                print(f"\nRUNTIME ERROR at epoch {epoch + 1}: {e}")
            traceback.print_exc()
            break
        
        except Exception as e:
            print(f"\nUNEXPECTED ERROR at epoch {epoch + 1}: {e}")
            traceback.print_exc()
            break

    epoch_pbar.close()

    # Final save
    if best_model_state:
        try:
            os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
            torch.save(best_model_state, config['best_model_path'])
            print(f"\nBest model saved: {config['best_model_path']}")
            print(f"Best validation loss: {best_val_loss:.5f}")
        except Exception as e:
            print(f"\nWarning: Failed to save final model: {e}")

    # Save final metrics
    try:
        pd.DataFrame(metrics).to_csv('training_metrics.csv', index=False)
        print(f"Metrics saved: training_metrics.csv")
    except Exception as e:
        print(f"\nWarning: Failed to save metrics: {e}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLO Landmark Detection Training")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to configuration JSON file'
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
