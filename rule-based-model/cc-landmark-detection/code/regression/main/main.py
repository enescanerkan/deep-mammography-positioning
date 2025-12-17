# main.py - CC Nipple Detection Training
import argparse
import json
import pandas as pd
import torch
import os

from utils.dataloader import create_dataloaders, preprocess_data
from utils.train import Trainer
from utils.validate import Validator
from utils.models import CRAUNet

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    return config

def main(config):
    print(f"\n{'='*60}")
    print(f"CC Nipple Detection Training - CRAUNet")
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"{'='*60}\n")

    unified_df = preprocess_data(config)
    dataloaders = create_dataloaders(unified_df, config)

    train_loader, val_loader = dataloaders['Train'], dataloaders['Validation']
    
    # Checkpoint directory
    checkpoint_dir = '../checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # CC only predicts nipple coordinates (x, y) - 2 output features
    out_features = 2

    # Instantiate CRAUNet model
    model = CRAUNet(in_channels=1, out_features=out_features)
    model = model.to(config['device'])
    
    print(f"Model: CRAUNet - Output features: {out_features}")
    
    trainer = Trainer(config, train_loader, model)
    validator = Validator(config, val_loader, model)

    best_model_state = None
    best_val_loss = float('inf')
    metrics = []
    start_epoch = 0
    
    # Check for existing checkpoint to resume training
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        trainer.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state'])
        trainer.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state'])
        trainer.current_step = checkpoint['current_step']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        metrics = checkpoint['metrics']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config['num_epochs']):
        try:
            train_loss = trainer.train(epoch)
            validation_loss, is_best = validator.validate()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            # Safety check: detect training explosion
            if train_loss > 1e6 or validation_loss > 1e6:
                print(f"\nCRITICAL ERROR: Loss explosion at epoch {epoch+1}")
                break
            
            # Check for NaN/Inf in losses
            if train_loss != train_loss or validation_loss != validation_loss:
                print(f"\nERROR: NaN in loss at epoch {epoch+1}. Stopping.")
                break

            metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'learning_rate': current_lr
            }) 

            if is_best:
                best_model_state = model.state_dict()
                best_val_loss = validation_loss
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': trainer.optimizer.state_dict(),
                    'warmup_scheduler_state': trainer.warmup_scheduler.state_dict(),
                    'cosine_scheduler_state': trainer.cosine_scheduler.state_dict(),
                    'current_step': trainer.current_step,
                    'best_val_loss': best_val_loss,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"  Checkpoint saved at epoch {epoch+1}")
            
            # Always save latest checkpoint (for resume)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'warmup_scheduler_state': trainer.warmup_scheduler.state_dict(),
                'cosine_scheduler_state': trainer.cosine_scheduler.state_dict(),
                'current_step': trainer.current_step,
                'best_val_loss': best_val_loss,
                'metrics': metrics
            }, latest_checkpoint)
            
        except RuntimeError as e:
            print(f"\nRuntimeError at epoch {epoch+1}: {str(e)}")
            break
        except Exception as e:
            print(f"\nUnexpected error at epoch {epoch+1}: {str(e)}")
            break

    if best_model_state:
        os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
        torch.save(best_model_state, config['best_model_path'])
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best model saved: {config['best_model_path']}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"{'='*60}")

    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(os.path.dirname(config['best_model_path']), '..', 'cc_training_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CC Nipple Detection Training")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
