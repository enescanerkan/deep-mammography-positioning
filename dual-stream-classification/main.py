"""
Dual-stream mammography quality classification training script.

Usage:
    python main.py --model resnet18
    python main.py --model efficientnet_b0
    python main.py --model mobilenet_v2
    python main.py --model resnet50
"""

import argparse
import os

import pandas as pd
import torch

from model_configs import get_model_config, print_model_info, get_available_models
from utils import (
    get_dual_dataloader,
    get_dual_model,
    DualTrainer,
    DualValidator,
    EarlyStopping,
)


class TrainingPipeline:
    """Orchestrates the training process for dual-stream classification."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        self.model = None
        self.trainer = None
        self.validator = None
        self.early_stopping = None
        self.scheduler = None
        self.metrics = []
        self.best_state = {
            'model_state': None,
            'val_loss': float('inf'),
            'val_f1': 0.0,
            'epoch': 0
        }
    
    def setup(self) -> bool:
        """Initialize all components."""
        self._print_header()
        
        train_loader = self._create_dataloader('Train', mirror_bad=True, weighted_sampler=True)
        val_loader = self._create_dataloader('Validation', mirror_bad=False, weighted_sampler=False)
        
        if not train_loader or not val_loader:
            print("[ERROR] Failed to load data.")
            return False
        
        print(f"[OK] Train batches: {len(train_loader)}")
        print(f"[OK] Validation batches: {len(val_loader)}")
        
        self._create_model()
        self.trainer = DualTrainer(self.config, train_loader, self.model)
        self.validator = DualValidator(self.config, val_loader, self.model)
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 15),
            min_delta=0.001,
            mode='max',
            verbose=True
        )
        self._setup_scheduler()
        return True
    
    def _print_header(self) -> None:
        """Print training configuration header."""
        print("=" * 60)
        print("DUAL-STREAM MAMMOGRAPHY QUALITY CLASSIFICATION")
        print("=" * 60)
        print(f"Device:     {self.device}")
        print(f"Backbone:   {self.config.get('backbone', 'resnet18')}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Epochs:     {self.config['num_epochs']}")
        print("=" * 60)
    
    def _create_dataloader(self, split: str, mirror_bad: bool, weighted_sampler: bool):
        """Create dataloader for specified split."""
        use_aug = self.config.get('use_augmentation', True) if split == 'Train' else False
        return get_dual_dataloader(
            mlo_dir=self.config['mlo_dir'],
            cc_dir=self.config['cc_dir'],
            mlo_labels_csv=self.config['mlo_labels'],
            cc_labels_csv=self.config['cc_labels'],
            split_type=split,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 0),
            mirror_bad=mirror_bad,
            weighted_sampler=weighted_sampler,
            use_augmentation=use_aug
        )
    
    def _create_model(self) -> None:
        """Initialize the model."""
        print("\n[MODEL] Creating model...")
        self.model = get_dual_model(
            model_type='dual',
            num_classes=self.config['num_classes'],
            fusion_method=self.config['fusion_method'],
            pretrained=self.config['pretrained'],
            backbone=self.config.get('backbone', 'resnet18'),
            dropout_rate1=self.config.get('dropout_rate1', 0.5),
            dropout_rate2=self.config.get('dropout_rate2', 0.4)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[OK] Total parameters: {total_params:,}")
        print(f"[OK] Trainable parameters: {trainable_params:,}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.trainer.optimizer,
            mode='min',
            factor=self.config.get('lr_scheduler_factor', 0.5),
            patience=self.config.get('lr_scheduler_patience', 5),
            min_lr=self.config.get('min_lr', 1e-7),
            verbose=True
        )
    
    def train(self) -> None:
        """Execute training loop."""
        print(f"\n[TRAIN] Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            train_metrics = self.trainer.train(epoch)
            val_metrics = self.validator.validate()
            
            self._log_epoch(epoch, train_metrics, val_metrics)
            self._update_best_model(epoch, val_metrics)
            
            self.scheduler.step(val_metrics[0])
            
            if self.early_stopping(epoch + 1, val_metrics[1]):
                print(f"\n[STOP] Early stopping at epoch {epoch + 1}")
                if self.best_state['model_state']:
                    self.model.load_state_dict(self.best_state['model_state'])
                break
        
        self._save_results()
    
    def _log_epoch(self, epoch: int, train_metrics: tuple, val_metrics: tuple) -> None:
        """Log epoch metrics."""
        train_loss, train_f1, train_acc, train_prec, train_sens, train_spec, train_auc = train_metrics
        val_loss, val_f1, val_acc, val_prec, val_sens, val_spec, val_auc, _ = val_metrics
        
        accuracy_gap = train_acc - val_acc
        loss_gap = val_loss - train_loss
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        
        self.metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_f1': train_f1, 'val_f1': val_f1,
            'train_accuracy': train_acc, 'val_accuracy': val_acc,
            'train_precision': train_prec, 'val_precision': val_prec,
            'train_sensitivity': train_sens, 'val_sensitivity': val_sens,
            'train_specificity': train_spec, 'val_specificity': val_spec,
            'train_auc': train_auc, 'val_auc': val_auc,
            'accuracy_gap': accuracy_gap, 'loss_gap': loss_gap,
            'learning_rate': current_lr
        })
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
        print(f"Gap   - Acc: {accuracy_gap*100:.2f}% | LR: {current_lr:.2e}")
    
    def _update_best_model(self, epoch: int, val_metrics: tuple) -> None:
        """Update best model if validation improved."""
        val_loss, val_f1, _, _, _, _, _, is_best = val_metrics
        
        if is_best:
            self.best_state = {
                'model_state': self.model.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'epoch': epoch + 1
            }
            print(f"[BEST] New best model (Epoch {epoch + 1}, F1: {val_f1:.4f})")
    
    def _save_results(self) -> None:
        """Save model and metrics."""
        if self.best_state['model_state']:
            os.makedirs(os.path.dirname(self.config['best_model_path']), exist_ok=True)
            torch.save(self.best_state['model_state'], self.config['best_model_path'])
            print(f"\n[SAVE] Model saved: {self.config['best_model_path']}")
            print(f"   Best Epoch: {self.best_state['epoch']}")
            print(f"   Best F1: {self.best_state['val_f1']:.4f}")
        
        metrics_df = pd.DataFrame(self.metrics)
        os.makedirs(os.path.dirname(self.config['metrics_path']), exist_ok=True)
        metrics_df.to_csv(self.config['metrics_path'], index=False)
        print(f"[SAVE] Metrics saved: {self.config['metrics_path']}")
        
        self._print_summary(metrics_df)
    
    def _print_summary(self, metrics_df: pd.DataFrame) -> None:
        """Print training summary."""
        final_gap = metrics_df['accuracy_gap'].iloc[-1]
        print(f"\n[SUMMARY] Final accuracy gap: {final_gap*100:.2f}%")
        print("[DONE] Training completed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Dual-Stream Mammography Classification')
    parser.add_argument(
        '--model', 
        type=str, 
        default='resnet18',
        choices=get_available_models(),
        help='Model backbone'
    )
    args = parser.parse_args()
    
    print_model_info(args.model)
    
    config = get_model_config(args.model)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = TrainingPipeline(config)
    if pipeline.setup():
        pipeline.train()


if __name__ == "__main__":
    main()
