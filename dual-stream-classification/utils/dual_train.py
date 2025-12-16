"""
Training module for dual-stream mammography classification.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, roc_auc_score, accuracy_score

from utils.loss import CategoricalCrossEntropyLoss, FocalLoss, CombinedLoss
from utils.metrics import calculate_sensitivity_specificity


class ClassWeightCalculator:
    """Calculates class weights using Effective Number of Samples method."""
    
    def __init__(self, beta: float = 0.9999):
        self.beta = beta
    
    def calculate(self, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
        """Calculate class weights from dataloader."""
        label_counts = {0: 0, 1: 0}
        
        for _, _, targets, _ in dataloader:
            for target in targets:
                label_counts[target.item()] += 1
        
        effective_num = {
            i: (1.0 - self.beta) / (1.0 - self.beta ** count)
            for i, count in label_counts.items()
        }
        
        weights_sum = sum(effective_num.values())
        weights = torch.tensor([
            effective_num[0] / weights_sum * 2,
            effective_num[1] / weights_sum * 2
        ], dtype=torch.float32)
        
        print(f"[INFO] Class counts: Bad={label_counts[0]}, Good={label_counts[1]}")
        print(f"[INFO] Class weights: Bad={weights[0]:.3f}, Good={weights[1]:.3f}")
        
        return weights.to(device)


class LossFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create(
        loss_type: str,
        class_weights: Optional[torch.Tensor],
        label_smoothing: float,
        device: torch.device
    ) -> nn.Module:
        """Create loss function based on type."""
        if loss_type == 'focal':
            alpha = [class_weights[0].item(), class_weights[1].item()] if class_weights is not None else None
            return FocalLoss(alpha=alpha, gamma=2.0).to(device)
        
        elif loss_type == 'combined':
            alpha = [class_weights[0].item(), class_weights[1].item()] if class_weights is not None else None
            return CombinedLoss(class_weights=class_weights, alpha=alpha, gamma=2.0).to(device)
        
        else:
            return CategoricalCrossEntropyLoss(
                class_weights=class_weights,
                label_smoothing=label_smoothing
            ).to(device)


class MixupAugmentation:
    """Mixup data augmentation."""
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        self.alpha = alpha
        self.probability = probability
    
    def should_apply(self) -> bool:
        return self.alpha > 0 and np.random.rand() < self.probability
    
    def apply(
        self,
        mlo_images: torch.Tensor,
        cc_images: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation."""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = mlo_images.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_mlo = lam * mlo_images + (1 - lam) * mlo_images[index]
        mixed_cc = lam * cc_images + (1 - lam) * cc_images[index]
        
        return mixed_mlo, mixed_cc, targets, targets[index], lam


class DualTrainer:
    """Trainer for dual-stream mammography classification model."""
    
    def __init__(self, config: dict, train_loader: DataLoader, model: nn.Module):
        self.config = config
        self.train_loader = train_loader
        self.device = config['device']
        self.model = model.to(self.device)
        
        self.criterion = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.mixup = MixupAugmentation(config.get('mixup_alpha', 0.0))
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        class_weights = None
        if self.config.get('use_class_weights', True):
            calculator = ClassWeightCalculator()
            class_weights = calculator.calculate(self.train_loader, self.device)
        
        return LossFactory.create(
            self.config.get('loss_type', 'weighted_ce'),
            class_weights,
            self.config.get('label_smoothing', 0.0),
            self.device
        )
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def _compute_loss(
        self,
        mlo_images: torch.Tensor,
        cc_images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss with optional mixup."""
        if self.mixup.should_apply():
            mixed_mlo, mixed_cc, targets_a, targets_b, lam = self.mixup.apply(
                mlo_images, cc_images, targets, self.device
            )
            outputs = self.model(mixed_mlo, mixed_cc)
            loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
        else:
            outputs = self.model(mlo_images, cc_images)
            loss = self.criterion(outputs, targets)
        
        return loss, outputs
    
    def train(self, epoch: int) -> Tuple[float, float, float, float, float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, f1, accuracy, precision, sensitivity, specificity, auc)
        """
        self.model.train()
        total_loss = 0.0
        all_targets, all_outputs = [], []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]}'
        )
        
        for mlo_images, cc_images, targets, _ in progress_bar:
            mlo_images = mlo_images.to(self.device)
            cc_images = cc_images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            loss, outputs = self._compute_loss(mlo_images, cc_images, targets)
            loss.backward()
            
            gradient_clip = self.config.get('gradient_clip_max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.detach().cpu().numpy())
            all_outputs.extend(predicted.detach().cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())
        
        return self._compute_metrics(total_loss, all_targets, all_outputs)
    
    def _compute_metrics(
        self,
        total_loss: float,
        targets: list,
        predictions: list
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Compute training metrics."""
        avg_loss = total_loss / len(self.train_loader)
        
        f1 = f1_score(targets, predictions, average='weighted')
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(targets, predictions)
        sensitivity, specificity = calculate_sensitivity_specificity(
            np.array(predictions), np.array(targets)
        )
        
        try:
            auc = roc_auc_score(targets, predictions, multi_class='ovo')
        except ValueError:
            auc = float('nan')
        
        return avg_loss, f1, accuracy, precision, sensitivity, specificity, auc
