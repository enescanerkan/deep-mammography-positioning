"""
Validation module for dual-stream mammography classification.
"""

from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, roc_auc_score, accuracy_score

from utils.loss import CategoricalCrossEntropyLoss
from utils.metrics import calculate_sensitivity_specificity


class DualValidator:
    """Validator for dual-stream mammography classification model."""
    
    def __init__(self, config: dict, val_loader: DataLoader, model: nn.Module):
        self.config = config
        self.val_loader = val_loader
        self.device = config['device']
        self.model = model.to(self.device)
        self.criterion = CategoricalCrossEntropyLoss(class_weights=None).to(self.device)
        self.best_val_f1 = 0.0
    
    def validate(self) -> Tuple[float, float, float, float, float, float, float, bool]:
        """
        Validate model on validation set.
        
        Returns:
            Tuple of (loss, f1, accuracy, precision, sensitivity, specificity, auc, is_best)
        """
        self.model.eval()
        total_loss = 0.0
        all_targets, all_outputs = [], []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            
            for mlo_images, cc_images, targets, _ in progress_bar:
                mlo_images = mlo_images.to(self.device)
                cc_images = cc_images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(mlo_images, cc_images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(predicted.cpu().numpy())
        
        metrics = self._compute_metrics(total_loss, all_targets, all_outputs)
        
        is_best = metrics[1] > self.best_val_f1
        if is_best:
            self.best_val_f1 = metrics[1]
        
        return (*metrics, is_best)
    
    def _compute_metrics(
        self,
        total_loss: float,
        targets: list,
        predictions: list
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Compute validation metrics."""
        avg_loss = total_loss / len(self.val_loader)
        
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
