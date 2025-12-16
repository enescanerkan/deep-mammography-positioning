"""
Loss functions for classification tasks.
"""

from typing import Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalCrossEntropyLoss(nn.Module):
    """Cross entropy loss with optional class weights and label smoothing."""
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(outputs, targets.long())


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        num_classes = outputs.shape[1]
        
        probs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=outputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined cross entropy and focal loss."""
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce = self.ce_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        return self.ce_weight * ce + self.focal_weight * focal
