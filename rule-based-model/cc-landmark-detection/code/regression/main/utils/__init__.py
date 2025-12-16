"""
CC Nipple Detection Training Utilities

Exports:
    - CRAUNet: Coordinate-aware Residual Attention UNet model
    - create_model: Factory function for model creation
    - create_dataloaders, preprocess_data: Data loading utilities
    - Trainer, Validator: Training and validation classes
    - MultifacetedLoss, WingLoss: Loss functions
    - EarlyStopping: Early stopping callback
"""

from .models import CRAUNet, create_model
from .dataloader import create_dataloaders, preprocess_data
from .train import Trainer
from .validate import Validator
from .loss import MultifacetedLoss, WingLoss
from .early_stopping import EarlyStopping

__all__ = [
    'CRAUNet',
    'create_model',
    'create_dataloaders',
    'preprocess_data',
    'Trainer',
    'Validator',
    'MultifacetedLoss',
    'WingLoss',
    'EarlyStopping',
]