"""
Model configurations for dual-stream mammography quality classification.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    backbone: str
    dropout_rate1: float
    dropout_rate2: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_max_norm: float
    num_epochs: int
    patience: int
    label_smoothing: float
    mixup_alpha: float


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'resnet18': {
        'backbone': 'resnet18',
        'dropout_rate1': 0.4,
        'dropout_rate2': 0.3,
        'batch_size': 32,
        'learning_rate': 3e-5,
        'weight_decay': 1e-3,
        'gradient_clip_max_norm': 1.0,
        'num_epochs': 50,
        'patience': 15,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.2,
    },
    'resnet50': {
        'backbone': 'resnet50',
        'dropout_rate1': 0.5,
        'dropout_rate2': 0.4,
        'batch_size': 16,
        'learning_rate': 1.5e-5,
        'weight_decay': 3e-3,
        'gradient_clip_max_norm': 0.5,
        'num_epochs': 50,
        'patience': 12,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.1,
    },
    'efficientnet_b0': {
        'backbone': 'efficientnet_b0',
        'dropout_rate1': 0.5,
        'dropout_rate2': 0.4,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 2e-3,
        'gradient_clip_max_norm': 0.8,
        'num_epochs': 50,
        'patience': 15,
        'label_smoothing': 0.08,
        'mixup_alpha': 0.15,
    },
    'mobilenet_v2': {
        'backbone': 'mobilenet_v2',
        'dropout_rate1': 0.35,
        'dropout_rate2': 0.25,
        'batch_size': 16,
        'learning_rate': 4e-5,
        'weight_decay': 5e-4,
        'gradient_clip_max_norm': 1.2,
        'num_epochs': 50,
        'patience': 18,
        'label_smoothing': 0.03,
        'mixup_alpha': 0.25,
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    'model_type': 'dual',
    'fusion_method': 'concat',
    'num_classes': 2,
    'pretrained': True,
    'lr_scheduler_patience': 7,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-7,
    'freeze_epochs': 0,
    'use_class_weights': True,
    'use_augmentation': True,
    'loss_type': 'weighted_ce',
    # Data paths - preprocessed NPY images from centralized data directory
    'mlo_dir': '../data/processed/mlo/images',
    'cc_dir': '../data/processed/cc/images',
    'mlo_labels': '../labels/mlo_labels.csv',
    'cc_labels': '../labels/cc_labels.csv',
    'best_model_path': 'results/dual_best_model.pth',
    'metrics_path': 'results/dual_training_metrics.csv',
    'num_workers': 4,
}


def get_model_config(model_name: str = 'resnet18') -> Dict[str, Any]:
    """
    Get complete model configuration.
    
    Args:
        model_name: Model backbone name.
        
    Returns:
        Complete configuration dictionary.
    """
    if model_name not in MODEL_CONFIGS:
        print(f"[WARNING] '{model_name}' not found. Using 'resnet18'.")
        model_name = 'resnet18'
    
    # Start with defaults, then override with model-specific settings
    config = DEFAULT_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    
    # Set model-specific paths
    config['best_model_path'] = f'results/{model_name}/dual_best_model.pth'
    config['metrics_path'] = f'results/{model_name}/dual_training_metrics.csv'
    
    return config


def get_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def print_model_info(model_name: str = 'resnet18') -> None:
    """Print model configuration summary."""
    if model_name not in MODEL_CONFIGS:
        print(f"Model '{model_name}' not found.")
        return
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Batch Size:     {cfg['batch_size']}")
    print(f"Learning Rate:  {cfg['learning_rate']}")
    print(f"Epochs:         {cfg['num_epochs']}")
    print(f"Patience:       {cfg['patience']}")
    print(f"Dropout:        {cfg['dropout_rate1']}, {cfg['dropout_rate2']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Available models:", get_available_models())
    for model in get_available_models():
        print_model_info(model)
