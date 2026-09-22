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
    'resnet50_radimagenet': {
        'backbone': 'resnet50_radimagenet',
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
    'convnext_tiny': {
        'backbone': 'convnext_tiny',
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

# Micro-batch used for gradient accumulation, per (backbone, resolution).
# The optimizer still steps once per full batch, so the effective batch size and
# learning rate are identical across backbones; this only bounds peak activation
# memory.
#
# Accumulation is exact for the loss, but BatchNorm normalizes over the
# micro-batch rather than the full batch. The BatchNorm backbones (ResNet,
# EfficientNet) are therefore left un-accumulated at 512 px, where they fit;
# ConvNeXt uses LayerNorm and is unaffected.
MICRO_BATCH: Dict[str, Dict[int, int]] = {
    'resnet18': {512: 32, 1024: 8},
    'resnet50': {512: 32, 1024: 8},
    'resnet50_radimagenet': {512: 32, 1024: 8},
    'efficientnet_b0': {512: 32, 1024: 8},
    'convnext_tiny': {512: 8, 1024: 2},
    'mobilenet_v2': {512: 32, 1024: 8},
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


def get_model_config(
    model_name: str = 'resnet18',
    fold: int = None,
    hparams_from: str = None,
    image_size: int = 512,
    learning_rate: float = None,
    augment: str = 'paper',
    normalize: str = 'none',
) -> Dict[str, Any]:
    """
    Get complete model configuration.

    Args:
        model_name: Model backbone name.
        fold: Cross-validation fold index. When given, labels are read from
            labels/folds/ and results are written per fold.
        hparams_from: Take every training hyperparameter from this model
            instead of the backbone's own entry, keeping only the backbone.
            Used to compare architectures under identical training settings.
        image_size: Input resolution. Anything other than 512 reads from the
            correspondingly suffixed preprocessed data directory.

    Returns:
        Complete configuration dictionary.
    """
    if model_name not in MODEL_CONFIGS:
        print(f"[WARNING] '{model_name}' not found. Using 'resnet18'.")
        model_name = 'resnet18'

    # Start with defaults, then override with model-specific settings
    config = DEFAULT_CONFIG.copy()
    config.update(MODEL_CONFIGS[hparams_from or model_name])
    config['backbone'] = MODEL_CONFIGS[model_name]['backbone']
    config['image_size'] = image_size
    micro = MICRO_BATCH.get(model_name, {}).get(image_size, config['batch_size'])
    config['micro_batch_size'] = min(micro, config['batch_size'])

    # Results are tagged so that runs under borrowed hyperparameters or at a
    # different resolution never overwrite the baseline run.
    run_name = model_name
    if hparams_from and hparams_from != model_name:
        run_name += f'_hp-{hparams_from}'
    if learning_rate is not None and learning_rate != config['learning_rate']:
        config['learning_rate'] = learning_rate
        run_name += f'_lr{learning_rate:.0e}'.replace('-0', '-')
    if augment != 'paper':
        config['augment_preset'] = augment
        run_name += f'_aug-{augment}'
    if normalize != 'none':
        config['normalize'] = normalize
        run_name += f'_norm-{normalize}'
    if image_size != 512:
        run_name += f'_{image_size}px'
        suffix = f'_{image_size}'
        config['mlo_dir'] = f'../data/processed{suffix}/mlo/images'
        config['cc_dir'] = f'../data/processed{suffix}/cc/images'

    config['run_name'] = run_name

    # Set model-specific paths
    if fold is None:
        config['best_model_path'] = f'results/{run_name}/dual_best_model.pth'
        config['metrics_path'] = f'results/{run_name}/dual_training_metrics.csv'
    else:
        config['mlo_labels'] = f'../labels/folds/mlo_fold{fold}.csv'
        config['cc_labels'] = f'../labels/folds/cc_fold{fold}.csv'
        # ../.. is the workspace root, alongside repo/ and drive/
        cv = f'../../cv_results/cls_{run_name}/fold{fold}'
        config['best_model_path'] = f'{cv}/dual_best_model.pth'
        config['metrics_path'] = f'{cv}/dual_training_metrics.csv'

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
