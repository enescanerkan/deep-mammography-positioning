# Dual-Stream Mammography Quality Classification

End-to-end deep learning model for mammography positioning quality assessment using paired MLO and CC images.

## Overview

This module implements a dual-stream classification approach that directly predicts positioning quality without explicit landmark detection.

**Architecture**: Dual-stream CNN with separate encoders for MLO and CC images

**Output**: Binary classification (Good/Bad quality)

## Key Features

- Processes paired MLO-CC images simultaneously
- Multiple backbone options (ResNet-18/50, RadImageNet ResNet-50, ConvNeXt-Tiny, EfficientNet-B0, MobileNet-V2)
- Feature fusion strategies (concatenation, addition, attention)
- Handles class imbalance with weighted sampling
- Extensive data augmentation for mammography

## Model Architecture

```
MLO Image → MLO Encoder (ResNet) → Features_MLO ─┐
                                                   ├→ Fusion → FC Layers → Good/Bad
CC Image  → CC Encoder (ResNet)  → Features_CC  ─┘
```

### Fusion Methods

1. **Concatenation** (default): `concat[F_mlo, F_cc]`
2. **Addition**: `F_mlo + F_cc`
3. **Attention**: Learnable attention weights for each stream

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train with ResNet-18 (default, recommended)
python main.py --model resnet18

# Train with other backbones
python main.py --model resnet50
python main.py --model efficientnet_b0
python main.py --model mobilenet_v2

# Radiology-pretrained ResNet-50 (RadImageNet); weights: see ../weights/README.md
python main.py --model resnet50_radimagenet

# Architecture comparison under identical training settings:
# borrow every hyperparameter from the ResNet-18 entry, keep only the backbone
python main.py --model resnet50_radimagenet --hparams-from resnet18

# One cross-validation fold (labels read from ../labels/folds/, see ../make_folds.py)
python main.py --model resnet18 --fold 0
```

### Command-line options

| Option | Default | Meaning |
|--------|---------|---------|
| `--model` | `resnet18` | Backbone: `resnet18`, `resnet50`, `resnet50_radimagenet`, `convnext_tiny`, `efficientnet_b0`, `mobilenet_v2` |
| `--fold` | none | Cross-validation fold index; reads `../labels/folds/{mlo,cc}_fold{i}.csv` and writes results per fold |
| `--hparams-from` | none | Take all training hyperparameters from this model's entry, keeping only the chosen backbone |
| `--image-size` | `512` | Input resolution; other sizes read the correspondingly suffixed data directory |
| `--lr` | config value | Override the learning rate |
| `--augment` | `paper` | `paper` = published recipe; `domain` = wider brightness/contrast, gamma and chest-wall-anchored zoom |
| `--normalize` | `none` | `tissue` rescales each image so its tissue median is 0.35 |
| `--selection` | `f1` | Checkpoint criterion: weighted F1 (published) or `balanced` = (sensitivity + specificity) / 2 |

Runs that deviate from a backbone's own settings are tagged in the result name
(for example `resnet50_radimagenet_hp-resnet18`), so they never overwrite the
baseline run. Large backbones are trained with gradient accumulation
(`MICRO_BATCH` in `model_configs.py`) so the effective batch size stays as configured.

### Configuration

Edit `model_configs.py` to adjust hyperparameters:

```python
MODEL_CONFIGS = {
    'resnet18': {
        'backbone': 'resnet18',
        'dropout_rate1': 0.4,
        'dropout_rate2': 0.3,
        'batch_size': 32,
        'learning_rate': 3e-5,
        'weight_decay': 1e-3,
        'num_epochs': 50,
        'patience': 15,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.2,
    }
}
```

## Supported Backbones

| Backbone | Pretraining | Notes |
|----------|-------------|-------|
| ResNet-18 | ImageNet | default, published configuration |
| ResNet-50 | ImageNet | |
| ResNet-50 (`resnet50_radimagenet`) | RadImageNet | radiology-pretrained; weights downloaded separately, see `../weights/README.md` |
| ConvNeXt-Tiny | ImageNet | |
| EfficientNet-B0 | ImageNet | |
| MobileNet-V2 | ImageNet | |

All backbones take single-channel input: the pretrained RGB first convolution is
collapsed to one channel by averaging its filters.

## Training Parameters

### Optimizer
- **Type**: Adam
- **Weight Decay**: L2 regularization
- **Gradient Clipping**: Max norm = 1.0

### Learning Rate Scheduler
- **Type**: ReduceLROnPlateau
- **Mode**: Minimize validation loss
- **Factor**: 0.5 (halve LR on plateau)
- **Patience**: 7 epochs
- **Min LR**: 1e-7

### Regularization
- Dropout layers (0.4, 0.3)
- Label smoothing (ε=0.05)
- Mixup augmentation (α=0.2)
- Weight decay (1e-3)

### Early Stopping
- **Metric**: Validation F1-Score
- **Patience**: 15 epochs
- **Min Delta**: 0.001

## Data Augmentation

Conservative transformations preserving diagnostic features:

- Random rotation: ±10°
- Brightness adjustment: ±20%
- Contrast adjustment: ±20%
- Gaussian noise: σ=0.03
- Gaussian blur: σ=0.5
- Elastic deformation: α=50, σ=5
- Horizontal flip for Bad samples (class balancing)

## Loss Function

**Weighted Cross-Entropy Loss** with:
- Automatic class weight calculation (Effective Number of Samples)
- Label smoothing to prevent overconfidence
- Mixup data augmentation

## Data Requirements

### Input Format
- **MLO Images**: `.npy` files in `../data/processed/mlo/images/` (centralized)
- **CC Images**: `.npy` files in `../data/processed/cc/images/` (centralized)
- **Labels**: CSV files in `../labels/` directory

> **Note**: This module uses the centralized data structure at project root.
> Preprocessing is done in `rule-based-model/*/preprocessing/` and outputs to `data/processed/`.

### Label CSV Format

```csv
SOPInstanceUID,StudyInstanceUID,SeriesDescription,qualitativeLabel,Split
mlo_001,study001,L-MLO,Good,Train
cc_001,study001,L-CC,Good,Train
mlo_002,study002,R-MLO,Bad,Val
cc_002,study002,R-CC,Bad,Val
```

### Pairing Strategy
- MLO and CC images paired by `StudyInstanceUID` and `Side` (L/R)
- Quality label taken from CC image
- Only "Good" quality MLO images used
- Class balancing via weighted sampling and mirroring

## Output

### Trained Model
Saved to: `results/dual_best_model.pth`

### Training Metrics
Saved to: `results/dual_training_metrics.csv`

Includes per-epoch:
- Train/Val Loss
- Train/Val Accuracy, F1, Precision, Recall
- Train/Val Sensitivity, Specificity, AUC-ROC
- Accuracy Gap (overfitting indicator)
- Learning Rate

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| F1-Score | Primary metric for model selection |
| Accuracy | Overall classification accuracy |
| Precision | Positive predictive value |
| Recall (Sensitivity) | True positive rate |
| Specificity | True negative rate |
| AUC-ROC | Area under ROC curve |

## Usage Example

```python
from utils import get_dual_model
import torch

# Load trained model
model = get_dual_model(
    model_type='dual',
    backbone='resnet18',
    num_classes=2,
    fusion_method='concat',
    pretrained=False
)
model.load_state_dict(torch.load('results/dual_best_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    mlo_img = torch.randn(1, 1, 512, 512)  # Example
    cc_img = torch.randn(1, 1, 512, 512)
    
    output = model(mlo_img, cc_img)
    prediction = torch.argmax(output, dim=1)
    
    quality = "Good" if prediction == 1 else "Bad"
    print(f"Predicted Quality: {quality}")
```

## Directory Structure

```
dual-stream-classification/
├── README.md                        # This file
├── requirements.txt
├── main.py                          # Training entry point
├── model_configs.py                 # Hyperparameter configurations
├── utils/
│   ├── __init__.py
│   ├── dual_models.py               # Model architectures
│   ├── dual_dataloader.py           # Data loading and pairing
│   ├── dual_train.py                # Training loop
│   ├── dual_validate.py             # Validation loop
│   ├── augmentation.py              # Data augmentation
│   ├── loss.py                      # Loss functions
│   ├── metrics.py                   # Evaluation metrics
│   └── early_stopping.py            # Early stopping
├── ../weights/                      # Pretrained weights (not tracked; see ../weights/README.md)
├── test-models/                     # Evaluation scripts
│   └── evaluate_dual_model.py       # Main evaluation script
└── results/
    ├── dual_best_model.pth          # Best model weights
    └── dual_training_metrics.csv    # Training history

# Data is stored at project root (centralized):
../data/processed/
├── mlo/images/                      # Preprocessed MLO images (.npy)
└── cc/images/                       # Preprocessed CC images (.npy)
```

## Comparison with Rule-Based Approach

| Aspect | Dual-Stream Classification | Rule-Based (Landmark + Rules) |
|--------|---------------------------|---------------------------|
| Approach | End-to-end learning | Two-stage (detection + rules) |
| Interpretability | Black box | Clinical rules |
| Training Data | Paired images + labels | Images + landmark annotations |
| Inference Speed | Fast (single forward pass) | Slower (two models + rules) |
| Generalization | Depends on training data | Relies on clinical knowledge |

## Troubleshooting

### Out of Memory
- Reduce batch size in `model_configs.py`
- Use smaller backbone (MobileNet-V2)

### Poor Performance
- Check MLO-CC pairing (verify `StudyInstanceUID`)
- Verify label distribution (Good vs Bad)
- Increase training epochs
- Adjust class weights

### Overfitting
- Increase dropout rates
- Increase weight decay
- Use more data augmentation
- Reduce model capacity

## Notes

- Model trained on paired MLO-CC images
- Ground truth is CC image quality label
- MLO images filtered to "Good" quality only
- Test set should contain same patient pairs
- Inference requires both MLO and CC images

## Next Steps

After training:
1. Evaluate on test set
2. Compare with rule-based approach (`../rule-based-model/`)
3. Analyze failure cases
4. Consider ensemble with rule-based model
