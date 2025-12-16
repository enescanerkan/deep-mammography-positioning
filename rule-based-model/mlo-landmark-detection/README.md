# MLO Landmark Detection Model Training

Training pipeline for detecting anatomical landmarks in MLO (Medio-Lateral Oblique) mammography images.

## Overview

This module trains a CRAUNet model to detect three key anatomical landmarks in MLO views:
- **Pectoral muscle line**: 2 points defining the pectoral boundary
- **Nipple center**: 1 point marking nipple position

**Total output**: 6 normalized coordinates (x, y pairs for each point)

## Architecture

**Model**: CRAUNet (Coordinate-aware Residual Attention UNet)

**Key Features**:
- CoordConv for spatial awareness
- Attention gates for focusing on relevant regions
- U-Net encoder-decoder with skip connections
- Wing Loss for robust coordinate regression

**Parameters**: ~31M trainable parameters

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
cd code/regression/main
python main.py --config configs/example_config.json
```

### 3. Output

Trained model automatically saved to: `../../models/CRAUNet_model.pth`

## Training Configuration

Edit `code/regression/main/configs/example_config.json`:

```json
{
    "model_type": "CRAUNet",
    "batch_size": 16,
    "num_epochs": 300,
    "learning_rate": 1e-4,
    "loss_function": {
        "w": 3,
        "epsilon": 1.5,
        "alpha": 1.8,
        "beta": 1.8,
        "gamma": 0.6
    }
}
```

**Note**: All paths are relative - no manual editing needed!

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate |
| Learning Rate | 1e-4 | Initial LR with warmup |
| Scheduler | Cosine + Warmup | 5 epoch warmup, then cosine annealing |
| Batch Size | 16 | Adjust based on GPU memory |
| Epochs | 300 | With early stopping (patience=40) |
| Dropout | 0.3 | Regularization |
| Weight Decay | 1e-5 | L2 regularization |

### Loss Function

**Multifaceted Wing Loss**:
- `alpha=1.8`: Weight for pec1 coordinate loss
- `beta=1.8`: Weight for pec2 coordinate loss
- `gamma=0.6`: Weight for nipple coordinate loss

Pectoral muscle line has higher priority as it's critical for quality assessment.

## Data Requirements

### Input Format
- **Images**: Preprocessed .npy files (grayscale, normalized [0, 1])
- **Labels**: CSV file with landmark coordinates
- **Quality Filter**: Only "Good" quality images for training/validation

### Preprocessing

Run preprocessing pipeline:

```bash
cd code/regression/preprocessing
python generate_landmarks.py
python crop_pad_resize_dataset.py
```

## Model Usage

```python
import torch
from utils.models import create_model

# Load model
model = create_model(in_channels=1, out_features=6, dropout_rate=0.3)
model.load_state_dict(torch.load('../../models/CRAUNet_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 1, 512, 512)  # Example input
    landmarks = model(image)  # Output: [batch, 6]
    
    # Extract landmarks
    pec1 = (landmarks[0, 0], landmarks[0, 1])
    pec2 = (landmarks[0, 2], landmarks[0, 3])
    nipple = (landmarks[0, 4], landmarks[0, 5])
```

## Directory Structure

```
mlo-landmark-detection/
├── README.md                        # This file
├── requirements.txt
├── code/
│   └── regression/
│       ├── main/
│       │   ├── main.py              # Training script
│       │   ├── configs/
│       │   │   └── example_config.json
│       │   └── utils/
│       │       ├── models.py
│       │       ├── dataloader.py
│       │       ├── train.py
│       │       ├── validate.py
│       │       ├── loss.py
│       │       └── early_stopping.py
│       └── preprocessing/
│           ├── generate_landmarks.py
│           └── crop_pad_resize_dataset.py
├── processed_data/
│   ├── images/                      # Preprocessed .npy images
│   └── transformation_details.csv
├── landmark_coords/                 # Generated landmarks
└── models/
    └── CRAUNet_model.pth           # Trained model (after training)
```

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```json
"batch_size": 8
```

### Training too slow
Check if GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### NaN Loss
- Check data preprocessing
- Reduce learning rate
- Verify landmark coordinates are normalized [0, 1]

## Notes

- Only "Good" quality MLO images are used for training
- Test set includes both Good and Bad for comprehensive evaluation
- Model outputs normalized coordinates [0, 1]
- Use pixel spacing from DICOM for mm distance calculations

## Next Steps

After training MLO model:
1. Train CC model: `cd ../../cc-landmark-detection/`
2. Evaluate both models: `cd ../quality-evaluation/`
