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

Trained model saved to: `code/models/mlo_model.pth`

## Training Configuration

Edit `code/regression/main/configs/example_config.json`:

```json
{
    "model_type": "CRAUNet",
    "batch_size": 8,
    "num_epochs": 300,
    "learning_rate": 1e-4,
    "target_task": "all",
    "w": 3,
    "epsilon": 1.5,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0
}
```

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate |
| Learning Rate | 1e-4 | Initial LR with warmup |
| Scheduler | Cosine + Warmup | 5 epoch warmup, then cosine annealing |
| Batch Size | 8 | Adjust based on GPU memory |
| Epochs | 300 | Full training |
| Weight Decay | 1e-5 | L2 regularization |

### Loss Function

**Multifaceted Wing Loss**:
- `alpha`: Weight for pec1 coordinate loss
- `beta`: Weight for pec2 coordinate loss
- `gamma`: Weight for nipple coordinate loss

## Data Requirements

### Input Format
- **Images**: Preprocessed .npy files (512x512, grayscale, normalized [0, 1])
- **Labels**: CSV file with landmark coordinates
- **Quality Filter**: Only "Good" quality images for training/validation

## Model Usage

```python
import torch
from utils.models import CRAUNet

# Load model
model = CRAUNet(in_channels=1, out_features=6)
model.load_state_dict(torch.load('code/models/mlo_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 1, 512, 512)  # Example input
    landmarks = model(image)  # Output: [batch, 6]
    
    # Denormalize (multiply by 512)
    landmarks = landmarks * 512
    
    # Extract landmarks
    pec1 = (landmarks[0, 0], landmarks[0, 1])
    pec2 = (landmarks[0, 2], landmarks[0, 3])
    nipple = (landmarks[0, 4], landmarks[0, 5])
```

## Directory Structure

```
mlo-landmark-detection/
├── README.md
├── requirements.txt
├── code/
│   ├── models/
│   │   ├── mlo_model.pth           # Trained model (after training)
│   │   └── checkpoints/            # Training checkpoints
│   └── regression/
│       ├── main/
│       │   ├── main.py             # Training script
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
└── landmark_coords/
```

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```json
"batch_size": 4
```

### Training too slow
Check if GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

## Notes

- Only "Good" quality MLO images are used for training
- Test set includes both Good and Bad for comprehensive evaluation
- Model outputs normalized coordinates [0, 1]
- Use pixel spacing from DICOM for mm distance calculations
