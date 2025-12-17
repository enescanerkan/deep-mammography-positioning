# CC Landmark Detection Model Training

Training pipeline for detecting nipple position in CC (Cranio-Caudal) mammography images.

## Overview

This module trains a CRAUNet model to detect the nipple center in CC views.

**Output**: 2 normalized coordinates (nipple_x, nipple_y)

## Architecture

**Model**: CRAUNet (Coordinate-aware Residual Attention UNet)

Same base architecture as MLO model, but:
- Output dimension: 2 (nipple only)
- Uses global pooling regression head
- Training data: All quality levels (Good + Bad)

**Rationale**: Nipple detection is needed regardless of positioning quality for quality assessment.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
cd code/regression/main
python main.py --config configs/cc_training_config.json
```

### 3. Output

Trained model saved to: `code/models/cc_model.pth`

## Training Configuration

Edit `code/regression/main/configs/cc_training_config.json`:

```json
{
    "model_type": "CRAUNet",
    "batch_size": 8,
    "num_epochs": 300,
    "learning_rate": 1e-4,
    "target_task": "nipple",
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

**Wing Loss** for robust nipple coordinate regression.

## Data Requirements

### Input Format
- **Images**: Preprocessed .npy files (512x512, grayscale, normalized [0, 1])
- **Labels**: CSV file with nipple coordinates
- **Quality Filter**: None (uses both Good and Bad quality images)

## Model Usage

```python
import torch
from utils.models import CRAUNet

# Load model
model = CRAUNet(in_channels=1, out_features=2)
model.load_state_dict(torch.load('code/models/cc_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 1, 512, 512)  # Example input
    landmarks = model(image)  # Output: [batch, 2]
    
    # Denormalize (multiply by 512)
    landmarks = landmarks * 512
    
    # Extract nipple coordinates
    nipple_x, nipple_y = landmarks[0, 0], landmarks[0, 1]
```

## Directory Structure

```
cc-landmark-detection/
├── README.md
├── requirements.txt
├── code/
│   ├── models/
│   │   ├── cc_model.pth            # Trained model (after training)
│   │   └── checkpoints/            # Training checkpoints
│   └── regression/
│       ├── main/
│       │   ├── main.py             # Training script
│       │   ├── configs/
│       │   │   └── cc_training_config.json
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

## Differences from MLO Model

| Aspect | MLO | CC |
|--------|-----|-----|
| Landmarks | Pec line (2 pts) + Nipple | Nipple only |
| Output Size | 6 coordinates | 2 coordinates |
| Training Data | Good quality only | All quality levels |
| Regression Head | Spatial (16x16) | Global pooling |

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

- CC model uses **ALL quality images** (Good + Bad) for training
- This differs from MLO which uses only Good quality images
- Nipple detection is quality-independent
- Model outputs normalized coordinates [0, 1]
- Use pixel spacing from DICOM for mm distance calculations
