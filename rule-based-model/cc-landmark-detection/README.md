# CC Landmark Detection Model Training

Training pipeline for detecting anatomical landmarks in CC (Cranio-Caudal) mammography images.

## Overview

This module trains a CRAUNet model to detect the nipple center in CC views.

**Output**: 2 normalized coordinates (nipple_x, nipple_y)

## Architecture

**Model**: CRAUNet (Coordinate-aware Residual Attention UNet)

Same architecture as MLO model, but:
- Output dimension: 2 (nipple only)
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

Trained model automatically saved to: `../../models/CC_CRAUNet_model.pth`

## Training Configuration

Edit `code/regression/main/configs/cc_training_config.json`:

```json
{
    "model_type": "CRAUNet",
    "batch_size": 16,
    "num_epochs": 300,
    "learning_rate": 1e-4,
    "loss_function": {
        "w": 3,
        "epsilon": 1.5,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0
    },
    "model_architecture": {
        "target_task": "nipple"
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
- `alpha=1.0`: Weight for nipple_x coordinate loss
- `beta=1.0`: Weight for nipple_y coordinate loss
- `gamma=1.0`: Overall multiplier

Equal weighting for x and y coordinates.

## Data Requirements

### Input Format
- **Images**: Preprocessed .npy files (grayscale, normalized [0, 1])
- **Labels**: CSV file with nipple coordinates
- **Quality Filter**: None (uses both Good and Bad quality images)

### Preprocessing

Run preprocessing pipeline:

```bash
cd code/regression/preprocessing
python generate_cc_landmarks.py
python crop_pad_resize_cc_dataset.py
```

## Model Usage

```python
import torch
from utils.models import create_model

# Load model
model = create_model(in_channels=1, out_features=2, dropout_rate=0.3)
model.load_state_dict(torch.load('../../models/CC_CRAUNet_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 1, 512, 512)  # Example input
    landmarks = model(image)  # Output: [batch, 2]
    
    # Extract nipple coordinates
    nipple_x, nipple_y = landmarks[0, 0], landmarks[0, 1]
```

## Directory Structure

```
cc-landmark-detection/
├── README.md                        # This file
├── requirements.txt
├── code/
│   └── regression/
│       ├── main/
│       │   ├── main.py              # Training script
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
│           ├── generate_cc_landmarks.py
│           └── crop_pad_resize_cc_dataset.py
├── processed_data/
│   ├── images/                      # Preprocessed .npy images
│   └── transformation_details.csv
├── landmark_coords/                 # Generated landmarks
└── models/
    └── CC_CRAUNet_model.pth        # Trained model (after training)
```

## Differences from MLO Model

| Aspect | MLO | CC |
|--------|-----|-----|
| Landmarks | Pec line (2 pts) + Nipple | Nipple only |
| Output Size | 6 coordinates | 2 coordinates |
| Training Data | Good quality only | All quality levels |
| Loss Weights | α=1.8, β=1.8, γ=0.6 | α=1.0, β=1.0 |
| Architecture | CRAUNet | Same CRAUNet |

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

- CC model uses **ALL quality images** (Good + Bad) for training
- This differs from MLO which uses only Good quality images
- Nipple detection is quality-independent
- Model outputs normalized coordinates [0, 1]
- Use pixel spacing from DICOM for mm distance calculations

## Next Steps

After training both MLO and CC models:
1. Proceed to evaluation: `cd ../quality-evaluation/`
2. Test on DICOM images with clinical quality rules
