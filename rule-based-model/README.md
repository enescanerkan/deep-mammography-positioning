# Rule-Based Mammography Quality Assessment

Landmark detection and clinical rule-based approach for automated mammography positioning quality evaluation.

## Overview

This module implements a **two-stage approach**:

1. **Stage 1 - Landmark Detection**: Train CRAUNet models to detect anatomical landmarks
   - MLO: Pectoral muscle line (2 points) + nipple (1 point) = 6 coordinates
   - CC: Nipple center only = 2 coordinates

2. **Stage 2 - Quality Assessment**: Apply clinical rules (10mm PNL rule) using detected landmarks

## Module Structure

```
rule-based-model/
│
├── mlo-landmark-detection/
│   ├── code/
│   │   ├── models/                      # Trained model weights
│   │   │   └── mlo_model.pth
│   │   └── regression/
│   │       ├── preprocessing/           # Preprocessing scripts
│   │       └── main/                    # Training code
│   │           ├── main.py
│   │           └── configs/example_config.json
│   └── landmark_coords/
│
├── cc-landmark-detection/
│   ├── code/
│   │   ├── models/                      # Trained model weights
│   │   │   └── cc_model.pth
│   │   └── regression/
│   │       ├── preprocessing/           # Preprocessing scripts
│   │       └── main/                    # Training code
│   │           ├── main.py
│   │           └── configs/cc_training_config.json
│   └── landmark_coords/
│
└── test-models/                         # Evaluation pipeline
    ├── run_full_evaluation.py
    └── images_dicom/
```

## Workflow

### 1. Train MLO Model

```bash
cd mlo-landmark-detection/code/regression/main
python main.py --config configs/example_config.json
```

Detects 3 landmarks (6 coordinates):
- Pectoral line point 1: (x1, y1)
- Pectoral line point 2: (x2, y2)
- Nipple center: (x, y)

**Output**: `mlo-landmark-detection/code/models/mlo_model.pth`

### 2. Train CC Model

```bash
cd cc-landmark-detection/code/regression/main
python main.py --config configs/cc_training_config.json
```

Detects 1 landmark (2 coordinates):
- Nipple center: (x, y)

**Output**: `cc-landmark-detection/code/models/cc_model.pth`

### 3. Evaluate Quality

```bash
cd test-models
python run_full_evaluation.py
```

## Clinical Quality Rules

### 10mm Rule

The primary quality criterion:

```
PNL_MLO = Perpendicular distance from nipple to pectoral line in MLO view
PNL_CC = Distance from nipple to chest wall in CC view

Quality = Good  if  |PNL_MLO - PNL_CC| < 10mm
Quality = Bad   if  |PNL_MLO - PNL_CC| ≥ 10mm
```

## Model Outputs

| Model | Output Location |
|-------|-----------------|
| MLO | `mlo-landmark-detection/code/models/mlo_model.pth` |
| CC | `cc-landmark-detection/code/models/cc_model.pth` |

## Key Features

- **CRAUNet Architecture**: Coordinate-aware Attention U-Net for precise landmark localization
- **Wing Loss**: Robust loss function for coordinate regression
- **Cosine Annealing + Warmup**: Stable training with learning rate scheduling
- **Quality Filtering**: MLO trains on Good quality only, CC trains on all

## Notes

- MLO model trains on **Good quality images only** (consistent pectoral line)
- CC model trains on **all quality levels** (nipple always visible)
- Test set includes both Good and Bad for comprehensive evaluation
- Model outputs normalized coordinates [0, 1], multiply by 512 for pixel coordinates
