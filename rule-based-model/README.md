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
├── mlo-landmark-detection/              # MLO landmark detection
│   ├── code/regression/
│   │   ├── preprocessing/               # Preprocessing scripts
│   │   │   ├── generate_landmarks.py    # Create landmark JSONs from labels
│   │   │   └── crop_pad_resize_dataset.py  # Create 512x512 NPYs
│   │   └── main/                        # Training code
│   │       ├── main.py                  # Entry point
│   │       └── configs/example_config.json
│   ├── models/                          # Trained model weights
│   │   └── mlo_model.pth
│   └── landmark_coords/                 # Landmark JSON files
│
├── cc-landmark-detection/               # CC landmark detection
│   ├── code/regression/
│   │   ├── preprocessing/               # Preprocessing scripts
│   │   │   ├── generate_cc_landmarks.py # Create nipple landmark JSONs
│   │   │   └── crop_pad_resize_cc_dataset.py  # Create 512x512 NPYs
│   │   └── main/                        # Training code
│   │       ├── main.py                  # Entry point
│   │       └── configs/cc_training_config.json
│   ├── models/                          # Trained model weights
│   │   └── cc_model.pth
│   └── landmark_coords/                 # Landmark JSON files
│
└── test-models/                         # Evaluation pipeline
    └── mlo-cc-test/                     # Combined MLO+CC testing
```

## Workflow

### Prerequisites

Raw DICOM files should be in:
```
data/raw/mlo/   →  MLO DICOM files (.dicom)
data/raw/cc/    →  CC DICOM files (.dicom)
```

### 1. Generate Landmarks

First, extract landmark coordinates from label CSV:

```bash
# MLO: pectoral line + nipple
cd mlo-landmark-detection/code/regression/preprocessing
python generate_landmarks.py

# CC: nipple only
cd ../../cc-landmark-detection/code/regression/preprocessing
python generate_cc_landmarks.py
```

### 2. Preprocess Images

Create 512x512 normalized NPY files:

```bash
# MLO preprocessing
cd mlo-landmark-detection/code/regression/preprocessing
python crop_pad_resize_dataset.py

# CC preprocessing
cd ../../cc-landmark-detection/code/regression/preprocessing
python crop_pad_resize_cc_dataset.py
```

**Output**: `data/processed/mlo/images/` and `data/processed/cc/images/`

### 3. Train MLO Model

```bash
cd mlo-landmark-detection/code/regression/main
python main.py --config configs/example_config.json
```

Detects 3 landmarks (6 coordinates):
- Pectoral line point 1: (x1, y1)
- Pectoral line point 2: (x2, y2)
- Nipple center: (x, y)

### 4. Train CC Model

```bash
cd cc-landmark-detection/code/regression/main
python main.py --config configs/cc_training_config.json
```

Detects 1 landmark (2 coordinates):
- Nipple center: (x, y)

### 5. Evaluate Quality

```bash
cd test-models/mlo-cc-test
python separate_mlo_evaluator.py
python separate_cc_evaluator.py
python combined_quality_evaluator.py
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

## Data Flow

```
data/raw/mlo/           →  preprocessing  →  data/processed/mlo/images/
     ↓                                              ↓
labels/mlo_labels.csv   →  landmarks  →  landmark_coords/*.json
                                                    ↓
                                         Training (CRAUNet)
                                                    ↓
                                         models/mlo_model.pth
```

## Key Features

- **Centralized Data**: All raw/processed data in project root `data/` directory
- **Automatic Path Resolution**: No manual path configuration needed
- **Portable**: Works on any system after cloning
- **Modular**: Independent training of MLO and CC models

## Model Outputs

| Model | Output Location |
|-------|-----------------|
| MLO | `mlo-landmark-detection/models/mlo_model.pth` |
| CC | `cc-landmark-detection/models/cc_model.pth` |

## Notes

- MLO model trains on **Good quality images only** (consistent pectoral line)
- CC model trains on **all quality levels** (nipple always visible)
- Test set includes both Good and Bad for comprehensive evaluation
- Preprocessing uses the same images for both rule-based and classification approaches
