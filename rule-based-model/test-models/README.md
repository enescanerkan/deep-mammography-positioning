# Mammography Quality Evaluation Pipeline

A comprehensive evaluation system for mammography image quality assessment using deep learning-based landmark detection. This pipeline evaluates both MLO (Mediolateral Oblique) and CC (Craniocaudal) views following SOLID principles and OOP best practices.

## Overview

This tool automatically:
- Detects anatomical landmarks in mammography images
- Calculates quality metrics based on landmark positions
- Generates visualizations with predicted vs ground truth comparisons
- Produces comprehensive evaluation reports

## Architecture

The system follows **SOLID principles** with a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    run_full_evaluation.py                   │
├─────────────────────────────────────────────────────────────┤
│  PathConfig          - Centralized path management (SRP)    │
│  ImagePreprocessor   - DICOM loading & preprocessing (SRP)  │
│  MLOCRAUNet         - MLO-specific model architecture       │
│  CCCRAUNet          - CC-specific model architecture        │
│  MLOEvaluator       - MLO landmark detection & metrics      │
│  CCEvaluator        - CC landmark detection & metrics       │
│  CombinedEvaluator  - Quality assessment & reporting        │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Single Responsibility (SRP)**: Each class has one clear purpose
- **Open/Closed (OCP)**: Easy to extend with new view types or metrics
- **Dependency Inversion (DIP)**: High-level modules depend on abstractions
- **Separation of Concerns**: Model, preprocessing, and evaluation logic are isolated

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pydicom>=2.2.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
Pillow>=8.0.0
scipy>=1.7.0
```

## Project Structure

```
mlo-cc-test/
├── run_full_evaluation.py    # Main evaluation script
├── images_dicom/             # DICOM images directory
├── evaluation_results/       # Output directory
│   ├── mlo_visualizations/   # MLO prediction visualizations
│   ├── cc_visualizations/    # CC prediction visualizations
│   └── combined_visualizations/  # Quality assessment plots
└── README.md
```

## Usage

### Basic Usage

```bash
python run_full_evaluation.py
```

### Expected Directory Structure

The script expects the following structure relative to the project root:

```
deep-mammography-positioning/
├── rule-based-model/
│   ├── mlo-landmark-detection/
│   │   └── models/
│   │       └── mlo_model.pth
│   ├── cc-landmark-detection/
│   │   └── models/
│   │       └── cc_model.pth
│   └── test-models/
│       └── mlo-cc-test/        # This directory
│           ├── run_full_evaluation.py
│           └── images_dicom/
└── labels/
    ├── mlo_labels.csv
    ├── cc_labels.csv
    └── metadata.csv
```

## Model Architectures

### MLO Model (CRAUNet - 6 outputs)
- **Purpose**: Detects 3 landmarks (Pectoral line endpoints + Nipple)
- **Output**: 6 values (x, y coordinates for each landmark)
- **Architecture**: Coordinate-aware Attention U-Net with spatial regression head

### CC Model (CRAUNet - 2 outputs)
- **Purpose**: Detects nipple position
- **Output**: 2 values (x, y coordinates)
- **Architecture**: Coordinate-aware Attention U-Net with global pooling head

## Output

The pipeline generates:

1. **Visualizations**: PNG images showing predicted landmarks overlaid on original DICOM
2. **JSON Results**: Detailed results for each processed image
3. **Metrics Report**: Accuracy, precision, recall, and confusion matrix
4. **Quality Assessment**: Combined MLO-CC quality evaluation

## Key Classes

| Class | Responsibility |
|-------|---------------|
| `PathConfig` | Manages all file paths and directory setup |
| `ImagePreprocessor` | DICOM loading, cropping, padding, resizing |
| `MLOCRAUNet` | MLO-specific neural network architecture |
| `CCCRAUNet` | CC-specific neural network architecture |
| `MLOEvaluator` | MLO landmark prediction and distance calculation |
| `CCEvaluator` | CC landmark prediction and distance calculation |
| `CombinedQualityEvaluator` | Combines results and generates reports |

## Quality Metrics

- **MLO Distance**: Perpendicular distance from nipple to pectoral line (mm)
- **CC Distance**: Distance from nipple to chest wall (mm)
- **10mm Rule**: Quality threshold based on clinical guidelines

## License

This project is part of the deep-mammography-positioning research project.

