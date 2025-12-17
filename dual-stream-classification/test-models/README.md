# Dual-Stream Model Evaluation

Evaluate trained dual-stream classification models on test set.

## Quick Start

```bash
# Evaluate ResNet-18 model
python evaluate.py --config configs/evaluate_config.json --model resnet18

# Evaluate other backbones
python evaluate.py --config configs/evaluate_config.json --model resnet50
python evaluate.py --config configs/evaluate_config.json --model efficientnet_b0
python evaluate.py --config configs/evaluate_config.json --model mobilenet_v2
```

## Available Models

| Backbone | Parameters | Description |
|----------|------------|-------------|
| `resnet18` | 11M | Default, balanced performance |
| `resnet50` | 23M | Higher accuracy, more computation |
| `efficientnet_b0` | 4M | Efficient, good accuracy |
| `mobilenet_v2` | 2M | Fastest, mobile-friendly |

## Configuration

Edit `configs/evaluate_config.json` to customize:

```json
{
    "data": {
        "mlo_dir": "../../data/processed/mlo/images",
        "cc_dir": "../../data/processed/cc/images",
        "mlo_labels": "../../labels/mlo_labels.csv",
        "cc_labels": "../../labels/cc_labels.csv"
    },
    "model": {
        "model_path": "../results/dual_best_model.pth"
    },
    "evaluation": {
        "batch_size": 16,
        "threshold": 0.5
    }
}
```

## Output

Results are saved to `evaluation_results/<backbone>/`:

```
evaluation_results/
└── resnet18/
    ├── confusion_matrix.png      # Confusion matrix visualization
    ├── roc_pr_curves.png         # ROC and PR curves
    ├── prediction_distribution.png  # Probability distributions
    ├── metrics.json              # All metrics in JSON format
    ├── metrics.txt               # Human-readable metrics
    └── predictions.csv           # Per-sample predictions
```

## Command Line Options

```bash
python evaluate.py --config <config_path> --model <backbone> [--threshold <value>]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--config` | Yes | Path to evaluation config JSON |
| `--model` | Yes | Backbone: resnet18, resnet50, efficientnet_b0, mobilenet_v2 |
| `--threshold` | No | Classification threshold (default: 0.5) |

## Prerequisites

1. **Trained model**: Run `python main.py --model <backbone>` first
2. **Preprocessed data**: In `data/processed/mlo/images/` and `data/processed/cc/images/`
3. **Labels**: In `labels/mlo_labels.csv` and `labels/cc_labels.csv`

## Metrics

The evaluation generates:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Confusion Matrix**: TP, TN, FP, FN counts






