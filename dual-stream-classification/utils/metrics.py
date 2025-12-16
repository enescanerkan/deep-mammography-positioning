"""
Evaluation metrics for classification tasks.
"""

from typing import Tuple, List, Optional
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def calculate_sensitivity_specificity(
    predictions: np.ndarray,
    truths: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate sensitivity and specificity for binary classification.
    
    Sensitivity (Recall) is calculated for 'Bad' cases (label 0).
    Specificity is calculated for 'Good' cases (label 1).
    
    Args:
        predictions: Predicted labels
        truths: Ground truth labels
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    tp = np.sum((predictions == 0) & (truths == 0))
    fn = np.sum((predictions == 1) & (truths == 0))
    tn = np.sum((predictions == 1) & (truths == 1))
    fp = np.sum((predictions == 0) & (truths == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def compute_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    num_classes: int
) -> Tuple[float, float, float, float, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted labels
        truths: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Tuple of (accuracy, f1, sensitivity, specificity, auc) as percentages
    """
    accuracy = accuracy_score(truths, predictions) * 100
    f1 = f1_score(truths, predictions, average='macro') * 100
    sensitivity, specificity = calculate_sensitivity_specificity(predictions, truths)
    sensitivity *= 100
    specificity *= 100
    
    try:
        auc = roc_auc_score(truths, predictions) * 100 if num_classes == 2 else float('nan')
    except ValueError:
        auc = float('nan')
    
    return accuracy, f1, sensitivity, specificity, auc


def save_metrics_csv(metrics: List[float], file_path: str) -> None:
    """Save metrics to CSV file."""
    fieldnames = ["Accuracy", "F1 Score", "Sensitivity", "Specificity", "AUC"]
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Accuracy": f"{metrics[0]:.2f}",
            "F1 Score": f"{metrics[1]:.2f}",
            "Sensitivity": f"{metrics[2]:.2f}",
            "Specificity": f"{metrics[3]:.2f}",
            "AUC": f"{metrics[4]:.2f}" if not np.isnan(metrics[4]) else "N/A"
        })


def save_predictions_csv(
    predictions: List[int],
    truths: List[int],
    study_uids: List[str],
    sop_uids: List[str],
    file_path: str
) -> None:
    """Save predictions to CSV file."""
    rows = [
        {
            "index": i + 1,
            "StudyInstanceUID": study_uids[i],
            "SOPInstanceUID": sop_uids[i],
            "Ground Truth": truths[i],
            "Prediction": predictions[i]
        }
        for i in range(len(predictions))
    ]
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "StudyInstanceUID", "SOPInstanceUID", "Ground Truth", "Prediction"]
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    metrics_csv: str,
    results_csv: str,
    annotation_path: str
) -> None:
    """
    Evaluate model on test set and save results.
    
    Args:
        model: PyTorch model
        loader: Test DataLoader
        device: Computation device
        num_classes: Number of classes
        metrics_csv: Path to save metrics
        results_csv: Path to save predictions
        annotation_path: Path to annotation CSV
    """
    model.eval()
    all_preds, all_labels = [], []
    
    annotations = pd.read_csv(annotation_path)
    test_annotations = annotations[annotations['Split'] == 'Test']
    study_uids = test_annotations['StudyInstanceUID'].tolist()
    sop_uids = test_annotations['SOPInstanceUID'].tolist()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels), num_classes)
    save_metrics_csv(list(metrics), metrics_csv)
    save_predictions_csv(all_preds, all_labels, study_uids, sop_uids, results_csv)
