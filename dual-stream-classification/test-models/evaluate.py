#!/usr/bin/env python3
"""
Dual-Stream Model Evaluation Script
====================================
Evaluate trained dual-stream classification models on test set.

Usage:
    python evaluate.py --config configs/evaluate_config.json --model resnet18
    python evaluate.py --config configs/evaluate_config.json --model efficientnet_b0
    python evaluate.py --config configs/evaluate_config.json --model mobilenet_v2
    python evaluate.py --config configs/evaluate_config.json --model resnet50

The script will:
1. Load the trained model (based on backbone architecture)
2. Evaluate on test set
3. Generate confusion matrix, ROC curves, and metrics
4. Save predictions to CSV
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
DUAL_STREAM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DUAL_STREAM_DIR.parent
sys.path.insert(0, str(DUAL_STREAM_DIR))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score
)
from tqdm import tqdm

from utils.dual_dataloader import get_dual_dataloader
from utils.dual_models import get_dual_model


class Config:
    """Configuration loader and path resolver."""
    
    def __init__(self, config_path: str, backbone: str = None):
        with open(config_path, 'r') as f:
            self.raw_config = json.load(f)
        
        self.config_dir = Path(config_path).parent
        self.backbone = backbone or self.raw_config['model']['backbone']
        
        # Resolve paths relative to config file location
        self._resolve_paths()
        
    def _resolve_paths(self):
        """Resolve all relative paths."""
        data = self.raw_config['data']
        
        self.mlo_dir = str((self.config_dir / data['mlo_dir']).resolve())
        self.cc_dir = str((self.config_dir / data['cc_dir']).resolve())
        self.mlo_labels = str((self.config_dir / data['mlo_labels']).resolve())
        self.cc_labels = str((self.config_dir / data['cc_labels']).resolve())
        
        model = self.raw_config['model']
        # Use template with backbone name for dynamic model path
        model_path_template = model.get('model_path_template', model.get('model_path', ''))
        model_path_resolved = model_path_template.format(backbone=self.backbone)
        self.model_path = str((self.config_dir / model_path_resolved).resolve())
        self.fusion_method = model['fusion_method']
        self.num_classes = model['num_classes']
        
        eval_cfg = self.raw_config['evaluation']
        self.batch_size = eval_cfg['batch_size']
        self.threshold = eval_cfg['threshold']
        self.num_workers = eval_cfg['num_workers']
        self.device = eval_cfg['device']
        
        output = self.raw_config['output']
        self.results_dir = str((self.config_dir / output['results_dir']).resolve())
        self.save_predictions = output['save_predictions']
        self.save_visualizations = output['save_visualizations']
        
        # Get backbone-specific dropout rates
        backbone_cfg = self.raw_config['backbones'].get(self.backbone, {})
        self.dropout_rate1 = backbone_cfg.get('dropout_rate1', 0.5)
        self.dropout_rate2 = backbone_cfg.get('dropout_rate2', 0.4)
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("EVALUATION CONFIGURATION")
        print("=" * 60)
        print(f"Backbone:     {self.backbone}")
        print(f"Model Path:   {self.model_path}")
        print(f"MLO Dir:      {self.mlo_dir}")
        print(f"CC Dir:       {self.cc_dir}")
        print(f"Batch Size:   {self.batch_size}")
        print(f"Threshold:    {self.threshold}")
        print(f"Device:       {self.device}")
        print(f"Results Dir:  {self.results_dir}")
        print("=" * 60)


class DualModelEvaluator:
    """Evaluator for dual-stream mammography classification model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Create results directory
        self.results_dir = Path(config.results_dir) / config.backbone
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # Load test data
        self._load_data()
        
        # Load MLO labels for filtering
        self._load_mlo_labels()
        
        # Storage for results
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
        self.all_metadata = []
    
    def _load_model(self):
        """Load trained model."""
        print(f"\n[MODEL] Loading {self.config.backbone} model...")
        
        self.model = get_dual_model(
            model_type='dual',
            num_classes=self.config.num_classes,
            fusion_method=self.config.fusion_method,
            pretrained=False,
            backbone=self.config.backbone,
            dropout_rate1=self.config.dropout_rate1,
            dropout_rate2=self.config.dropout_rate2
        ).to(self.device)
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.config.model_path}\n"
                f"Please train the model first using: python main.py --model {self.config.backbone}"
            )
        
        checkpoint = torch.load(
            self.config.model_path, 
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[OK] Model loaded ({total_params:,} parameters)")
    
    def _load_data(self):
        """Load test dataloader."""
        print("\n[DATA] Loading test data...")
        
        self.test_loader = get_dual_dataloader(
            mlo_dir=self.config.mlo_dir,
            cc_dir=self.config.cc_dir,
            mlo_labels_csv=self.config.mlo_labels,
            cc_labels_csv=self.config.cc_labels,
            split_type='Test',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            mirror_bad=False,
            weighted_sampler=False,
            use_augmentation=False
        )
        print(f"[OK] Test batches: {len(self.test_loader)}")
    
    def _load_mlo_labels(self):
        """Load MLO labels for Good quality filtering."""
        print("[DATA] Loading MLO labels for filtering...")
        
        mlo_df = pd.read_csv(self.config.mlo_labels)
        mlo_df = mlo_df[mlo_df['Split'] == 'Test'].copy()
        mlo_df = mlo_df.drop_duplicates(subset=['SOPInstanceUID'], keep='first')
        self.mlo_label_map = dict(
            zip(mlo_df['SOPInstanceUID'], mlo_df['qualitativeLabel'])
        )
        print(f"[OK] MLO label mapping: {len(self.mlo_label_map)} samples")
    
    def evaluate(self):
        """Run evaluation on test set."""
        print(f"\n[EVAL] Running evaluation (threshold={self.config.threshold})...")
        
        self.model.eval()
        with torch.no_grad():
            for mlo_imgs, cc_imgs, labels, metadata in tqdm(
                self.test_loader, desc="Evaluating"
            ):
                mlo_imgs = mlo_imgs.to(self.device)
                cc_imgs = cc_imgs.to(self.device)
                
                outputs = self.model(mlo_imgs, cc_imgs)
                probs = F.softmax(outputs, dim=1)
                probs_good = probs[:, 1]
                predictions = (probs_good > self.config.threshold).long()
                
                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_labels.extend(labels.numpy())
                self.all_probs.extend(probs.cpu().numpy())
                
                for i in range(len(labels)):
                    meta = {
                        'study_id': metadata['study_id'][i] if isinstance(metadata, dict) else 'Unknown',
                        'side': metadata['side'][i] if isinstance(metadata, dict) else 'Unknown',
                        'mlo_sop': metadata['mlo_sop'][i] if isinstance(metadata, dict) else 'Unknown'
                    }
                    self.all_metadata.append(meta)
        
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)
        
        print(f"[OK] Total samples: {len(self.all_predictions)}")
        
        # Apply MLO-Good filter
        self._apply_mlo_filter()
    
    def _apply_mlo_filter(self):
        """Filter to only include samples where MLO is Good quality."""
        print("\n[FILTER] Applying MLO-Good filter...")
        
        mlo_good_mask = []
        for meta in self.all_metadata:
            mlo_sop = meta.get('mlo_sop', 'Unknown')
            mlo_label = self.mlo_label_map.get(mlo_sop, 'Unknown')
            is_good = (str(mlo_label).lower() == 'good')
            mlo_good_mask.append(is_good)
        
        mlo_good_mask = np.array(mlo_good_mask)
        self.all_predictions = self.all_predictions[mlo_good_mask]
        self.all_labels = self.all_labels[mlo_good_mask]
        self.all_probs = self.all_probs[mlo_good_mask]
        self.all_metadata = [m for m, keep in zip(self.all_metadata, mlo_good_mask) if keep]
        
        print(f"[OK] After MLO-Good filter: {len(self.all_predictions)} samples")
    
    def calculate_metrics(self) -> dict:
        """Calculate all evaluation metrics."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Overall metrics
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        precision = precision_score(self.all_labels, self.all_predictions, average='binary', zero_division=0)
        recall = recall_score(self.all_labels, self.all_predictions, average='binary', zero_division=0)
        f1 = f1_score(self.all_labels, self.all_predictions, average='binary', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        
        # ROC and PR AUC
        probs_good = self.all_probs[:, 1]
        fpr, tpr, _ = roc_curve(self.all_labels, probs_good)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(self.all_labels, probs_good)
        pr_auc = auc(recall_curve, precision_curve)
        
        metrics = {
            'backbone': self.config.backbone,
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc)
            },
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'per_class': {
                'bad': {
                    'precision': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
                    'recall': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
                    'f1_score': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0
                },
                'good': {
                    'precision': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0,
                    'recall': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,
                    'f1_score': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0
                }
            },
            'sample_info': {
                'total_samples': int(len(self.all_predictions)),
                'correct_predictions': int(np.sum(self.all_labels == self.all_predictions)),
                'incorrect_predictions': int(np.sum(self.all_labels != self.all_predictions))
            }
        }
        
        return metrics
    
    def plot_confusion_matrix(self):
        """Generate and save confusion matrix plot."""
        print("\n[PLOT] Generating confusion matrix...")
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad', 'Good'],
            yticklabels=['Bad', 'Good'],
            ax=ax,
            annot_kws={'fontsize': 16, 'fontweight': 'bold'}
        )
        ax.set_title(
            f'Confusion Matrix - {self.config.backbone}\n(n={len(self.all_predictions)})',
            fontsize=14, fontweight='bold'
        )
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] {save_path}")
    
    def plot_roc_curve(self):
        """Generate and save ROC and PR curves."""
        print("\n[PLOT] Generating ROC and PR curves...")
        
        probs_good = self.all_probs[:, 1]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(self.all_labels, probs_good)
        roc_auc = auc(fpr, tpr)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(self.all_labels, probs_good)
        pr_auc = auc(recall, precision)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC
        axes[0].plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        axes[0].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'ROC Curve - {self.config.backbone}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # PR
        axes[1].plot(recall, precision, 'green', lw=2, label=f'AUC = {pr_auc:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'Precision-Recall Curve - {self.config.backbone}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'roc_pr_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] {save_path}")
        print(f"ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
    
    def plot_prediction_distribution(self):
        """Generate and save prediction probability distribution."""
        print("\n[PLOT] Generating prediction distribution...")
        
        probs_good = self.all_probs[:, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(probs_good[self.all_labels == 0], bins=30, alpha=0.7, label='Bad', color='red')
        axes[0].hist(probs_good[self.all_labels == 1], bins=30, alpha=0.7, label='Good', color='green')
        axes[0].axvline(self.config.threshold, color='black', linestyle='--', lw=2, label='Threshold')
        axes[0].set_xlabel('Probability (Good)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Prediction Distribution - {self.config.backbone}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        bp = axes[1].boxplot(
            [probs_good[self.all_labels == 0], probs_good[self.all_labels == 1]],
            labels=['Bad', 'Good'],
            patch_artist=True
        )
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        axes[1].axhline(self.config.threshold, color='black', linestyle='--', lw=2)
        axes[1].set_ylabel('Probability (Good)')
        axes[1].set_title(f'Probability by Class - {self.config.backbone}')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.results_dir / 'prediction_distribution.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"[SAVED] {save_path}")
    
    def print_report(self) -> dict:
        """Print classification report and save metrics."""
        print("\n" + "=" * 60)
        print(f"CLASSIFICATION REPORT - {self.config.backbone.upper()}")
        print("=" * 60)
        
        report = classification_report(
            self.all_labels, self.all_predictions,
            target_names=['Bad', 'Good'],
            digits=4
        )
        print(report)
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print("=" * 60)
        
        # Calculate and save metrics
        metrics = self.calculate_metrics()
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics: dict):
        """Save metrics to JSON and TXT files."""
        # JSON format
        json_path = self.results_dir / 'metrics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[SAVED] {json_path}")
        
        # Human-readable TXT format
        txt_path = self.results_dir / 'metrics.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"MODEL EVALUATION METRICS - {self.config.backbone.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:     {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy'] * 100:.2f}%)\n")
            f.write(f"Precision:    {metrics['overall']['precision']:.4f}\n")
            f.write(f"Recall:       {metrics['overall']['recall']:.4f}\n")
            f.write(f"F1-Score:     {metrics['overall']['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:      {metrics['overall']['roc_auc']:.4f}\n")
            f.write(f"PR-AUC:       {metrics['overall']['pr_auc']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 60 + "\n")
            f.write(f"True Negative (TN):   {metrics['confusion_matrix']['true_negative']}\n")
            f.write(f"False Positive (FP):  {metrics['confusion_matrix']['false_positive']}\n")
            f.write(f"False Negative (FN):  {metrics['confusion_matrix']['false_negative']}\n")
            f.write(f"True Positive (TP):   {metrics['confusion_matrix']['true_positive']}\n\n")
            
            f.write("SAMPLE INFORMATION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Samples:         {metrics['sample_info']['total_samples']}\n")
            f.write(f"Correct Predictions:   {metrics['sample_info']['correct_predictions']}\n")
            f.write(f"Incorrect Predictions: {metrics['sample_info']['incorrect_predictions']}\n")
            f.write("=" * 60 + "\n")
        
        print(f"[SAVED] {txt_path}")
    
    def save_predictions(self):
        """Save individual predictions to CSV."""
        print("\n[SAVE] Exporting predictions to CSV...")
        
        mlo_labels_list = [
            str(self.mlo_label_map.get(m.get('mlo_sop', 'Unknown'), 'Unknown'))
            for m in self.all_metadata
        ]
        
        df = pd.DataFrame({
            'study_id': [m['study_id'] for m in self.all_metadata],
            'side': [m['side'] for m in self.all_metadata],
            'mlo_label': mlo_labels_list,
            'cc_true_label': ['Bad' if l == 0 else 'Good' for l in self.all_labels],
            'predicted': ['Bad' if p == 0 else 'Good' for p in self.all_predictions],
            'prob_bad': self.all_probs[:, 0],
            'prob_good': self.all_probs[:, 1],
            'correct': self.all_labels == self.all_predictions
        })
        
        csv_path = self.results_dir / 'predictions.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"[SAVED] {csv_path}")
        print(f"  Total: {len(df)}, Correct: {df['correct'].sum()} ({df['correct'].mean() * 100:.2f}%)")
    
    def run(self):
        """Run complete evaluation pipeline."""
        self.evaluate()
        metrics = self.print_report()
        
        if self.config.save_visualizations:
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_prediction_distribution()
        
        if self.config.save_predictions:
            self.save_predictions()
        
        return metrics


def get_available_backbones():
    """Return list of available backbone names."""
    return ['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate dual-stream mammography classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --config configs/evaluate_config.json --model resnet18
  python evaluate.py --config configs/evaluate_config.json --model efficientnet_b0
  python evaluate.py --config configs/evaluate_config.json --model mobilenet_v2
  python evaluate.py --config configs/evaluate_config.json --model resnet50
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation config JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=get_available_backbones(),
        required=True,
        help='Model backbone architecture'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Classification threshold (default: from config)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DUAL-STREAM MODEL EVALUATION")
    print("=" * 60)
    
    # Load configuration
    config = Config(args.config, backbone=args.model)
    
    # Override threshold if provided
    if args.threshold is not None:
        config.threshold = args.threshold
    
    config.print_config()
    
    # Run evaluation
    evaluator = DualModelEvaluator(config)
    metrics = evaluator.run()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.results_dir}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()

