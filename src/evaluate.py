import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

def evaluate_baseline(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Experiment 1: Baseline Similarity Detection
    Tests whether the model can distinguish between similar and different clips.
    """
    # ROC Analysis
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    
    # Precision-Recall Analysis
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)
    
    # Confusion Matrix at 0.5 threshold
    pred_labels = (predictions > 0.5).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    
    return {
        'roc_auc': float(roc_auc),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'confusion_matrix': cm.tolist(),
        'average_precision': float(avg_precision)
    }

def analyze_thresholds(predictions: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """
    Experiment 2: Threshold-based Flagging
    Tests different thresholds for copyright violation detection.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    results = []
    
    for threshold in thresholds:
        pred_labels = (predictions > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'false_positive_rate': float(fpr),
            'metrics': {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }
        })
    
    return results

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    save_dir: str = 'results',
    device: str = 'cpu'
) -> Dict:
    """Run both evaluation experiments and save results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get predictions
    predictions, labels = [], []
    model.eval()
    
    with torch.no_grad():
        for x1, x2, label in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            pred = model(x1, x2).squeeze()
            predictions.extend(pred.cpu().numpy())
            labels.extend(label.squeeze().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Run both experiments
    baseline_results = evaluate_baseline(predictions, labels)
    threshold_results = analyze_thresholds(predictions, labels)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # ROC Curve (Experiment 1)
    plt.subplot(2, 2, 1)
    plt.plot(baseline_results['fpr'], baseline_results['tpr'])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve (AUC = {baseline_results["roc_auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # PR Curve (Experiment 1)
    plt.subplot(2, 2, 2)
    plt.plot(baseline_results['recall'], baseline_results['precision'])
    plt.title(f'PR Curve (AP = {baseline_results["average_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # Confusion Matrix (Experiment 1)
    plt.subplot(2, 2, 3)
    sns.heatmap(baseline_results['confusion_matrix'], annot=True, fmt='d')
    plt.title('Confusion Matrix (threshold=0.5)')
    
    # Threshold Analysis (Experiment 2)
    plt.subplot(2, 2, 4)
    thresholds = [r['threshold'] for r in threshold_results]
    plt.plot(thresholds, [r['precision'] for r in threshold_results], label='Precision')
    plt.plot(thresholds, [r['recall'] for r in threshold_results], label='Recall')
    plt.plot(thresholds, [r['false_positive_rate'] for r in threshold_results], label='FPR')
    plt.title('Threshold Analysis')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'evaluation_results.png')
    
    # Save results
    results = {
        'experiment1_baseline': baseline_results,
        'experiment2_thresholds': threshold_results
    }
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results