import os
import logging
import warnings

# Set environment variables
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure logging before any imports
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Now do the rest of the imports
import numpy as np
import torch
import matplotlib
# Configure matplotlib before importing pyplot
matplotlib.use('Agg')  # Non-interactive backend
matplotlib.set_loglevel('WARNING')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure all matplotlib-related logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith('matplotlib'):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Set style with minimal output
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'axes.grid': True
})  # Removed invalid 'verbose.level' parameter

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    save_dir: str = 'results',
    device: str = 'cpu'
) -> Dict:
    """Evaluate model and generate metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Collect predictions
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
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)
    
    # Generate confusion matrix
    pred_labels = (predictions > 0.5).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    
    # Create plots with minimal output
    plt.ioff()  # Turn off interactive mode
    
    # Plot ROC curve
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    
    # Plot PR curve
    ax2.plot(recall, precision)
    ax2.set_title(f'Precision-Recall (AP = {avg_precision:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', ax=ax3)
    ax3.set_title('Confusion Matrix')
    
    # Plot score distribution
    ax4.hist(predictions[labels == 0], alpha=0.5, label='Negative')
    ax4.hist(predictions[labels == 1], alpha=0.5, label='Positive')
    ax4.set_title('Score Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'evaluation_results.png')
    plt.close()
    
    # Save metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'threshold_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

if __name__ == "__main__":
    import argparse
    from model import SiameseNetwork
    from dataset import PairedAudioDataset
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                    help='Path to saved model')
    parser.add_argument('--test_csv', type=str, default='data/processed/val_pairs.csv',
                    help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    try:
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Create test dataloader
        test_dataset = PairedAudioDataset(args.test_csv)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Run evaluation
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            save_dir='results',
            device=device
        )
        
        # Print results
        print("\nEvaluation Results:")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Average Precision: {results['average_precision']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(results['confusion_matrix']))
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise