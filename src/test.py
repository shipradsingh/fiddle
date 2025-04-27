import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import SiameseNetwork
from dataset import PairedAudioDataset

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def test_model(
    model_path: str,
    test_csv: str,
    results_dir: str = 'results',
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """Run comprehensive evaluation on test set."""
    logger.info("Starting model testing...")
    Path(results_dir).mkdir(exist_ok=True)
    
    # Load model with dropout enabled for uncertainty estimation
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create test dataloader
    test_dataset = PairedAudioDataset(test_csv, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize metrics
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0
    criterion = torch.nn.BCELoss(reduction='none')  # Changed to get per-sample loss
    
    # Create progress bar
    pbar = tqdm(test_loader, desc="Testing")
    
    model.eval()
    with torch.no_grad():
        for x1, x2, label in pbar:
            # Move data to device
            x1, x2 = x1.to(device), x2.to(device)
            label = label.to(device).float().squeeze()
            
            # Forward pass
            output = model(x1, x2).squeeze()
            loss = criterion(output, label)
            
            # Store predictions and labels
            pred = (output > 0.5).float()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(output.cpu().numpy())
            running_loss += loss.mean().item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{running_loss/len(test_loader):.4f}'})
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate comprehensive metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc_roc = roc_auc_score(y_true, y_prob)
    accuracy = (y_true == y_pred).mean()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        save_path=f"{results_dir}/confusion_matrix.png"
    )
    
    # Analyze error cases
    error_indices = np.where(y_true != y_pred)[0]
    error_analysis = {
        'false_positives': sum((y_pred == 1) & (y_true == 0)),
        'false_negatives': sum((y_pred == 0) & (y_true == 1)),
        'error_rate': len(error_indices) / len(y_true)
    }
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'avg_loss': running_loss / len(test_loader),
        'error_analysis': error_analysis,
        'total_samples': len(y_true)
    }
    
    # Log detailed results
    logger.info("Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"Error Analysis: {error_analysis}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_pairs.csv')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    try:
        results = test_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            results_dir=args.results_dir,
            batch_size=args.batch_size
        )
        logger.info("Testing completed successfully!")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise