import os
import logging
import warnings
from tqdm import tqdm

# Configure logging before any imports
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(message)s'
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Now do the rest of the imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from datetime import datetime

from model import SiameseNetwork
from dataset import PairedAudioDataset
from evaluate import evaluate_model

class TrainingConfig:
    """Training configuration matching project requirements."""
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get('epochs', 50)
        self.batch_size: int = kwargs.get('batch_size', 32)
        self.learning_rate: float = kwargs.get('learning_rate', 1e-4)
        self.val_freq: int = kwargs.get('val_freq', 1)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'checkpoints')
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> Tuple[float, float]:
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for x1, x2, label in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        label = label.to(device).squeeze()
        
        optimizer.zero_grad()
        output = model(x1, x2).squeeze()
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(correct/total):.4f}'
        })
    
    epoch_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return epoch_loss, accuracy

@torch.no_grad()
def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for x1, x2, label in val_loader:
        x1, x2 = x1.to(device), x2.to(device)
        label = label.to(device).squeeze()
        
        output = model(x1, x2).squeeze()
        loss = criterion(output, label)
        
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)
    
    val_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return val_loss, accuracy

def train(train_csv: str,
        val_csv: str,
        config: Optional[Dict] = None) -> Tuple[nn.Module, Dict]:
    """Main training function with early stopping."""
    if config is None:
        config = TrainingConfig()
    else:
        config = TrainingConfig(**config)
    
    logger.info(f"Training on device: {config.device}")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    
    # Initialize metrics collection
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Data loading
    train_dataset = PairedAudioDataset(train_csv)
    val_dataset = PairedAudioDataset(val_csv)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 4
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Reduced from 4
        pin_memory=True
    )
    
    # Model setup
    model = SiameseNetwork().to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.device
        )
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            checkpoint_path = Path(config.checkpoint_dir) / f"best_model.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, metrics

def log_experiment_results(config: Dict, metrics: Dict, save_dir: str) -> None:
    """Log experiment configuration and metrics to JSON file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'config': config,
        'metrics': metrics
    }
    
    save_path = save_dir / f'experiment_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Experiment results saved to {save_path}")

if __name__ == "__main__":
    try:
        # Define data paths
        train_csv = 'data/processed/train_pairs.csv'
        val_csv = 'data/processed/val_pairs.csv'
        
        # Create necessary directories
        Path('checkpoints').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
        # Training configuration
        config = {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'val_freq': 1,
            'checkpoint_dir': 'checkpoints',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Verify data files exist
        if not (Path(train_csv).exists() and Path(val_csv).exists()):
            raise FileNotFoundError("Dataset files not found. Run process_fma.py first.")
        
        # Train and evaluate model
        model, train_metrics = train(train_csv, val_csv, config)
        
        # Run evaluation
        test_loader = DataLoader(
            PairedAudioDataset(val_csv),
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Run evaluation and store results
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            save_dir='results',
            device=config['device']  # Keep as dictionary access since config is still a dict here
        )
        
        # Log experiment results
        log_experiment_results(
            config=config,  # This is already a dictionary
            metrics={**train_metrics, 'evaluation': results},
            save_dir='results'
        )
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise