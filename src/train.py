import os
import logging
from tqdm import tqdm  # Add tqdm import
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import PairedAudioDataset
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
import json
from datetime import datetime
from model import SiameseNetwork
from evaluate import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

class TrainingConfig:
    """Enhanced training configuration with regularization."""    
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get('epochs', 50)  # Increased from 11
        self.batch_size: int = kwargs.get('batch_size', 64)  # Increased from 32
        self.learning_rate: float = kwargs.get('learning_rate', 1e-3)  # Increased from 1e-4
        self.weight_decay: float = kwargs.get('weight_decay', 1e-4)  # Added L2 regularization
        self.dropout_rate: float = kwargs.get('dropout_rate', 0.3)  # Added dropout rate
        self.val_freq: int = kwargs.get('val_freq', 1)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'checkpoints')
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> Tuple[float, float]:
    """Train for one epoch with gradient clipping."""    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for x1, x2, label in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        label = label.to(device).float().squeeze()
        
        optimizer.zero_grad()
        output = model(x1, x2).squeeze()
        loss = criterion(output, label)
        
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        label = label.to(device).float().squeeze()  # Added .float()
        
        output = model(x1, x2).squeeze()
        loss = criterion(output, label)
        
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)
    
    val_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return val_loss, accuracy

def train(train_csv: str, val_csv: str, config: Optional[Dict] = None) -> Tuple[nn.Module, Dict]:
    """Enhanced training function with better regularization."""    
    if config is None:
        config = TrainingConfig()
    else:
        config = TrainingConfig(**config)
    
    logger.info(f"Training on device: {config.device}")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience = 7  # Increased from 5
    no_improve = 0
    
    # Initialize metrics collection
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Data loading
    train_loader = DataLoader(
        PairedAudioDataset(train_csv, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        PairedAudioDataset(val_csv),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model setup with dropout
    model = SiameseNetwork(dropout_rate=config.dropout_rate).to(config.device)
    
    # Remove label smoothing for compatibility
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        amsgrad=True
    )
    
    # Enhanced learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # More aggressive reduction
        patience=4,   # Increased patience
        verbose=True,
        min_lr=1e-6  # Add minimum learning rate
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
        
        # Enhanced training configuration
        config = {
            'epochs': 15,              # Increased epochs
            'batch_size': 32,          # Larger batch size
            'learning_rate': 1e-3,     # Higher initial learning rate
            'weight_decay': 1e-4,      # L2 regularization
            'dropout_rate': 0.3,       # Moderate dropout
            'val_freq': 1,
            'checkpoint_dir': 'checkpoints',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Verify data files exist
        if not (Path(train_csv).exists() and Path(val_csv).exists()):
            raise FileNotFoundError("Dataset files not found. Run process_fma.py first.")
        
        # Train model
        model, train_metrics = train(train_csv, val_csv, config)
        
        # Log training results
        log_experiment_results(
            config=config,
            metrics=train_metrics,
            save_dir='results'
        )
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model state...")
        checkpoint_path = Path(config['checkpoint_dir']) / 'interrupted_model.pt'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved interrupted model to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise