import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime
import pandas as pd
import numpy as np
import soundfile as sf
import random

from model import SiameseNetwork
from dataset import PairedAudioDataset

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration."""
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get('epochs', 20)
        self.batch_size: int = kwargs.get('batch_size', 11)  # Changed to match your data
        self.learning_rate: float = kwargs.get('learning_rate', 1e-3)
        self.val_freq: int = kwargs.get('val_freq', 1)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'checkpoints')
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x1, x2, label) in enumerate(train_loader):
        # Move data to device
        x1 = x1.to(device)  # Shape: [batch, 1, height, width]
        x2 = x2.to(device)  # Shape: [batch, 1, height, width]
        label = label.to(device).squeeze()  # Shape: [batch]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x1, x2).squeeze()  # Ensure output is [batch]
        loss = criterion(output, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)
        
        if batch_idx % 10 == 0:
            logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    epoch_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return epoch_loss, accuracy

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for x1, x2, label in val_loader:
        # Move data to device
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device).squeeze()  # Shape: [batch]
        
        output = model(x1, x2).squeeze()  # Shape: [batch]
        loss = criterion(output, label)
        
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)
    
    val_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return val_loss, accuracy

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str
) -> str:
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return str(checkpoint_path)

def train(
    train_csv: str,
    val_csv: str,
    config: Optional[Dict] = None
) -> nn.Module:
    """Main training function."""
    # Setup
    if config is None:
        config = TrainingConfig()
    else:
        config = TrainingConfig(**config)
    
    logger.info(f"Training on device: {config.device}")
    patience = 5
    no_improve = 0
    # Data loading
    train_dataset = PairedAudioDataset(train_csv)
    val_dataset = PairedAudioDataset(val_csv)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Model setup
    model = SiameseNetwork().to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}"
        )
        
        # Validate
        if (epoch + 1) % config.val_freq == 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, config.device
            )
            logger.info(
                f"Validation - "
                f"Loss: {val_loss:.4f}, "
                f"Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = save_checkpoint(
                    model, optimizer, epoch, val_loss, config.checkpoint_dir
                )
                logger.info(f"Saved best model to {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    return model

def create_test_data():
    # Setup directories
    data_dir = Path('data')
    audio_dir = data_dir / 'audio'
    data_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    # Audio parameters
    sr = 22050  # Sample rate
    duration = 3  # Duration in seconds
    t = np.linspace(0, duration, int(sr * duration))  # Time vector
    
    # Increase number of samples
    n_samples = 100  # Instead of 10
    
    # Generate more varied frequencies
    freq_ranges = [(220, 440), (440, 880)]
    audio_files = []
    
    for i in range(n_samples):
        # Pick random frequency range
        low, high = random.choice(freq_ranges)
        freq = np.random.uniform(low, high)
        
        # Add some noise/variation
        signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.01, len(t))
        signal = signal + noise
        
        # Save audio file
        filename = f'clip_{i}.wav'
        filepath = audio_dir / filename
        sf.write(filepath, signal, sr)
        audio_files.append((str(filepath), freq))  # Store frequency for pairing
    
    # Create train/val splits
    n_train = int(0.8 * len(audio_files))
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:]
    
    # Create more balanced pairs
    def create_pairs(files):
        pairs = []
        # Similar pairs (same frequency range)
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                if abs(files[i][1] - files[j][1]) / files[i][1] < 0.1:  # Within 10%
                    pairs.append({
                        'clip1_path': files[i][0],
                        'clip2_path': files[j][0],
                        'label': 1
                    })
                    if len(pairs) >= len(files):  # Control positive pair count
                        break
        
        # Different pairs
        n_neg = len(pairs)  # Match number of positive pairs
        for i in range(n_neg):
            idx1, idx2 = random.sample(range(len(files)), 2)
            if abs(files[idx1][1] - files[idx2][1]) / files[idx1][1] > 0.5:  # Very different
                pairs.append({
                    'clip1_path': files[idx1][0],
                    'clip2_path': files[idx2][0],
                    'label': 0
                })
        return pairs
    
    # Create and save CSVs
    train_pairs = create_pairs(train_files)
    val_pairs = create_pairs(val_files)
    
    pd.DataFrame(train_pairs).to_csv(data_dir / 'train_pairs.csv', index=False)
    pd.DataFrame(val_pairs).to_csv(data_dir / 'val_pairs.csv', index=False)
    
    logger.info(f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG for more verbose output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test data
    create_test_data()
    
    # Training configuration
    config = {
        'epochs': 50,  # More epochs
        'batch_size': 32,  # Larger batch size
        'learning_rate': 1e-4,  # Lower learning rate
        'val_freq': 1,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Train model
    try:
        model = train(
            train_csv='data/train_pairs.csv',
            val_csv='data/val_pairs.csv',
            config=config
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise