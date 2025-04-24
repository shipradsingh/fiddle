import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from datetime import datetime
import pandas as pd
import numpy as np
import soundfile as sf
import random
import librosa

from model import SiameseNetwork
from dataset import PairedAudioDataset
from evaluate import evaluate_model

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration matching project requirements."""
    def __init__(self, **kwargs):
        self.epochs: int = kwargs.get('epochs', 50)  # As per project spec
        self.batch_size: int = kwargs.get('batch_size', 32)
        self.learning_rate: float = kwargs.get('learning_rate', 1e-4)
        self.val_freq: int = kwargs.get('val_freq', 1)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'checkpoints')
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def apply_audio_transform(signal: np.ndarray, sr: int, transform_type: str) -> np.ndarray:
    """Apply audio transformations as per project requirements."""
    if transform_type == 'pitch':
        n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    elif transform_type == 'tempo':
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(signal, rate=rate)
    elif transform_type == 'noise':
        noise = np.random.normal(0, 0.01, len(signal))
        return signal + noise
    elif transform_type == 'reverb':
        return librosa.effects.reverb(signal, sr=sr)
    return signal

def save_experiment_config(config: Dict, metrics: Dict, save_dir: str):
    """Save full experiment configuration and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir) / f"experiment_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_path / "config.json", "w") as f:
        json.dump({
            'training_config': config,
            'results': metrics,
            'timestamp': timestamp
        }, f, indent=4)
    
    return save_path

def create_dataset(base_audio_dir: str, output_dir: str) -> Tuple[str, str]:
    """Create dataset from real audio files with required transformations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    audio_files = []
    pairs = []
    
    # Process base audio files
    for audio_path in Path(base_audio_dir).glob('*.wav'):
        signal, sr = librosa.load(str(audio_path), sr=22050, duration=3.0)
        
        # Save original
        orig_path = output_dir / f"orig_{audio_path.name}"
        sf.write(orig_path, signal, sr)
        audio_files.append(str(orig_path))
        
        # Create transformed versions
        for transform in ['pitch', 'tempo', 'noise', 'reverb']:
            trans_signal = apply_audio_transform(signal, sr, transform)
            trans_path = output_dir / f"{transform}_{audio_path.name}"
            sf.write(trans_path, trans_signal, sr)
            
            # Create positive pair
            pairs.append({
                'clip1_path': str(orig_path),
                'clip2_path': str(trans_path),
                'label': 1,
                'transform': transform
            })
    
    # Create negative pairs
    n_positive = len(pairs)
    for _ in range(n_positive):
        file1, file2 = random.sample(audio_files, 2)
        pairs.append({
            'clip1_path': file1,
            'clip2_path': file2,
            'label': 0,
            'transform': 'none'
        })
    
    # Split into train/val
    random.shuffle(pairs)
    n_train = int(0.8 * len(pairs))
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    
    # Save to CSV
    train_csv = output_dir / 'train_pairs.csv'
    val_csv = output_dir / 'val_pairs.csv'
    
    pd.DataFrame(train_pairs).to_csv(train_csv, index=False)
    pd.DataFrame(val_pairs).to_csv(val_csv, index=False)
    
    logger.info(f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    logger.info(f"Training pairs: {sum(p['label'] == 1 for p in train_pairs)} positive, "
               f"{sum(p['label'] == 0 for p in train_pairs)} negative")
    logger.info(f"Validation pairs: {sum(p['label'] == 1 for p in val_pairs)} positive, "
               f"{sum(p['label'] == 0 for p in val_pairs)} negative")
    
    return str(train_csv), str(val_csv)

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x1, x2, label) in enumerate(train_loader):
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
          config: Optional[Dict] = None) -> nn.Module:
    """Main training function with early stopping."""
    if config is None:
        config = TrainingConfig()
    else:
        config = TrainingConfig(**config)
    
    logger.info(f"Training on device: {config.device}")
    
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
    
    return model

def log_experiment_results(config: Dict, metrics: Dict, save_dir: str):
    """Log comprehensive experiment results as required by project."""
    save_dir = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment summary
    summary = {
        'timestamp': timestamp,
        'config': config,
        'training_metrics': {
            'final_train_loss': metrics['train_loss'][-1],
            'final_train_acc': metrics['train_acc'][-1],
            'final_val_loss': metrics['val_loss'][-1],
            'final_val_acc': metrics['val_acc'][-1],
            'best_val_loss': min(metrics['val_loss']),
            'epochs_trained': len(metrics['train_loss'])
        },
        'evaluation_metrics': metrics['evaluation']
    }
    
    # Save complete results
    with open(save_dir / f'experiment_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Log key metrics
    logger.info("\nExperiment Results:")
    logger.info(f"ROC AUC Score: {metrics['evaluation']['roc_auc']:.3f}")
    logger.info(f"Best Validation Loss: {min(metrics['val_loss']):.3f}")
    logger.info("\nThreshold Analysis:")
    for thresh_result in metrics['evaluation']['threshold_analysis']:
        logger.info(
            f"Threshold {thresh_result['threshold']:.2f}: "
            f"Precision={thresh_result['precision']:.3f}, "
            f"Recall={thresh_result['recall']:.3f}, "
            f"FPR={thresh_result['false_positive_rate']:.3f}"
        )

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dataset from real audio files
    train_csv, val_csv = create_dataset(
        base_audio_dir='data/base_audio',
        output_dir='data/processed'
    )
    
    # Training configuration
    config = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'val_freq': 1,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Train and evaluate model
    try:
        model = train(train_csv, val_csv, config)
        
        # Run evaluation
        test_loader = DataLoader(
            PairedAudioDataset(val_csv),  # Using validation set for testing
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            save_dir='results',
            device=config.device
        )
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise