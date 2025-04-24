import torch
import pandas as pd
import logging
import numpy as np
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Callable
import os

logger = logging.getLogger(__name__)

class AudioAugmenter:
    """Audio augmentation class implementing project requirements."""
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def pitch_shift(self, signal: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift pitch by n_steps semitones."""
        return librosa.effects.pitch_shift(signal, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """Stretch signal by rate (>1 = slower, <1 = faster)."""
        return librosa.effects.time_stretch(signal, rate=rate)
    
    def add_noise(self, signal: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to signal."""
        noise = np.random.normal(0, noise_level, len(signal))
        return signal + noise
    
    def add_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Add reverb effect to signal."""
        return librosa.effects.reverb(signal, sr=self.sr)

class PairedAudioDataset(Dataset):
    """
    Dataset for loading pairs of audio clips for similarity comparison.
    
    Parameters match project requirements:
    - Mel spectrogram: 128 bands
    - Window size: 2048 samples
    - Hop length: 512 samples
    - Duration: 3 seconds
    - Sample rate: 22050 Hz
    
    Augmentations available:
    - Pitch shifting (-2 to +2 semitones)
    - Time stretching (0.8x to 1.2x)
    - Gaussian noise addition
    - Reverb effect
    """
    def __init__(
        self, 
        csv_path: str, 
        sr: int = 22050,
        duration: float = 3.0,
        n_mels: int = 128,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        super().__init__()
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.augmenter = AudioAugmenter(sr=sr) if augment else None
        
        # Load and validate CSV
        try:
            self.pairs = pd.read_csv(csv_path)
            required_cols = ['clip1_path', 'clip2_path', 'label']
            if not all(col in self.pairs.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
                
            # Validate labels are binary
            if not self.pairs['label'].isin([0, 1]).all():
                raise ValueError("Labels must be binary (0 or 1)")
                
            logger.info(f"Loaded {len(self.pairs)} audio pairs")
            logger.info(f"Positive pairs: {sum(self.pairs['label'] == 1)}")
            logger.info(f"Negative pairs: {sum(self.pairs['label'] == 0)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def load_and_process_audio(self, file_path: str, idx: int = None) -> torch.Tensor:
        """Load audio file and convert to mel spectrogram with project parameters."""
        try:
            # Load audio with consistent duration
            signal, _ = librosa.load(
                file_path, 
                sr=self.sr, 
                duration=self.duration
            )
            
            # Ensure consistent length
            target_length = int(self.sr * self.duration)
            if len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))
            else:
                signal = signal[:target_length]
            
            # Apply augmentations if enabled and this is a positive pair
            if self.augmenter and idx is not None:
                if self.pairs.iloc[idx]['label'] == 1:
                    aug_type = np.random.choice(['pitch', 'stretch', 'noise', 'reverb'])
                    if aug_type == 'pitch':
                        signal = self.augmenter.pitch_shift(signal, np.random.uniform(-2, 2))
                    elif aug_type == 'stretch':
                        signal = self.augmenter.time_stretch(signal, np.random.uniform(0.8, 1.2))
                    elif aug_type == 'noise':
                        signal = self.augmenter.add_noise(signal)
                    else:
                        signal = self.augmenter.add_reverb(signal)
            
            # Convert to mel spectrogram with exact project parameters
            mel_spec = librosa.feature.melspectrogram(
                y=signal,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=2048,  # window_size in project specs
                hop_length=512,
                window='hann'
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # Convert to tensor with shape [1, height, width]
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            logger.debug(f"Processed spectrogram shape: {mel_spec.shape}")
            return mel_spec
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single pair of spectrograms and their similarity label."""
        try:
            row = self.pairs.iloc[idx]
            
            # Load and process both audio files
            x1 = self.load_and_process_audio(row['clip1_path'], idx)
            x2 = self.load_and_process_audio(row['clip2_path'], idx)
            
            if self.transform:
                x1 = self.transform(x1)
                x2 = self.transform(x2)
            
            # Convert label to tensor
            label = torch.FloatTensor([row['label']])
                
            return x1, x2, label
            
        except Exception as e:
            logger.error(f"Error loading pair at index {idx}: {str(e)}")
            raise

    def get_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced batches."""
        labels = self.pairs['label'].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        return torch.from_numpy(sample_weights)

    def validate_dataset(self) -> bool:
        """Validate dataset meets project requirements."""
        try:
            # Check class balance
            labels = self.pairs['label'].values
            pos_ratio = labels.mean()
            if not (0.4 <= pos_ratio <= 0.6):
                logger.warning(f"Imbalanced dataset: {pos_ratio:.2f} positive ratio")
                
            # Verify audio files
            for _, row in self.pairs.iterrows():
                for path in [row['clip1_path'], row['clip2_path']]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Audio file not found: {path}")
                        
                    signal, sr = librosa.load(path, sr=22050)
                    if not validate_audio(signal, sr):
                        raise ValueError(f"Invalid audio file: {path}")
                        
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False

def validate_audio(signal: np.ndarray, sr: int) -> bool:
    """Validate audio meets project requirements."""
    duration = len(signal) / sr
    if not (2.9 <= duration <= 3.1):  # Allow small tolerance
        return False
    if np.max(np.abs(signal)) > 1.0:  # Check normalization
        return False
    return True

if __name__ == "__main__":
    import os
    import soundfile as sf
    import shutil
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create test directory
    test_dir = "test_audio"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    try:
        # Generate test audio files
        sr = 22050
        duration = 3
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create test audio signals
        test_files = {
            'test1.wav': np.sin(2 * np.pi * 440 * t),  # A4 note
            'test2.wav': np.sin(2 * np.pi * 880 * t),  # A5 note
            'test1_mod.wav': np.sin(2 * np.pi * 440 * t) * 0.8  # Quieter A4
        }
        
        # Save test audio files
        for filename, signal in test_files.items():
            filepath = os.path.join(test_dir, filename)
            sf.write(filepath, signal, sr)
            logger.debug(f"Created test file: {filepath}")
        
        # Create test CSV
        test_data = {
            'clip1_path': [
                os.path.join(test_dir, 'test1.wav'), 
                os.path.join(test_dir, 'test2.wav')
            ],
            'clip2_path': [
                os.path.join(test_dir, 'test1_mod.wav'),
                os.path.join(test_dir, 'test1.wav')
            ],
            'label': [1, 0]  # Similar pair, different pair
        }
        
        test_csv = os.path.join(test_dir, 'test_pairs.csv')
        pd.DataFrame(test_data).to_csv(test_csv, index=False)
        logger.debug(f"Created test CSV: {test_csv}")
        
        # Test dataset loading with augmentations
        dataset = PairedAudioDataset(test_csv, augment=True)
        logger.info(f"Dataset size: {len(dataset)}")
        
        # Test all pairs
        for i in range(len(dataset)):
            x1, x2, label = dataset[i]
            logger.info(f"Pair {i + 1}/{len(dataset)}")
            logger.info(f"  Shapes: {x1.shape}, {x2.shape}")
            logger.info(f"  Label: {label}")
            
        logger.info("All tests passed successfully!")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.debug(f"Cleaned up test directory: {test_dir}")