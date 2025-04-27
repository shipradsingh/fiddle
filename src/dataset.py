import os
import logging
import warnings
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
from typing import Optional, Callable, Tuple, List
import random

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

class AudioAugmenter:
    """Enhanced audio augmentation class with more transformations."""
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.noise_types = ['white', 'pink', 'brown']
    
    def pitch_shift(self, signal: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift pitch by n_steps semitones."""
        return librosa.effects.pitch_shift(y=signal, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """Stretch signal by rate (>1 = slower, <1 = faster)."""
        return librosa.effects.time_stretch(y=signal, rate=rate)
    
    def add_noise(self, signal: np.ndarray, noise_type: str = 'white', 
                noise_level: float = 0.01) -> np.ndarray:
        """Add different types of noise to signal."""
        noise = np.zeros_like(signal)
        if noise_type == 'white':
            noise = np.random.normal(0, noise_level, len(signal))
        elif noise_type == 'pink':
            noise = np.interp(
                np.linspace(0, len(signal), len(signal)), 
                np.arange(len(signal)), 
                np.random.normal(0, noise_level, len(signal))
            )
        elif noise_type == 'brown':
            noise = np.cumsum(np.random.normal(0, noise_level, len(signal)))
            noise = noise / np.max(np.abs(noise)) * noise_level
        return signal + noise
    
    def add_reverb(self, signal: np.ndarray, room_scale: float = 0.5) -> np.ndarray:
        """Add reverb with configurable room size."""
        decay = np.exp(-np.linspace(0, room_scale * 5, int(self.sr * room_scale)))
        impulse_response = decay * np.random.randn(len(decay))
        return np.convolve(signal, impulse_response, mode='same')
    
    def time_mask(self, signal: np.ndarray, max_mask_size: float = 0.2) -> np.ndarray:
        """Apply time masking."""
        mask_size = int(len(signal) * max_mask_size * random.random())
        start = random.randint(0, len(signal) - mask_size)
        masked_signal = signal.copy()
        masked_signal[start:start + mask_size] = 0
        return masked_signal
    
    def apply_random_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Apply a random combination of augmentations."""
        aug_types = ['pitch', 'stretch', 'noise', 'reverb', 'mask']
        n_augs = random.randint(1, 3)  # Apply 1-3 augmentations
        chosen_augs = random.sample(aug_types, n_augs)
        
        for aug_type in chosen_augs:
            if aug_type == 'pitch':
                signal = self.pitch_shift(signal, np.random.uniform(-3, 3))
            elif aug_type == 'stretch':
                signal = self.time_stretch(signal, np.random.uniform(0.7, 1.3))
            elif aug_type == 'noise':
                noise_type = random.choice(self.noise_types)
                signal = self.add_noise(signal, noise_type, np.random.uniform(0.01, 0.05))
            elif aug_type == 'reverb':
                signal = self.add_reverb(signal, np.random.uniform(0.3, 0.7))
            elif aug_type == 'mask':
                signal = self.time_mask(signal)
        
        return signal

class PairedAudioDataset(Dataset):
    """Enhanced dataset with more sophisticated augmentations."""
    def __init__(
        self, 
        csv_path: str, 
        sr: int = 22050,
        duration: float = 3.0,
        n_mels: int = 128,
        transform: Optional[Callable] = None,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.transform = transform
        self.augmenter = AudioAugmenter(sr=sr) if augment else None
        
        # Load and validate CSV
        self.data = pd.read_csv(csv_path)
        required_cols = ['clip1_path', 'clip2_path', 'label']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Log dataset info
        logger.info(f"Loaded {len(self.data)} audio pairs")
        logger.info(f"Positive pairs: {sum(self.data['label'] == 1)}")
        logger.info(f"Negative pairs: {sum(self.data['label'] == 0)}")
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.data)

    def process_audio_to_melspec(self, signal: np.ndarray) -> torch.Tensor:
        """Convert audio signal to mel spectrogram with fixed size."""
        # Ensure fixed length
        target_length = int(self.sr * self.duration)
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            signal = signal[:target_length]
        
        # Fixed parameters
        n_fft = 2048
        hop_length = 512
        n_frames = 1 + (target_length // hop_length)  # Calculate expected frames
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Fix time dimension
        if mel_spec.shape[1] != n_frames:
            if mel_spec.shape[1] < n_frames:
                # Pad if too short
                pad_width = ((0, 0), (0, n_frames - mel_spec.shape[1]))
                mel_spec = np.pad(mel_spec, pad_width)
            else:
                # Truncate if too long
                mel_spec = mel_spec[:, :n_frames]
        
        # Convert to log scale and normalize
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return torch.FloatTensor(mel_spec).unsqueeze(0)

    def load_and_process_audio(self, file_path: str) -> torch.Tensor:
        """Load and process audio file to mel spectrogram with fixed size."""
        try:
            signal, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            return self.process_audio_to_melspec(signal)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a pair of spectrograms with enhanced augmentation."""
        row = self.data.iloc[idx]
        
        # Load audio files
        x1 = self.load_and_process_audio(row['clip1_path'])
        x2 = self.load_and_process_audio(row['clip2_path'])
        
        # Apply augmentation if enabled and this is a positive pair
        if self.augmenter and row['label'] == 1:
            if random.random() < self.augment_prob:
                signal = librosa.load(row['clip2_path'], sr=self.sr)[0]
                signal = self.augmenter.apply_random_augmentation(signal)
                x2 = self.process_audio_to_melspec(signal)
        
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        
        return x1, x2, torch.FloatTensor([row['label']])