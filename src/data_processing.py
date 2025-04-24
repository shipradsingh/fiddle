"""
Audio processing module for the Fiddle project.

Implements the following project requirements:
- Mel spectrogram parameters:
- 128 Mel bands
- Window size: 2048 samples
- Hop length: 512 samples
- Sample rate: 22050 Hz
- Duration: 3 seconds

- Audio augmentations:
- Pitch shifting (-2 to +2 semitones)
- Time stretching (0.8x to 1.2x)
- Gaussian noise addition
- Reverb effect
"""

import logging
import librosa
import numpy as np
import torch
import os
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_augmentation(y: np.ndarray, sr: int = 22050, aug_type: Optional[str] = None) -> np.ndarray:
    """Apply audio augmentations as specified in project requirements."""
    if aug_type is None:
        return y
        
    if aug_type == 'pitch':
        n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    elif aug_type == 'stretch':
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y, rate=rate)
    
    elif aug_type == 'noise':
        noise = np.random.normal(0, 0.01, len(y))
        return y + noise
    
    elif aug_type == 'reverb':
        return librosa.effects.reverb(y, sr=sr)
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def pad_or_truncate(y: np.ndarray, length: int = 66150) -> np.ndarray:  # 3 seconds at 22050 Hz
    """Ensures all audio clips are same length by padding or truncating."""
    if len(y) < length:
        return np.pad(y, (0, length - len(y)))
    return y[:length]

def load_audio(path: str, sr: int = 22050, duration: float = 3) -> np.ndarray:
    """
    Loads and validates audio file.
    
    Args:
        path (str): Path to audio file
        sr (int): Sample rate
        duration (float): Duration in seconds
    
    Returns:
        numpy.ndarray: Audio waveform
    """
    logger.debug(f"Loading audio file: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
        y = pad_or_truncate(y)
        return y
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Converts waveform to normalized Mel Spectrogram.
    
    Args:
        y (numpy.ndarray): Audio waveform
        sr (int): Sample rate
        n_mels (int): Number of Mel bands
        hop_length (int): Number of samples between frames
        n_fft (int): FFT window size
    
    Returns:
        numpy.ndarray: Normalized Mel spectrogram
    """
    logger.debug(f"Extracting mel spectrogram with shape: {y.shape}")
    if len(y) == 0:
        raise ValueError("Empty audio signal")
        
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        window='hann'
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to [-1, 1] range
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_db = (S_db * 2) - 1
    
    return S_db

def preprocess_pair(path1: str, path2: str, augment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes a pair of audio files into normalized Mel spectrograms.
    
    Args:
        path1 (str): Path to first audio file
        path2 (str): Path to second audio file
        augment (bool): Whether to apply random augmentation
    
    Returns:
        tuple: Two PyTorch tensors of shape (1, 1, 128, time)
    """
    y1 = load_audio(path1)
    y2 = load_audio(path2)
    
    if augment:
        aug_type = np.random.choice(['pitch', 'stretch', 'noise', 'reverb'])
        logger.debug(f"Applying {aug_type} augmentation to second clip")
        y2 = apply_augmentation(y2, aug_type=aug_type)
    
    mel1 = extract_mel_spectrogram(y1)
    mel2 = extract_mel_spectrogram(y2)
    mel1 = torch.tensor(mel1).unsqueeze(0).unsqueeze(0)
    mel2 = torch.tensor(mel2).unsqueeze(0).unsqueeze(0)
    return mel1.float(), mel2.float()

def apply_augmentation_chain(signal: np.ndarray, sr: int) -> np.ndarray:
    """Apply multiple random augmentations."""
    transforms = ['pitch', 'tempo', 'noise', 'reverb']
    n_transforms = np.random.randint(1, len(transforms) + 1)
    selected = np.random.choice(transforms, n_transforms, replace=False)
    
    for transform in selected:
        signal = apply_augmentation(signal, sr, transform)
    
    return signal

if __name__ == "__main__":
    # Set logging to DEBUG for more detailed output
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Create test directory
    test_dir = "test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        logger.info("Creating test audio signal...")
        sample_rate = 22050
        t = np.linspace(0, 3, sample_rate * 3)
        test_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save test audio
        test_path = os.path.join(test_dir, "test_audio.wav")
        import soundfile as sf
        sf.write(test_path, test_signal, sample_rate)
        logger.info(f"Saved test audio to {test_path}")
        
        # Test audio loading
        logger.info("Testing audio loading...")
        y = load_audio(test_path)
        logger.info(f"Loaded audio shape: {y.shape}, range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Test mel spectrogram extraction
        logger.info("Testing mel spectrogram extraction...")
        mel = extract_mel_spectrogram(y)
        logger.info(f"Mel spectrogram shape: {mel.shape}")
        logger.info(f"Mel spectrogram range: [{mel.min():.2f}, {mel.max():.2f}]")
        
        # Validate spectrogram properties
        assert mel.shape[0] == 128, "Expected 128 mel bands"
        assert -1 <= mel.min() <= mel.max() <= 1, "Expected normalized range [-1, 1]"
        
        # Test augmentations
        logger.info("Testing augmentations...")
        for aug_type in ['pitch', 'stretch', 'noise', 'reverb']:
            y_aug = apply_augmentation(y, aug_type=aug_type)
            logger.info(f"{aug_type.capitalize()} augmentation shape: {y_aug.shape}")
            assert len(y_aug) > 0, f"Empty signal after {aug_type} augmentation"
            assert not np.allclose(y, y_aug), f"No change after {aug_type} augmentation"
        
        # Test pair processing with and without augmentation
        logger.info("Testing pair processing...")
        mel1, mel2 = preprocess_pair(test_path, test_path, augment=False)
        logger.info(f"Regular pair shapes: {mel1.shape}, {mel2.shape}")
        
        mel1_aug, mel2_aug = preprocess_pair(test_path, test_path, augment=True)
        logger.info(f"Augmented pair shapes: {mel1_aug.shape}, {mel2_aug.shape}")
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Clean up test files
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            logger.info("Cleaned up test files")