import logging
import librosa
import numpy as np
import torch
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pad_or_truncate(y, length=66150):  # 3 seconds at 22050 Hz
    """Ensures all audio clips are same length by padding or truncating."""
    if len(y) < length:
        return np.pad(y, (0, length - len(y)))
    return y[:length]

def load_audio(path, sr=22050, duration=3):
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

def extract_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
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
        
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                    hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to [-1, 1] range
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_db = (S_db * 2) - 1
    
    return S_db

def preprocess_pair(path1, path2):
    """
    Processes a pair of audio files into normalized Mel spectrograms.
    
    Args:
        path1 (str): Path to first audio file
        path2 (str): Path to second audio file
    
    Returns:
        tuple: Two PyTorch tensors of shape (1, 1, 128, time)
    """
    y1 = load_audio(path1)
    y2 = load_audio(path2)
    mel1 = extract_mel_spectrogram(y1)
    mel2 = extract_mel_spectrogram(y2)
    mel1 = torch.tensor(mel1).unsqueeze(0).unsqueeze(0)
    mel2 = torch.tensor(mel2).unsqueeze(0).unsqueeze(0)
    return mel1.float(), mel2.float()

if __name__ == "__main__":
    # Set logging to DEBUG for more detailed output
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Example usage
    import soundfile as sf
    
    logger.info("Creating test audio signal...")
    sample_rate = 22050
    t = np.linspace(0, 3, sample_rate * 3)
    test_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save test audio
    test_path = "test_audio.wav"
    sf.write(test_path, test_signal, sample_rate)
    logger.info(f"Saved test audio to {test_path}")
    
    # Test pipeline
    try:
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
        
        # Test pair processing
        logger.info("Testing pair processing...")
        mel1, mel2 = preprocess_pair(test_path, test_path)
        logger.info(f"Processed pair shapes: {mel1.shape}, {mel2.shape}")
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)
            logger.info("Cleaned up test files")