import logging
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_audio(signal: np.ndarray, sr: int) -> bool:
    """Validate audio meets project requirements."""
    duration = len(signal) / sr  
    
    # More lenient duration check (2.5 to 3.5 seconds)
    if not (2.5 <= duration <= 3.5):
        logger.warning(f"Invalid duration: {duration:.2f}s")
        return False
    
    # Check if signal is too quiet
    if np.max(np.abs(signal)) < 1e-3:
        logger.warning("Signal too quiet")
        return False
    
    # Check if signal has any NaN values
    if np.any(np.isnan(signal)):
        logger.warning("Signal contains NaN values")
        return False
        
    return True

def apply_audio_transform(signal: np.ndarray, sr: int, transform_type: str) -> np.ndarray:
    """Apply audio transformations as per project requirements."""
    try:
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
            # Simple convolution-based reverb instead of using librosa
            impulse_response = np.exp(-np.linspace(0, 2, 8000))
            return np.convolve(signal, impulse_response, mode='same')
        else:
            logger.warning(f"Unknown transform type: {transform_type}")
            return signal
    except Exception as e:
        logger.error(f"Transform {transform_type} failed: {str(e)}")
        return signal

def create_dataset(base_audio_dir: str, output_dir: str) -> Tuple[str, str]:
    """Create dataset from audio files with required transformations."""
    base_audio_dir = Path(base_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    audio_files = []
    pairs = []
    
    # Check for audio files
    audio_paths = list(base_audio_dir.glob('*.wav'))
    if not audio_paths:
        raise ValueError(f"No WAV files found in {base_audio_dir}")
    
    logger.info(f"Found {len(audio_paths)} audio files")
    
    # Process each audio file
    for audio_path in audio_paths:
        try:
            # Load exactly 3 seconds and pad/trim if needed
            signal, sr = librosa.load(str(audio_path), sr=22050, duration=3.0)
            
            # Pad if too short
            if len(signal) < sr * 3:
                pad_length = (sr * 3) - len(signal)
                signal = np.pad(signal, (0, pad_length), mode='constant')
            
            # Trim if too long
            if len(signal) > sr * 3:
                signal = signal[:sr * 3]
            
            # Normalize audio
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
            
            # Save original
            orig_path = output_dir / f"orig_{audio_path.name}"
            sf.write(orig_path, signal, sr)
            audio_files.append(str(orig_path))
            
            # Create transformed versions
            for transform in ['pitch', 'tempo', 'noise', 'reverb']:
                trans_signal = apply_audio_transform(signal, sr, transform)
                if validate_audio(trans_signal, sr):
                    trans_path = output_dir / f"{transform}_{audio_path.name}"
                    sf.write(trans_path, trans_signal, sr)
                    
                    # Create positive pair
                    pairs.append({
                        'clip1_path': str(orig_path),
                        'clip2_path': str(trans_path),
                        'label': 1,
                        'transform': transform
                    })
                
            logger.info(f"Processed {audio_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            continue
    
    if not pairs:
        raise ValueError("No valid pairs created")
    
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
    
    # Log dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"  Positive: {sum(p['label'] == 1 for p in train_pairs)}")
    logger.info(f"  Negative: {sum(p['label'] == 0 for p in train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    logger.info(f"  Positive: {sum(p['label'] == 1 for p in val_pairs)}")
    logger.info(f"  Negative: {sum(p['label'] == 0 for p in val_pairs)}")
    
    return str(train_csv), str(val_csv)

def verify_setup(base_audio_dir: str, output_dir: str) -> bool:
    """Verify directory structure and create if needed."""
    try:
        # Create directories
        base_dir = Path(base_audio_dir)
        out_dir = Path(output_dir)
        
        base_dir.mkdir(exist_ok=True, parents=True)
        out_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if base directory is empty
        if not list(base_dir.glob('*.wav')):
            logger.warning(f"No WAV files found in {base_dir}")
            logger.info("Please add WAV files to the base_audio directory:")
            logger.info(f"  - Directory: {base_dir.absolute()}")
            logger.info("  - Required format: WAV")
            logger.info("  - Duration: ~3 seconds")
            logger.info("  - Sample rate: 22050 Hz")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    try:
        base_audio_dir = 'data/base_audio'
        output_dir = 'data/processed'
        
        # Verify setup first
        if not verify_setup(base_audio_dir, output_dir):
            logger.error("Please add WAV files and try again")
            exit(1)
            
        # Continue with dataset creation
        train_csv, val_csv = create_dataset(
            base_audio_dir=base_audio_dir,
            output_dir=output_dir
        )
        logger.info(f"\nDataset created successfully!")
        logger.info(f"Training pairs: {train_csv}")
        logger.info(f"Validation pairs: {val_csv}")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise