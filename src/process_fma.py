import os
import librosa
import logging
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_audio(signal: np.ndarray, sr: int) -> bool:
    """Less strict audio validation."""
    try:
        # Check basic requirements
        duration = len(signal) / sr
        
        # More lenient duration check (2-4 seconds)
        if not (2.0 <= duration <= 4.0):
            return False
            
        # Only check for NaN/Inf
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return False
            
        # More lenient amplitude check
        if np.max(np.abs(signal)) < 1e-7:  # Only check if essentially silent
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error in validate_audio: {str(e)}")
        return False

def load_audio_file(mp3_path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Try multiple methods to load audio file."""
    signal = None
    
    # Method 1: Try librosa with different offsets
    for offset in [0.0, 0.1, 0.5]:
        try:
            signal, sr = librosa.load(
                mp3_path,
                sr=sr,
                duration=3.5,  # Load slightly longer
                mono=True,
                offset=offset
            )
            if signal is not None and len(signal) >= sr * 2.5:
                break
        except Exception:
            continue
    
    # Method 2: Try soundfile if librosa failed
    if signal is None:
        try:
            import soundfile as sf
            raw_signal, raw_sr = sf.read(mp3_path)
            
            # Handle stereo
            if len(raw_signal.shape) > 1:
                raw_signal = np.mean(raw_signal, axis=1)
            
            # Resample if needed
            if raw_sr != sr:
                signal = librosa.resample(raw_signal, orig_sr=raw_sr, target_sr=sr)
            else:
                signal = raw_signal
                
        except Exception:
            # Method 3: Last resort - try audioread directly
            try:
                import audioread
                with audioread.audio_open(mp3_path) as f:
                    raw_signal = np.concatenate([
                        librosa.util.buf_to_float(buf, dtype=np.float32)
                        for buf in f.read_data()
                    ])
                    if f.channels > 1:
                        raw_signal = raw_signal.reshape(-1, f.channels).mean(axis=1)
                    signal = librosa.resample(raw_signal, orig_sr=f.samplerate, target_sr=sr)
            except Exception:
                return None, sr
    
    return signal, sr

def process_fma_files(fma_dir: str, output_dir: str) -> tuple[int, int]:
    """Process all FMA files into 3-second WAV clips."""
    fma_dir = Path(fma_dir) / "fma_small"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    processed = 0
    failed = 0
    failed_files = []
    
    # Get all subdirectories
    subdirs = sorted(list(fma_dir.glob("[0-9][0-9][0-9]")))
    total_files = sum(len(list(subdir.glob("*.mp3"))) for subdir in subdirs)
    logger.info(f"Found {total_files} MP3 files to process")
    
    with tqdm(total=total_files, desc="Processing audio files") as pbar:
        for subdir in subdirs:
            for mp3_file in subdir.glob("*.mp3"):
                try:
                    mp3_path = str(mp3_file.absolute())
                    
                    # Try loading with enhanced function
                    signal, sr = load_audio_file(mp3_path)
                    
                    if signal is None:
                        logger.warning(f"Failed to load {mp3_path}")
                        failed_files.append(str(mp3_file))
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    # Validate and normalize
                    if not validate_audio(signal, sr):
                        logger.warning(f"Invalid audio in {mp3_path}")
                        failed_files.append(str(mp3_file))
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    # Process to exact length
                    target_length = sr * 3
                    if len(signal) > target_length:
                        signal = signal[:target_length]
                    elif len(signal) < target_length:
                        signal = np.pad(
                            signal, 
                            (0, target_length - len(signal)),
                            mode='constant'
                        )
                    
                    # Normalize with safeguards
                    max_val = np.max(np.abs(signal))
                    if max_val > 1e-6:
                        signal = signal / max_val
                    else:
                        logger.warning(f"Silent audio in {mp3_path}")
                        failed_files.append(str(mp3_file))
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    # Save as WAV
                    output_path = output_dir / f"clip_{processed:05d}.wav"
                    import soundfile as sf
                    sf.write(str(output_path), signal, sr)
                    processed += 1
                    
                    if processed % 1000 == 0:
                        logger.info(f"Processed {processed}/{total_files} files")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    failed_files.append(str(mp3_file))
                    logger.error(f"Error processing {mp3_path}: {str(e)}")
                    failed += 1
                    pbar.update(1)
                    continue
    
    # Log results
    logger.info(f"\nProcessing complete:")
    logger.info(f"Successfully processed: {processed} files")
    logger.info(f"Failed to process: {failed} files")
    
    return processed, failed

def clean_directories() -> None:
    """Clean up data directories before processing."""
    dirs_to_clean = ['data/base_audio', 'data/processed']
    
    for dir_path in dirs_to_clean:
        dir_path = Path(dir_path)
        try:
            if dir_path.exists():
                logger.info(f"Cleaning directory: {dir_path}")
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning directory {dir_path}: {str(e)}")
            raise

def create_dataset_splits(base_audio_dir: str, output_dir: str,
                        train_split: float = 0.75,
                        val_split: float = 0.15,
                        test_split: float = 0.10) -> None:
    """Create train/val/test splits with challenging pairs."""
    logger.info("Creating dataset splits with challenging pairs...")
    
    base_audio_dir = Path(base_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all WAV files and ensure no overlap between splits
    audio_files = list(base_audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {base_audio_dir}")
    
    # Shuffle files before splitting
    np.random.shuffle(audio_files)
    
    n_files = len(audio_files)
    train_idx = int(n_files * train_split)
    val_idx = int(n_files * (train_split + val_split))
    
    splits = {
        'train': audio_files[:train_idx],
        'val': audio_files[train_idx:val_idx],
        'test': audio_files[val_idx:]
    }
    
    for split_name, files in splits.items():
        pairs = []
        n_files = len(files)
        
        # Create positive pairs without temporal shifts
        for file in files:
            # Base positive pair
            pairs.append([str(file), str(file), 1])
            
            # Additional positive pairs for training set only
            if split_name == 'train':
                # Create more positive pairs with same file
                for _ in range(2):  # Add 2 more positive pairs
                    pairs.append([str(file), str(file), 1])
        
        # Create challenging negative pairs
        n_positive = len(pairs)
        used_pairs = set()
        
        for _ in range(n_positive):  # Match number of positive pairs
            idx1, idx2 = np.random.choice(n_files, 2, replace=False)
            file1, file2 = str(files[idx1]), str(files[idx2])
            
            # Skip if pair already used
            pair_key = f"{file1}_{file2}"
            if pair_key in used_pairs:
                continue
            
            # Add negative pair
            pairs.append([file1, file2, 0])
            used_pairs.add(pair_key)
        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        # Save pairs
        df = pd.DataFrame(pairs, columns=['clip1_path', 'clip2_path', 'label'])
        csv_path = output_dir / f'{split_name}_pairs.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Created {split_name} split:")
        logger.info(f"  - Total pairs: {len(df)}")
        logger.info(f"  - Positive pairs: {sum(df['label'] == 1)}")
        logger.info(f"  - Negative pairs: {sum(df['label'] == 0)}")

def balance_pairs(pairs: list) -> list:
    """Ensure dataset has balanced positive and negative pairs."""
    pos_pairs = [p for p in pairs if p[2] == 1]
    neg_pairs = [p for p in pairs if p[2] == 0]
    
    # Balance by undersampling the majority class
    min_samples = min(len(pos_pairs), len(neg_pairs))
    
    if len(pos_pairs) > min_samples:
        pos_pairs = pos_pairs[:min_samples]
    if len(neg_pairs) > min_samples:
        neg_pairs = neg_pairs[:min_samples]
    
    return pos_pairs + neg_pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_audio_dir', type=str, default='data/base_audio')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--train_split', type=float, default=0.75)  # Match function default
    parser.add_argument('--val_split', type=float, default=0.15)    # Match function default
    parser.add_argument('--test_split', type=float, default=0.10)   # Match function default
    args = parser.parse_args()
    
    try:
        # Clean directories first
        clean_directories()
        
        # Process FMA files
        processed, failed = process_fma_files(
            fma_dir="temp_fma",
            output_dir="data/base_audio"
        )
        
        # Create train/val/test splits
        if processed > 0:
            create_dataset_splits(
                base_audio_dir=args.base_audio_dir,
                output_dir=args.output_dir,
                train_split=args.train_split,
                val_split=args.val_split,
                test_split=args.test_split
            )
            logger.info("Dataset creation completed successfully!")
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        raise