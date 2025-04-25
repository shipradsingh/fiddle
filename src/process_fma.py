import os
import librosa
import soundfile as sf
from pathlib import Path
import logging
from tqdm import tqdm
import shutil
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_audio(signal: np.ndarray, sr: int) -> bool:
    """Validate audio meets project requirements."""
    duration = len(signal) / sr
    if not (2.9 <= duration <= 3.1):
        return False
    if np.max(np.abs(signal)) > 1.0:
        return False
    return True

def process_fma_files(fma_dir: str, output_dir: str):
    """Process all FMA files into 3-second WAV clips at 22050Hz."""
    fma_dir = Path(fma_dir) / "fma_small"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Clean output directory
    if output_dir.exists():
        logger.info(f"Cleaning {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    processed = 0
    failed = 0
    
    # Get all subdirectories (000-076)
    subdirs = sorted(list(fma_dir.glob("[0-9][0-9][0-9]")))
    
    # Count total files first
    total_files = sum(len(list(subdir.glob("*.mp3"))) for subdir in subdirs)
    logger.info(f"Found {total_files} MP3 files to process")
    
    with tqdm(total=total_files, desc="Processing audio files") as pbar:
        for subdir in subdirs:
            # Process each MP3 file in the subdirectory
            for mp3_file in subdir.glob("*.mp3"):
                try:
                    # Load and resample to project requirements
                    signal, sr = librosa.load(
                        mp3_file,
                        sr=22050,  # Required sample rate
                        duration=3.0  # Required duration
                    )
                    
                    # Normalize
                    signal = signal / np.max(np.abs(signal))
                    
                    # Validate audio
                    if not validate_audio(signal, sr):
                        logger.warning(f"Skipping invalid audio: {mp3_file}")
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    # Save as WAV
                    output_path = output_dir / f"clip_{processed:05d}.wav"
                    sf.write(output_path, signal, sr)
                    
                    processed += 1
                    pbar.update(1)
                        
                except Exception as e:
                    logger.error(f"Error processing {mp3_file}: {e}")
                    failed += 1
                    pbar.update(1)
                    continue
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"Successfully processed: {processed} files")
    logger.info(f"Failed to process: {failed} files")
    return processed, failed

if __name__ == "__main__":
    try:
        processed, failed = process_fma_files(
            fma_dir="temp_fma",
            output_dir="data/base_audio"
        )
        
        if processed > 0:
            # Only cleanup if we successfully processed some files
            if Path("temp_fma").exists():
                logger.info("Cleaning up temporary files...")
                shutil.rmtree("temp_fma")
                
            logger.info("Dataset preparation completed successfully!")
        else:
            logger.error("No files were processed successfully!")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise