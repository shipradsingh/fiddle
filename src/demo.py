import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from pydub import AudioSegment
import logging
from model import SiameseNetwork
from dataset import AudioAugmenter

# Configure logging with a simplified format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format
    force=True
)
logger = logging.getLogger(__name__)

def modify_audio_file(input_path: Path, output_path: Path, augmenter: AudioAugmenter):
    """Create modified version of audio file using augmenter."""
    # Load audio
    signal, sr = librosa.load(str(input_path), sr=22050)
    
    # Apply random augmentations
    modified_signal = augmenter.apply_random_augmentation(signal)
    
    # Save modified audio
    sf.write(str(output_path), modified_signal, sr)

def find_demo_pairs(test_csv: str, output_dir: str = "data/demo"):
    """Find and copy good example pairs from test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize augmenter
    augmenter = AudioAugmenter(sr=22050)
    
    df = pd.read_csv(test_csv)
    original_pair = df[df['label'] == 1].iloc[0]
    different_pair = df[df['label'] == 0].iloc[0]
    
    # Copy original file
    original_path = output_dir / 'original.wav'
    shutil.copy(original_pair['clip1_path'], original_path)
    
    # Create modified versions with varying intensities
    modifications = {
        'slight_pitch': lambda s: augmenter.pitch_shift(s, n_steps=1),  # Subtle change
        'heavy_pitch': lambda s: augmenter.pitch_shift(s, n_steps=4),   # More noticeable
        'slight_tempo': lambda s: augmenter.time_stretch(s, rate=1.1),  # Slightly faster
        'heavy_tempo': lambda s: augmenter.time_stretch(s, rate=1.3),   # Much faster
        'light_noise': lambda s: augmenter.add_noise(s, noise_type='white', noise_level=0.01),
        'heavy_noise': lambda s: augmenter.add_noise(s, noise_type='white', noise_level=0.05),
        'combined': lambda s: augmenter.add_noise(                      # Multiple effects
            augmenter.time_stretch(
                augmenter.pitch_shift(s, n_steps=2),
                rate=1.2
            ),
            noise_type='white',
            noise_level=0.02
        )
    }
    
    modified_paths = {}
    for mod_name, mod_func in modifications.items():
        output_path = output_dir / f'{mod_name}.wav'
        signal, sr = librosa.load(str(original_path), sr=22050)
        modified_signal = mod_func(signal)
        sf.write(str(output_path), modified_signal, sr)
        modified_paths[mod_name] = output_path
    
    # Copy different file
    different_path = output_dir / 'different.wav'
    shutil.copy(different_pair['clip2_path'], different_path)
    
    return {
        'original': original_path,
        'different': different_path,
        **modified_paths
    }

def convert_to_mp3(wav_path: Path, output_dir: Path):
    """Convert a WAV file to MP3."""
    output_dir.mkdir(exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)
    mp3_path = output_dir / f"{wav_path.stem}.mp3"
    
    audio.export(
        mp3_path,
        format="mp3",
        bitrate="192k",
        parameters=["-q:a", "0"]
    )
    return mp3_path

def create_audio_visualization(audio_path, title="Audio Spectrogram"):
    """Create and plot spectrogram from audio file."""
    # Ensure fixed duration and sample rate
    signal, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    
    # Create mel spectrogram with fixed dimensions
    mel_spec = librosa.feature.melspectrogram(
        y=signal, 
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Enhanced normalization
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Apply more aggressive normalization
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Enhance contrast
    mel_spec_db = np.power(mel_spec_db, 1.5)  # Increase contrast
    
    # Ensure fixed time steps by padding or truncating
    target_width = 130
    current_width = mel_spec_db.shape[1]
    
    # Use center padding/truncating
    if current_width < target_width:
        pad_left = (target_width - current_width) // 2
        pad_right = target_width - current_width - pad_left
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (pad_left, pad_right)))
    elif current_width > target_width:
        start = (current_width - target_width) // 2
        mel_spec_db = mel_spec_db[:, start:start+target_width]
    
    # Plot for visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    return mel_spec_db

def compare_songs(model, audio1_path, audio2_path, output_dir="demo_results"):
    """Compare two songs and create visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create spectrograms
    logger.debug("Creating spectrograms...")
    spec1 = create_audio_visualization(audio1_path, "Song 1")
    plt.savefig(output_dir / "song1_spec.png")
    plt.close()
    
    spec2 = create_audio_visualization(audio2_path, "Song 2")
    plt.savefig(output_dir / "song2_spec.png")
    plt.close()
    
    # Change to debug level
    logger.debug(f"Spectrogram 1 shape: {spec1.shape}, range: [{spec1.min():.3f}, {spec1.max():.3f}]")
    logger.debug(f"Spectrogram 2 shape: {spec2.shape}, range: [{spec2.min():.3f}, {spec2.max():.3f}]")
    
    # Convert to tensors with proper shape
    spec1_tensor = torch.FloatTensor(spec1).unsqueeze(0).unsqueeze(0)
    spec2_tensor = torch.FloatTensor(spec2).unsqueeze(0).unsqueeze(0)
    
    # Normalize each spectrogram independently
    def normalize_tensor(x):
        # Standard normalization
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x
    
    spec1_tensor = normalize_tensor(spec1_tensor)
    spec2_tensor = normalize_tensor(spec2_tensor)
    
    # Debug normalized tensors
    logger.debug(f"Tensor 1 range: [{spec1_tensor.min():.3f}, {spec1_tensor.max():.3f}]")
    logger.debug(f"Tensor 2 range: [{spec2_tensor.min():.3f}, {spec2_tensor.max():.3f}]")
    
    with torch.no_grad():
        # Get embeddings
        embeddings1 = model.get_embedding(spec1_tensor)
        embeddings2 = model.get_embedding(spec2_tensor)
        
        # Calculate similarity metrics
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
        l1_dist = torch.nn.functional.l1_loss(embeddings1, embeddings2).item()
        
        # Combine metrics for final similarity score
        similarity = (cos_sim + (1 - min(l1_dist, 1))) / 2
        
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spec1, sr=22050, x_axis='time', y_axis='mel')
    plt.title("Song 1 Spectrogram")
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(spec2, sr=22050, x_axis='time', y_axis='mel')
    plt.title("Song 2 Spectrogram")
    
    plt.suptitle(f'Similarity Score: {similarity:.2f}')
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png")
    plt.close()
    
    return similarity

def main():
    """Run complete demo pipeline."""
    logger.debug("Starting demo script...")
    
    # Setup paths
    test_csv = Path("data/processed/test_pairs.csv")
    model_path = Path("checkpoints/best_model.pt")
    demo_dir = Path("data/demo")
    mp3_dir = Path("data/demo_mp3")
    results_dir = Path("demo_results")
    
    # Check if required files exist
    if not test_csv.exists():
        logger.error(f"Test CSV file not found: {test_csv}")
        return
        
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    # Create directories
    for dir_path in [demo_dir, mp3_dir, results_dir]:
        dir_path.mkdir(exist_ok=True)
        
    logger.debug("Finding example pairs...")
    file_paths = find_demo_pairs(test_csv)
    
    logger.debug("Converting files to MP3...")
    mp3_files = {
        name: convert_to_mp3(path, mp3_dir)
        for name, path in file_paths.items()
    }
    
    # Step 3: Load model
    logger.debug("Loading model...")
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    logger.info("\nComparing audio modifications (0.0 = different, 1.0 = identical):")
    # Create a more descriptive mapping for the modifications
    mod_descriptions = {
        'different': 'Different song',
        'slight_pitch': 'Pitch shifted up by 1 semitone',
        'heavy_pitch': 'Pitch shifted up by 4 semitones',
        'slight_tempo': 'Tempo increased by 10%',
        'heavy_tempo': 'Tempo increased by 30%',
        'light_noise': 'Light white noise added',
        'heavy_noise': 'Heavy white noise added',
        'combined': 'Combined (pitch + tempo + noise)'
    }
    
    # Print scores with descriptions
    for mod_name, mod_path in file_paths.items():
        if mod_name != 'original':
            similarity = compare_songs(
                model,
                file_paths['original'],
                mod_path,
                results_dir / mod_name
            )
            description = mod_descriptions.get(mod_name, mod_name)
            logger.info(f"{description}: {similarity:.2f}")
    
    logger.info("\nDemo completed. Results saved to demo_results/")
    logger.info(f"MP3 files saved to {mp3_dir}")

if __name__ == "__main__":
    main()