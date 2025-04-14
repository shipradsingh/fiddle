import librosa
import numpy as np
import torch

"""
Loads audio from a given path with a fixed sample rate and duration.
We may later extend this to handle longer clips or sliding windows.
"""
def load_audio(path, sr=22050, duration=3):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    return y

"""
Converts waveform to Mel Spectrogram in decibel scale.
We normalize using max power, so values are scaled relative to peak.
"""
def extract_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, 
                                    hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

"""
This function loads two audio files, converts them into Mel Spectrograms,
and returns them as PyTorch tensors ready for input into a CNN.
The final shape will be (1, 1, 128, time) per input.
"""
def preprocess_pair(path1, path2):
    y1 = load_audio(path1)
    y2 = load_audio(path2)
    mel1 = extract_mel_spectrogram(y1)
    mel2 = extract_mel_spectrogram(y2)
    mel1 = torch.tensor(mel1).unsqueeze(0).unsqueeze(0)
    mel2 = torch.tensor(mel2).unsqueeze(0).unsqueeze(0)
    return mel1.float(), mel2.float()
