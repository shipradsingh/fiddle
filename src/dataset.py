import pandas as pd
from torch.utils.data import Dataset
from data_processing import preprocess_pair

"""
Custom Dataset for loading a CSV file of audio clip pairs.
Expected format of CSV:
clip1_path,clip2_path,label
Each line defines a training example.
"""
class PairedAudioDataset(Dataset):
    def __init__(self, csv_path):
        self.pairs = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        x1, x2 = preprocess_pair(row['clip1_path'], row['clip2_path'])
        label = row['label']  # label should be 0 or 1
        return x1, x2, label