import pandas as pd
from torch.utils.data import Dataset
from data_processing import preprocess_pair

class PairedAudioDataset(Dataset):
    def __init__(self, csv_path):
        self.pairs = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        x1, x2 = preprocess_pair(row['clip1_path'], row['clip2_path'])
        label = row['label']
        return x1, x2, label
