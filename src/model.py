import torch
import torch.nn as nn

"""
CNN feature extractor block. This is shared across both branches of the Siamese model.
The input should be (batch_size, 1, 128, time). This architecture is simple but effective.
"""
class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128)  # TODO: Adjust this if your input spectrogram shape changes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

"""
Siamese network compares two audio spectrogram embeddings.
Computes the absolute difference between the feature vectors and feeds it to a dense output.
"""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.branch = CNNBranch()
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        emb1 = self.branch(x1)
        emb2 = self.branch(x2)
        diff = torch.abs(emb1 - emb2)
        return self.out(diff)