import torch
import torch.nn as nn
import torch.optim as optim
from model import SiameseNetwork
from dataset import PairedAudioDataset
from torch.utils.data import DataLoader

"""
Training loop for our Siamese model.
Given a DataLoader of audio pairs and similarity labels, this will optimize the model using binary cross entropy.
"""
def train(model, dataloader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1, x2, label in dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device).float()
            optimizer.zero_grad()

            output = model(x1, x2).squeeze()  # Outputs a probability between 0 and 1
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")  # Optional: add val loss tracking here
