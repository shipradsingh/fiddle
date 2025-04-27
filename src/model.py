import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class CNNBranch(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout_rate: float = 0.5):
        super(CNNBranch, self).__init__()
        
        self.conv = nn.Sequential(
            # First conv block with increased regularization
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Added BatchNorm
            nn.Dropout2d(dropout_rate),  # Added 2D dropout
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            # Third conv block for more complexity
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 130)
            conv_out = self.conv(dummy)
            flat_size = conv_out.numel()
        
        # Multi-layer fully connected with residual connections
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, embedding_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected input shape (batch_size, 1, 128, 130), got {x.shape}")
        
        x = self.conv(x)
        return self.fc(x)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout_rate: float = 0.5):
        super(SiameseNetwork, self).__init__()
        
        # Shared CNN branch
        self.branch = CNNBranch(embedding_dim, dropout_rate)
        
        # Multiple dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Enhanced similarity head with multiple layers
        self.similarity_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Get embeddings with dropout
        emb1 = self.dropout1(self.branch(x1))
        emb2 = self.dropout1(self.branch(x2))
        
        # Compute L1 distance
        diff = torch.abs(emb1 - emb2)
        
        # Additional dropout before classification
        diff = self.dropout2(diff)
        
        # Predict similarity
        return self.similarity_head(diff)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)