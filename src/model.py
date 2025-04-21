import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class CNNBranch(nn.Module):
    """CNN feature extractor for audio spectrograms."""
    
    def __init__(self, embedding_dim: int = 128, dropout_rate: float = 0.3):
        super(CNNBranch, self).__init__()
        
        self.conv = nn.Sequential(
            # First conv block: (1, 128, 130) -> (32, 64, 65)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            
            # Second conv block: (32, 64, 65) -> (64, 32, 33)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            
            # Third conv block: (64, 32, 33) -> (128, 16, 17)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            
            nn.Dropout(dropout_rate)
        )
        
        # Calculate flattened size for 128x130 input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 130)
            conv_out = self.conv(dummy)
            flat_size = conv_out.numel()
            logger.debug(f"Flattened size: {flat_size}")
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),  # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape validation
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(
                f"Expected input shape (batch_size, 1, 128, 130), got {x.shape}"
            )
        
        # Add shape debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input shape: {x.shape}")
        
        x = self.conv(x)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"After conv shape: {x.shape}")
            
        x = self.fc(x)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Output shape: {x.shape}")
            
        return x
class SiameseNetwork(nn.Module):
    """Siamese network for audio similarity comparison.
    
    Architecture:
        - Shared CNN branch for feature extraction
        - L1 distance between embeddings
        - Dense layer for similarity prediction
        
    Args:
        embedding_dim: Dimension of the embedding space
        dropout_rate: Dropout rate for regularization
        
    Input shapes: 
        - x1, x2: (batch_size, 1, 128, time)
    Output shape: 
        - similarity: (batch_size, 1)
    """
    def __init__(self, embedding_dim: int = 128, dropout_rate: float = 0.3):
        super(SiameseNetwork, self).__init__()
        
        # Shared CNN branch
        self.branch = CNNBranch(embedding_dim, dropout_rate)
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Similarity head
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """Initialize model weights for better training."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        emb1 = self.branch(x1)
        emb2 = self.branch(x2)
        
        # Compute L1 distance
        diff = torch.abs(emb1 - emb2)
        
        # Apply dropout before final classification
        diff = self.dropout(diff)
        
        # Predict similarity
        return self.out(diff)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single input."""
        return self.branch(x)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test model
    model = SiameseNetwork()
    model.eval()  # Set to evaluation mode
    
    try:
        # Test default batch size
        x1 = torch.randn(32, 1, 128, 130)
        x2 = torch.randn(32, 1, 128, 130)
        
        with torch.no_grad():
            y = model(x1, x2)
            print("\nDefault batch test:")
            print(f"Output shape: {y.shape}")
            print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
            
            emb = model.get_embedding(x1)
            print(f"Embedding shape: {emb.shape}")
            
        # Test different batch sizes
        batch_sizes = [1, 16, 64]
        for batch_size in batch_sizes:
            x1 = torch.randn(batch_size, 1, 128, 130)
            x2 = torch.randn(batch_size, 1, 128, 130)
            
            with torch.no_grad():
                y = model(x1, x2)
                print(f"\nBatch size {batch_size}:")
                print(f"Output shape: {y.shape}")
                print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    else:
        logger.info("All tests passed successfully!")