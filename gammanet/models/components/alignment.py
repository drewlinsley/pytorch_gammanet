"""Distribution alignment layer for bridging fGRU and VGG activation spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionAlignment(nn.Module):
    """Aligns fGRU output distribution to VGG activation space.
    
    This layer transforms the output of fGRU modules (which may be in a 
    different distribution due to normalization and circuit operations) 
    back to the VGG activation distribution (ReLU-based, non-negative).
    
    The alignment consists of:
    1. A 1x1 convolution for channel-wise transformation
    2. Learnable scale and shift parameters for distribution matching
    3. ReLU activation to match VGG's activation pattern
    """
    
    def __init__(self, channels: int, init_scale: float = 1.0, init_shift: float = 0.0):
        """Initialize the distribution alignment layer.
        
        Args:
            channels: Number of input/output channels
            init_scale: Initial value for scale parameter (default: 1.0)
            init_shift: Initial value for shift parameter (default: 0.0)
        """
        super().__init__()
        
        self.channels = channels
        
        # 1x1 convolution for channel-wise transformation
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        # Learnable scale and shift parameters for distribution adjustment
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1) + init_shift)
        
        # Initialize projection to near-identity
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for near-identity transformation."""
        # Initialize projection as identity mapping
        nn.init.eye_(self.projection.weight.data.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of distribution alignment.
        
        Args:
            x: Input tensor from fGRU [B, C, H, W]
            
        Returns:
            Aligned tensor matching VGG activation distribution [B, C, H, W]
        """
        # Apply channel-wise projection
        x = self.projection(x)
        
        # Apply scale and shift for distribution matching
        x = x * self.scale + self.shift
        
        # Apply ReLU to match VGG's activation pattern (non-negative)
        x = F.relu(x)
        
        return x
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'channels={self.channels}'