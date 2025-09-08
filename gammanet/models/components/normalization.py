"""Custom normalization layers for GammaNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D convolutional features.
    
    PyTorch's LayerNorm expects channels last, but we use channels first.
    This wrapper handles the conversion.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            
        return x


class InstanceNorm2d(nn.InstanceNorm2d):
    """Instance normalization with optional learnable parameters.
    
    Extends PyTorch's InstanceNorm2d with better initialization.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = False, track_running_stats: bool = False):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        # Better initialization if affine
        if affine:
            nn.init.constant_(self.weight, 1.0)
            nn.init.constant_(self.bias, 0.0)