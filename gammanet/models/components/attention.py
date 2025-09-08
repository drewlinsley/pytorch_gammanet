"""Attention mechanisms for GammaNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GALABlock(nn.Module):
    """Global and Local Attention block.
    
    Combines global (channel) and local (spatial) attention mechanisms.
    Based on the GALA attention from the original GammaNet.
    """
    
    def __init__(self, channels: int, layers: int = 1, kernel_size: int = 5):
        super().__init__()
        self.layers = layers
        
        # Global attention (channel-wise)
        self.global_layers = nn.ModuleList()
        for _ in range(layers):
            self.global_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1, bias=True),
                    nn.LayerNorm([channels, 1, 1]) if layers > 1 else nn.Identity()
                )
            )
            
        # Spatial attention
        self.spatial_layers = nn.ModuleList()
        padding = kernel_size // 2
        for _ in range(layers):
            self.spatial_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True),
                    nn.LayerNorm([channels, 1, 1]) if layers > 1 else nn.Identity()
                )
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global attention path
        global_feat = F.adaptive_max_pool2d(x, 1)
        for layer in self.global_layers[:-1]:
            global_feat = F.relu(layer(global_feat))
        global_feat = self.global_layers[-1](global_feat)
        
        # Spatial attention path
        spatial_feat = x
        for layer in self.spatial_layers[:-1]:
            spatial_feat = F.relu(layer(spatial_feat))
        spatial_feat = self.spatial_layers[-1](spatial_feat)
        
        # Combine global and spatial
        attention = torch.sigmoid(global_feat * spatial_feat)
        
        return x * attention