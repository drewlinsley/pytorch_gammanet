"""VGG16 backbone for GammaNet.

This module provides a VGG16 backbone that outputs intermediate features
at specific layers for fGRU injection, similar to the original TensorFlow
implementation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


class VGG16Backbone(nn.Module):
    """VGG16 backbone that outputs features at multiple stages.
    
    Following the original implementation, we extract features after:
    - conv2_2 (layer index 8)
    - conv3_3 (layer index 15) 
    - conv4_3 (layer index 22)
    - conv5_3 (layer index 29)
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract feature layers (up to conv5_3)
        self.features = vgg16.features[:30]  # Up to and including conv5_3
        
        # Define extraction points (after ReLU activations)
        self.extraction_points = {
            'conv2_2': 8,   # After conv2_2 + ReLU
            'conv3_3': 15,  # After conv3_3 + ReLU  
            'conv4_3': 22,  # After conv4_3 + ReLU
            'conv5_3': 29   # After conv5_3 + ReLU
        }
        
        # Feature channels at each extraction point
        self.feature_channels = {
            'conv2_2': 128,
            'conv3_3': 256,
            'conv4_3': 512,
            'conv5_3': 512
        }
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass extracting features at multiple stages.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        features = {}
        
        # Run through VGG layers and extract at specified points
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Check if this is an extraction point
            for name, idx in self.extraction_points.items():
                if i == idx:
                    features[name] = x
                    
        return features
    
    def get_feature_info(self) -> List[Dict]:
        """Get information about feature extraction points.
        
        Returns:
            List of dicts with keys: name, channels, stride
        """
        # Strides are cumulative from pooling layers
        info = [
            {'name': 'conv2_2', 'channels': 128, 'stride': 2},   # After pool1
            {'name': 'conv3_3', 'channels': 256, 'stride': 4},   # After pool2
            {'name': 'conv4_3', 'channels': 512, 'stride': 8},   # After pool3
            {'name': 'conv5_3', 'channels': 512, 'stride': 16},  # After pool4
        ]
        return info