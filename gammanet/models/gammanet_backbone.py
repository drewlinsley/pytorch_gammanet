"""GammaNet with backbone support.

This module implements GammaNet that can work with various backbone architectures,
injecting fGRU layers at specific feature extraction points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .components import fGRU
from .components.normalization import LayerNorm2d, InstanceNorm2d
from .backbones import VGG16Backbone


class GammaNetBackbone(nn.Module):
    """GammaNet with flexible backbone support.
    
    This implementation follows the original TensorFlow version more closely,
    using a pretrained backbone (e.g., VGG16) and injecting fGRU layers at
    specific feature extraction points.
    """
    
    def __init__(
        self,
        config: Dict,
        backbone: Optional[nn.Module] = None,
        input_channels: int = 3,
        output_channels: int = 1,
    ):
        super().__init__()
        
        # Extract configuration
        self.timesteps = config.get('timesteps', 8)
        self.fgru_config = config.get('fgru', {})
        self.normalization = config.get('normalization', 'instance')
        self.activation_type = config.get('activation', 'elu')
        self.skip_connections = config.get('skip_connections', True)
        
        # Get activation
        self.activation = self._get_activation(self.activation_type)
        
        # Initialize backbone
        if backbone is None:
            self.backbone = VGG16Backbone(pretrained=True)
        else:
            self.backbone = backbone
            
        # Get backbone feature info
        feature_info = self.backbone.get_feature_info()
        
        # Create fGRU layers for each feature level
        self.fgru_layers = nn.ModuleDict()
        self.td_projection = nn.ModuleDict()  # Top-down projections
        
        # Build bottom-up fGRU layers
        for i, info in enumerate(feature_info):
            name = info['name']
            channels = info['channels']
            
            # Determine kernel size based on layer depth (following original)
            if i == 0:
                kernel_size = (3, 3)  # Was (3,3) in original for all
            elif i == 1:
                kernel_size = (3, 3)
            elif i == 2:
                kernel_size = (3, 3)
            else:
                kernel_size = (3, 3)
                
            # Create fGRU layer
            self.fgru_layers[f'fgru_{i}'] = fGRU(
                hidden_channels=channels,
                kernel_size=kernel_size,
                **self.fgru_config
            )
            
        # Build top-down connections (decoder path)
        for i in range(len(feature_info) - 1, 0, -1):
            curr_info = feature_info[i]
            prev_info = feature_info[i-1]
            
            # Project from current to previous feature size
            self.td_projection[f'td_{i}_to_{i-1}'] = nn.Sequential(
                nn.Conv2d(curr_info['channels'], prev_info['channels'], 1),
                self._get_norm(self.normalization, prev_info['channels']),
                self.activation
            )
            
        # Output projection - from first feature level
        first_channels = feature_info[0]['channels']
        self.output_projection = nn.Sequential(
            nn.Conv2d(first_channels, first_channels // 2, 3, padding=1),
            self._get_norm(self.normalization, first_channels // 2),
            self.activation,
            nn.Conv2d(first_channels // 2, output_channels, 1)
        )
        
        # Initialize hidden states storage
        self.hidden_states = None
        
    def _get_activation(self, act_type: str):
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif act_type == 'elu':
            return nn.ELU(inplace=True)
        elif act_type == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act_type}")
            
    def _get_norm(self, norm_type: str, channels: int):
        if norm_type == 'instance':
            return InstanceNorm2d(channels, affine=True)
        elif norm_type == 'layer':
            return LayerNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()
            
    def init_hidden_states(self, features: Dict[str, torch.Tensor]):
        """Initialize hidden states based on backbone features."""
        self.hidden_states = {}
        
        # Initialize hidden states for each fGRU layer
        for i, (name, feat) in enumerate(features.items()):
            batch_size = feat.shape[0]
            channels = feat.shape[1]
            height = feat.shape[2]
            width = feat.shape[3]
            
            # Each fGRU needs both h1 and h2 states
            self.hidden_states[f'h_{i}'] = torch.zeros(
                batch_size, channels, height, width, 
                device=feat.device, dtype=feat.dtype
            )
            
    def forward(self, x: torch.Tensor, timesteps: Optional[int] = None) -> torch.Tensor:
        """Forward pass through GammaNet with backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Number of timesteps (defaults to self.timesteps)
            
        Returns:
            Output tensor [B, output_channels, H, W]
        """
        batch_size = x.shape[0]
        timesteps = timesteps or self.timesteps
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Initialize hidden states if needed
        if (self.hidden_states is None or 
            list(self.hidden_states.values())[0].shape[0] != batch_size):
            self.init_hidden_states(backbone_features)
            
        # Convert to list for easier indexing
        feature_names = list(backbone_features.keys())
        feature_list = [backbone_features[name] for name in feature_names]
        
        # Run for multiple timesteps
        for t in range(timesteps):
            # Bottom-up pass with horizontal connections
            for i, (feat_name, features) in enumerate(backbone_features.items()):
                # Get fGRU layer
                fgru = self.fgru_layers[f'fgru_{i}']
                
                # Update hidden state with fGRU
                h_prev = self.hidden_states[f'h_{i}']
                h_new, _ = fgru(features, h_prev)
                self.hidden_states[f'h_{i}'] = h_new
                
            # Top-down pass
            for i in range(len(feature_list) - 1, 0, -1):
                # Get top-down projection
                td_proj = self.td_projection[f'td_{i}_to_{i-1}']
                
                # Project higher level features
                td_features = self.hidden_states[f'h_{i}']
                td_features = td_proj(td_features)
                
                # Resize to match lower level
                target_size = self.hidden_states[f'h_{i-1}'].shape[2:]
                if td_features.shape[2:] != target_size:
                    td_features = F.interpolate(
                        td_features, size=target_size, 
                        mode='bilinear', align_corners=False
                    )
                
                # Apply top-down modulation via fGRU
                fgru = self.fgru_layers[f'fgru_{i-1}']
                h_prev = self.hidden_states[f'h_{i-1}']
                h_new, _ = fgru(td_features, h_prev)
                
                # Combine with skip connection if enabled
                if self.skip_connections:
                    self.hidden_states[f'h_{i-1}'] = h_prev + h_new
                else:
                    self.hidden_states[f'h_{i-1}'] = h_new
                    
        # Output from first (highest resolution) hidden state
        output_features = self.hidden_states['h_0']
        
        # Resize to input resolution
        if output_features.shape[2:] != x.shape[2:]:
            output_features = F.interpolate(
                output_features, size=x.shape[2:], 
                mode='bilinear', align_corners=False
            )
            
        # Final output projection
        output = self.output_projection(output_features)
        
        return output
    
    def reset_hidden_states(self):
        """Reset hidden states between sequences."""
        self.hidden_states = None