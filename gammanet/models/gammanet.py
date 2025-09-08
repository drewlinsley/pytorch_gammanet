"""Main GammaNet architecture implementation.

This module implements the full GammaNet with encoder-decoder structure,
multiple timesteps, and proper hidden state management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .components import fGRU
from .components.normalization import LayerNorm2d, InstanceNorm2d


@dataclass
class LayerConfig:
    """Configuration for a single GammaNet layer."""
    features: int
    pool: bool
    h_kernel: Tuple[int, int]
    ff_kernel: Tuple[int, int] = (3, 3)
    ff_repeats: int = 1
    td: bool = False


class EncoderLayer(nn.Module):
    """Encoder layer with feedforward, pooling, and horizontal connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ff_kernel: Tuple[int, int] = (3, 3),
        ff_repeats: int = 1,
        h_kernel: Tuple[int, int] = (7, 7),
        pool: bool = True,
        pool_kernel: Tuple[int, int] = (2, 2),
        pool_stride: Tuple[int, int] = (2, 2),
        normalization: str = 'instance',
        activation: nn.Module = nn.ELU(),
        use_residual: bool = False,
        **fgru_kwargs
    ):
        super().__init__()
        
        self.pool = pool
        self.ff_repeats = ff_repeats
        self.use_residual = use_residual
        
        # Feedforward path
        ff_layers = []
        for i in range(ff_repeats):
            in_ch = in_channels if i == 0 else out_channels
            ff_layers.extend([
                nn.Conv2d(in_ch, out_channels, ff_kernel, padding=ff_kernel[0]//2),
                self._get_norm(normalization, out_channels),
                activation
            ])
        self.feedforward = nn.Sequential(*ff_layers)
        
        # Horizontal connections (h-fGRU)
        self.h_fgru = fGRU(out_channels, kernel_size=h_kernel, **fgru_kwargs)
        
        # Pooling
        if pool:
            self.pool_layer = nn.MaxPool2d(pool_kernel, pool_stride)
            
    def _get_norm(self, norm_type: str, channels: int):
        if norm_type == 'instance':
            return InstanceNorm2d(channels, affine=True)
        elif norm_type == 'layer':
            return LayerNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()
            
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder layer.
        
        Args:
            x: Input features
            h: Hidden state
            
        Returns:
            x_out: Output features (after optional pooling)
            h_new: Updated hidden state
        """
        # Feedforward processing
        ff_out = self.feedforward(x)
        
        # Update hidden state with horizontal connections
        h_new, _ = self.h_fgru(ff_out, h)
        
        # Apply residual connection if enabled (following original implementation)
        if self.use_residual:
            h_new = h_new + ff_out
        
        # Pool if needed
        x_out = self.pool_layer(h_new) if self.pool else h_new
        
        return x_out, h_new


class DecoderLayer(nn.Module):
    """Decoder layer with upsampling and top-down connections.
    
    According to the paper, decoder layers only receive top-down signals
    and modulate encoder hidden states. No skip connections are used.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: Optional[int] = None,  # Kept for compatibility but ignored
        ff_kernel: Tuple[int, int] = (3, 3),
        ff_repeats: int = 1,
        up_kernel: Tuple[int, int] = (4, 4),
        up_stride: Tuple[int, int] = (2, 2),
        normalization: str = 'instance',
        activation: nn.Module = nn.ELU(),
        **fgru_kwargs
    ):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, up_kernel, up_stride, 
            padding=up_kernel[0]//2, output_padding=0
        )
        self.up_kernel = up_kernel
        self.up_stride = up_stride
        
        # Feedforward processing (no skip connections)
        ff_layers = []
        for i in range(ff_repeats):
            in_ch = out_channels if i == 0 else out_channels
            ff_layers.extend([
                nn.Conv2d(in_ch, out_channels, ff_kernel, padding=ff_kernel[0]//2),
                self._get_norm(normalization, out_channels),
                activation
            ])
        self.feedforward = nn.Sequential(*ff_layers)
        
        # Top-down connections (td-fGRU)
        self.td_fgru = fGRU(out_channels, kernel_size=(1, 1), **fgru_kwargs)
        
    def _get_norm(self, norm_type: str, channels: int):
        if norm_type == 'instance':
            return InstanceNorm2d(channels, affine=True)
        elif norm_type == 'layer':
            return LayerNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()
            
    def forward(self, x: torch.Tensor, encoder_h: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder layer.
        
        Args:
            x: Input from higher layer (top-down signal)
            encoder_h: Encoder hidden state to modulate
            
        Returns:
            encoder_h_new: Updated encoder hidden state
        """
        # Apply learned upsampling first
        x = self.upsample(x)
        
        # Then ensure exact size match with encoder hidden state
        target_size = (encoder_h.shape[2], encoder_h.shape[3])
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Feedforward processing of top-down signal
        x = self.feedforward(x)
        
        # Update encoder hidden state via top-down connections
        # Per paper: Z = encoder_h (horizontal state), H = x (top-down state)
        encoder_h_new, _ = self.td_fgru(x, encoder_h)
        
        return encoder_h_new


class GammaNet(nn.Module):
    """Full GammaNet architecture with timestep loops.
    
    The network processes input through multiple timesteps:
    1. Bottom-up pass through encoder layers
    2. Top-down pass through decoder layers that modulate encoder hidden states
    """
    
    def __init__(
        self,
        config: Dict,
        input_channels: int = 3,
        output_channels: int = 1,
    ):
        super().__init__()
        
        # Extract configuration
        self.timesteps = config.get('timesteps', 8)
        self.layers_config = config.get('layers', self._default_layers())
        normalization = config.get('normalization', 'instance')
        activation_type = config.get('activation', 'elu')
        fgru_config = config.get('fgru', {})
        self.residual_connections = config.get('residual_connections', False)
        
        # Get activation
        self.activation = self._get_activation(activation_type)
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.layers_config[0]['features'], 3, padding=1),
            self._get_norm(normalization, self.layers_config[0]['features']),
            self.activation
        )
        
        # Dynamically determine encoder/decoder split
        # Encoder layers are those with 'pool' or before the bottleneck
        # Decoder layers are after the bottleneck (no pooling)
        encoder_count = 0
        for i, layer_cfg in enumerate(self.layers_config):
            if layer_cfg.get('pool', False) or (i > 0 and self.layers_config[i-1].get('pool', False)):
                encoder_count = i + 1
        
        # If all layers have pooling, split in the middle
        if encoder_count == 0 or encoder_count == len(self.layers_config):
            encoder_count = len(self.layers_config) // 2
        
        self.encoder_count = encoder_count
        self.decoder_count = len(self.layers_config) - encoder_count
        
        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        in_ch = self.layers_config[0]['features']
        
        for i, layer_cfg in enumerate(self.layers_config[:encoder_count]):  # Dynamic encoder count
            self.encoder_layers.append(
                EncoderLayer(
                    in_channels=in_ch,
                    out_channels=layer_cfg['features'],
                    ff_kernel=layer_cfg.get('ff_kernel', (3, 3)),
                    ff_repeats=layer_cfg.get('ff_repeats', 1),
                    h_kernel=layer_cfg['h_kernel'],
                    pool=layer_cfg['pool'],
                    normalization=normalization,
                    activation=self.activation,
                    use_residual=self.residual_connections,
                    **fgru_config
                )
            )
            in_ch = layer_cfg['features']
            
        # Build decoder layers
        self.decoder_layers = nn.ModuleList()
        decoder_configs = self.layers_config[encoder_count:]  # Dynamic decoder layers
        
        # Create encoder-decoder layer mapping
        # Each decoder layer modulates the corresponding encoder layer
        # decoder 0 -> encoder (encoder_count-2), decoder 1 -> encoder (encoder_count-3), etc.
        self.decoder_to_encoder_mapping = {}
        for i in range(self.decoder_count):
            # Map decoder i to encoder (encoder_count - 2 - i)
            # This skips the bottleneck encoder layer
            encoder_idx = encoder_count - 2 - i
            if encoder_idx >= 0:
                self.decoder_to_encoder_mapping[i] = encoder_idx
        
        for i, layer_cfg in enumerate(decoder_configs):
            # Remove skip connections based on paper description
            # Decoder only receives top-down signal and modulates encoder hidden states
            
            self.decoder_layers.append(
                DecoderLayer(
                    in_channels=in_ch,
                    out_channels=layer_cfg['features'],
                    skip_channels=None,  # No skip connections per paper
                    ff_kernel=layer_cfg.get('ff_kernel', (3, 3)),
                    ff_repeats=layer_cfg.get('ff_repeats', 1),
                    normalization=normalization,
                    activation=self.activation,
                    **fgru_config
                )
            )
            in_ch = layer_cfg['features']
            
        # Output projection
        self.output_conv = nn.Conv2d(
            self.layers_config[0]['features'], output_channels, 1
        )
        
        # Initialize hidden states storage
        self.hidden_states = None
        
    def _default_layers(self) -> List[Dict]:
        """Default layer configuration matching original GammaNet."""
        return [
            {'features': 24, 'pool': True,  'h_kernel': (9, 9)},
            {'features': 28, 'pool': True,  'h_kernel': (7, 7)},
            {'features': 36, 'pool': True,  'h_kernel': (5, 5)},
            {'features': 48, 'pool': True,  'h_kernel': (3, 3)},
            {'features': 64, 'pool': False, 'h_kernel': (1, 1)},
            {'features': 48, 'pool': False, 'h_kernel': (1, 1)},
            {'features': 36, 'pool': False, 'h_kernel': (1, 1)},
            {'features': 28, 'pool': False, 'h_kernel': (1, 1)},
            {'features': 24, 'pool': False, 'h_kernel': (1, 1)},
        ]
        
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
            
    def init_hidden_states(self, batch_size: int, height: int, width: int, device: torch.device):
        """Initialize hidden states for all encoder layers."""
        self.hidden_states = []
        
        # Calculate sizes for each layer
        h, w = height, width
        
        for layer in self.encoder_layers:
            self.hidden_states.append(
                torch.zeros(batch_size, layer.h_fgru.hidden_channels, h, w).to(device)
            )
            if layer.pool:
                h, w = h // 2, w // 2
                
    def forward(self, x: torch.Tensor, timesteps: Optional[int] = None) -> torch.Tensor:
        """Forward pass through GammaNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Number of timesteps (defaults to self.timesteps)
            
        Returns:
            Output tensor [B, output_channels, H, W]
        """
        batch_size, _, height, width = x.shape
        
        # Initialize hidden states if needed or if batch size/dimensions changed
        if (self.hidden_states is None or 
            len(self.hidden_states) == 0 or
            self.hidden_states[0].shape[0] != batch_size or
            self.hidden_states[0].shape[2] != height or
            self.hidden_states[0].shape[3] != width):
            self.init_hidden_states(batch_size, height, width, x.device)
            
        # Initial input processing
        x_init = self.input_conv(x)
        
        # Run for multiple timesteps
        timesteps = timesteps or self.timesteps
        
        for t in range(timesteps):
            # Bottom-up pass (encoder)
            x_current = x_init
            encoder_outputs = []
            
            for i, layer in enumerate(self.encoder_layers):
                x_current, self.hidden_states[i] = layer(x_current, self.hidden_states[i])
                encoder_outputs.append(self.hidden_states[i])
                
            # Top-down pass (decoder)
            for i, layer in enumerate(self.decoder_layers):
                # Use dynamic mapping to find which encoder layer to modulate
                if i in self.decoder_to_encoder_mapping:
                    encoder_idx = self.decoder_to_encoder_mapping[i]
                    
                    # Update encoder hidden state via top-down connection
                    # No skip connections - decoder only receives top-down signal
                    self.hidden_states[encoder_idx] = layer(
                        x_current, self.hidden_states[encoder_idx]
                    )
                    
                    # Use updated hidden state as input to next decoder
                    x_current = self.hidden_states[encoder_idx]
                else:
                    # If no mapping exists (shouldn't happen), just process through decoder
                    x_current = layer(x_current, x_current)
                
        # Final output projection
        output = self.output_conv(self.hidden_states[0])
        
        return output
    
    def reset_hidden_states(self):
        """Reset hidden states between sequences."""
        self.hidden_states = None