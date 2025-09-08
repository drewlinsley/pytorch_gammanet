"""Connectivity ablation models for GammaNet.

These models test the importance of different connectivity patterns
by selectively disabling horizontal, top-down, or recurrent connections.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..gammanet import GammaNet
from .base import ConnectivityAblationMixin, TimestepAblationMixin


class GammaNetFFOnly(ConnectivityAblationMixin, GammaNet):
    """GammaNet with only feedforward connections (no horizontal recurrence).
    
    This ablation disables all horizontal connections in the fGRU units,
    effectively making the model purely feedforward through layers while
    maintaining the encoder-decoder structure.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": False,
            "topdown_connections": True,
            "recurrent_timesteps": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable horizontal connections."""
        self._disable_horizontal_connections()
        
    def _disable_horizontal_connections(self) -> None:
        """Override to properly disable horizontal connections."""
        # For each encoder layer
        for i, layer in enumerate(self.encoder_layers):
            if hasattr(layer, 'fgru'):
                original_forward = layer.fgru.forward
                
                def make_ff_forward(original_fn):
                    def forward_ff_only(ff_input, h_prev, td_input=None):
                        # Skip all horizontal processing
                        # Just pass through feedforward input
                        return ff_input, ff_input
                    return forward_ff_only
                
                layer.fgru.forward = make_ff_forward(original_forward)


class GammaNetHOnly(ConnectivityAblationMixin, GammaNet):
    """GammaNet with only horizontal connections (no top-down feedback).
    
    This ablation disables all top-down connections from decoder to encoder,
    testing the importance of feedback for computation.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": True,
            "topdown_connections": False,
            "recurrent_timesteps": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable top-down connections."""
        # Override the forward method to skip decoder
        original_forward = self.forward
        
        def forward_without_topdown(x):
            # Initialize hidden states
            self.reset_hidden_states()
            
            # Initial feedforward pass
            x_init = self.stem(x)
            outputs = []
            
            # Run timesteps with only encoder (horizontal connections)
            for t in range(self.timesteps):
                x_current = x_init
                
                # Bottom-up pass through encoder only
                for i, layer in enumerate(self.encoder_layers):
                    x_current, self.hidden_states[i] = layer(
                        x_current, self.hidden_states[i]
                    )
                    
                outputs.append(x_current)
            
            # Use last timestep encoder output
            x_out = outputs[-1]
            
            # Final readout
            x_out = self.readout_norm(x_out)
            x_out = self.readout(x_out)
            
            return x_out
        
        self.forward = forward_without_topdown


class GammaNetTDOnly(ConnectivityAblationMixin, GammaNet):
    """GammaNet with only top-down connections (no horizontal recurrence).
    
    This ablation disables horizontal connections but keeps top-down feedback,
    testing whether feedback alone can support computation.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": False, 
            "topdown_connections": True,
            "recurrent_timesteps": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable horizontal but keep top-down."""
        # Disable horizontal connections in encoder fGRUs
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                original_forward = layer.fgru.forward
                
                def make_td_only_forward(original_fn):
                    def forward_td_only(ff_input, h_prev, td_input=None):
                        # If no top-down input, just pass through
                        if td_input is None:
                            return ff_input, ff_input
                        
                        # Apply only top-down modulation
                        # Skip horizontal processing
                        modulated = ff_input + td_input
                        return modulated, modulated
                    return forward_td_only
                
                layer.fgru.forward = make_td_only_forward(original_forward)


class GammaNetNoRecurrence(TimestepAblationMixin, GammaNet):
    """GammaNet with no recurrence (single forward pass).
    
    This ablation runs only a single timestep, effectively making the model
    non-recurrent while maintaining all connectivity patterns for that step.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": True,
            "topdown_connections": True, 
            "recurrent_timesteps": False,
            "num_timesteps": 1
        }
    
    def _apply_ablation(self) -> None:
        """Force single timestep."""
        self._set_timesteps(1)


class GammaNetBottomUpOnly(ConnectivityAblationMixin, GammaNet):
    """Pure bottom-up processing (no horizontal or top-down).
    
    This creates a standard feedforward CNN by disabling both
    horizontal and top-down connections.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": False,
            "topdown_connections": False,
            "recurrent_timesteps": False
        }
    
    def _apply_ablation(self) -> None:
        """Disable all recurrent and top-down connections."""
        # Override entire forward pass
        original_forward = self.forward
        
        def forward_bottom_up_only(x):
            # Simple feedforward through encoder
            x = self.stem(x)
            
            # Single pass through encoder layers (no recurrence)
            for layer in self.encoder_layers:
                # Extract conv layers, skip fGRU
                if hasattr(layer, 'conv'):
                    x = layer.conv(x)
                    if hasattr(layer, 'norm'):
                        x = layer.norm(x)
                else:
                    # If no conv, just pass through
                    x = layer(x, x)[0]  # Dummy call
                    
            # Readout
            x = self.readout_norm(x)
            x = self.readout(x)
            
            return x
            
        self.forward = forward_bottom_up_only


class GammaNetDelayedTD(ConnectivityAblationMixin, GammaNet):
    """GammaNet with delayed top-down connections.
    
    Top-down connections are only active after a certain number of timesteps,
    testing the temporal dynamics of feedback.
    """
    
    def __init__(self, config: Dict, delay_timesteps: int = 3, **kwargs):
        self.delay_timesteps = delay_timesteps
        super().__init__(config, **kwargs)
    
    def _get_ablation_config(self) -> Dict:
        return {
            "horizontal_connections": True,
            "topdown_connections": True,
            "recurrent_timesteps": True,
            "td_delay": self.delay_timesteps
        }
    
    def _apply_ablation(self) -> None:
        """Modify forward to delay top-down activation."""
        original_forward = self.forward
        
        def forward_with_delayed_td(x):
            # Initialize hidden states
            self.reset_hidden_states()
            
            # Initial feedforward pass
            x_init = self.stem(x)
            outputs = []
            
            # Process timesteps
            for t in range(self.timesteps):
                x_current = x_init
                
                # Bottom-up pass
                encoder_outputs = []
                for i, layer in enumerate(self.encoder_layers):
                    x_current, self.hidden_states[i] = layer(
                        x_current, self.hidden_states[i]
                    )
                    encoder_outputs.append(x_current)
                
                # Top-down pass (only after delay)
                if t >= self.delay_timesteps:
                    # Normal top-down processing
                    for i, layer in enumerate(self.decoder_layers):
                        encoder_idx = len(self.decoder_layers) - i - 1
                        skip = encoder_outputs[encoder_idx]
                        
                        # Top-down modulation
                        self.hidden_states[encoder_idx] = layer(
                            x_current, self.hidden_states[encoder_idx], skip
                        )
                        
                        if i < len(self.decoder_layers) - 1:
                            x_current = self.hidden_states[encoder_idx]
                
                outputs.append(self.hidden_states[0])
            
            # Use last timestep output
            x_out = outputs[-1]
            
            # Final readout
            x_out = self.readout_norm(x_out)
            x_out = self.readout(x_out)
            
            return x_out
            
        self.forward = forward_with_delayed_td