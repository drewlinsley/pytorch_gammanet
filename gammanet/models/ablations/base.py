"""Base classes for model ablations.

This module provides base classes and utilities for creating ablated
versions of GammaNet to study the contribution of different components.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class AblationMixin(ABC):
    """Mixin class for model ablations.
    
    This should be used with multiple inheritance alongside GammaNet.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize ablation."""
        # Store ablation config before parent init
        self.ablation_config = self._get_ablation_config()
        
        # Call parent class init
        super().__init__(*args, **kwargs)
        
        # Apply ablation after model is built
        self._apply_ablation()
        
    @abstractmethod
    def _get_ablation_config(self) -> Dict:
        """Get ablation configuration.
        
        Returns:
            Dictionary with ablation parameters
        """
        pass
    
    @abstractmethod
    def _apply_ablation(self) -> None:
        """Apply ablation to the model.
        
        This method should modify the model to implement the ablation.
        """
        pass
    
    def get_ablation_info(self) -> Dict:
        """Get information about the ablation.
        
        Returns:
            Dictionary with ablation details
        """
        return {
            "class": self.__class__.__name__,
            "config": self.ablation_config,
            "description": self.__doc__
        }


class ParameterAblationMixin(AblationMixin):
    """Mixin for ablations that modify fGRU parameters."""
    
    def _modify_fgru_params(self, 
                           multiplicative: Optional[bool] = None,
                           additive: Optional[bool] = None,
                           divisive: Optional[bool] = None,
                           gates: Optional[bool] = None) -> None:
        """Modify fGRU parameters across all layers.
        
        Args:
            multiplicative: Whether to enable multiplicative interactions
            additive: Whether to enable additive interactions  
            divisive: Whether to enable divisive normalization
            gates: Whether to enable gates
        """
        # Modify encoder fGRUs
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._modify_single_fgru(layer.fgru, multiplicative, 
                                       additive, divisive, gates)
                
        # Modify decoder fGRUs if they exist
        if hasattr(self, 'decoder_layers'):
            for layer in self.decoder_layers:
                if hasattr(layer, 'fgru'):
                    self._modify_single_fgru(layer.fgru, multiplicative,
                                           additive, divisive, gates)
    
    def _modify_single_fgru(self,
                           fgru: nn.Module,
                           multiplicative: Optional[bool] = None,
                           additive: Optional[bool] = None,
                           divisive: Optional[bool] = None,
                           gates: Optional[bool] = None) -> None:
        """Modify a single fGRU module."""
        if multiplicative is not None:
            if not multiplicative:
                # Set omega to 0 to disable multiplicative
                if hasattr(fgru, 'omega'):
                    nn.init.zeros_(fgru.omega.weight)
                    fgru.omega.weight.requires_grad = False
                    
        if additive is not None:
            if not additive:
                # Set kappa to 0 to disable additive
                if hasattr(fgru, 'kappa'):
                    nn.init.zeros_(fgru.kappa.weight)
                    fgru.kappa.weight.requires_grad = False
                    
        if divisive is not None:
            if not divisive:
                # Set alpha to 0 to disable divisive normalization
                if hasattr(fgru, 'alpha'):
                    nn.init.zeros_(fgru.alpha.weight)
                    fgru.alpha.weight.requires_grad = False
                    
        if gates is not None:
            if not gates:
                # Override gate computation in forward pass
                # This requires modifying the forward method
                self._override_gates(fgru)
    
    def _override_gates(self, fgru: nn.Module) -> None:
        """Override gate computation to use fixed values."""
        # Store original forward method
        original_forward = fgru.forward
        
        def forward_with_fixed_gates(ff_input, h_prev, td_input=None):
            # Temporarily override gate computation methods
            original_compute_input_gate = fgru.compute_input_gate
            original_compute_output_gate = fgru.compute_output_gate
            
            # Fixed gate values
            def fixed_input_gate(h, td=None):
                return torch.ones_like(h)
            
            def fixed_output_gate(h):
                return torch.ones_like(h) * 0.5
            
            fgru.compute_input_gate = fixed_input_gate
            fgru.compute_output_gate = fixed_output_gate
            
            # Run forward pass
            result = original_forward(ff_input, h_prev, td_input)
            
            # Restore original methods
            fgru.compute_input_gate = original_compute_input_gate
            fgru.compute_output_gate = original_compute_output_gate
            
            return result
        
        fgru.forward = forward_with_fixed_gates


class ConnectivityAblationMixin(AblationMixin):
    """Mixin for ablations that modify connectivity patterns."""
    
    def _disable_horizontal_connections(self) -> None:
        """Disable horizontal connections in fGRU."""
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                # Override fGRU forward to skip horizontal processing
                self._override_horizontal(layer.fgru)
                
    def _disable_topdown_connections(self) -> None:
        """Disable top-down connections."""
        # Override decoder forward passes to skip td modulation
        if hasattr(self, 'decoder_layers'):
            for layer in self.decoder_layers:
                self._override_topdown(layer)
                
    def _override_horizontal(self, fgru: nn.Module) -> None:
        """Override fGRU to disable horizontal connections."""
        original_forward = fgru.forward
        
        def forward_without_horizontal(ff_input, h_prev, td_input=None):
            # Simply return feedforward input as next state
            # This effectively disables horizontal connections
            return ff_input, ff_input
        
        fgru.forward = forward_without_horizontal
        
    def _override_topdown(self, decoder_layer: nn.Module) -> None:
        """Override decoder layer to disable top-down modulation."""
        original_forward = decoder_layer.forward
        
        def forward_without_topdown(x, encoder_state, skip_connection):
            # Ignore the top-down input (x) and just process encoder state
            return encoder_state
        
        decoder_layer.forward = forward_without_topdown


class TimestepAblationMixin(AblationMixin):
    """Mixin for ablations that modify timestep processing."""
    
    def _set_timesteps(self, timesteps: int) -> None:
        """Set number of timesteps."""
        self.timesteps = timesteps
        
    def _override_timestep_loop(self) -> None:
        """Override forward to use single timestep."""
        original_forward = self.forward
        
        def forward_single_timestep(x):
            # Store original timesteps
            original_timesteps = self.timesteps
            
            # Set to single timestep
            self.timesteps = 1
            
            # Run forward
            result = original_forward(x)
            
            # Restore timesteps
            self.timesteps = original_timesteps
            
            return result
            
        self.forward = forward_single_timestep