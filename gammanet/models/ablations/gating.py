"""Gating ablation models for GammaNet.

These models test the importance of different gating mechanisms and
arithmetic operations in the fGRU circuit.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..gammanet import GammaNet  
from .base import ParameterAblationMixin


class GammaNetAdditiveOnly(ParameterAblationMixin, GammaNet):
    """GammaNet with only additive interactions (no multiplicative gating).
    
    This ablation removes multiplicative interactions (ω=0) to test whether
    additive mechanisms alone are sufficient.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": False,
            "additive_excitation": True,
            "divisive_normalization": True,
            "gates": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable multiplicative interactions."""
        self._modify_fgru_params(multiplicative=False)
        
        # Also need to modify the output integration
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._modify_output_integration(layer.fgru)
    
    def _modify_output_integration(self, fgru: nn.Module) -> None:
        """Modify output integration to remove multiplicative term."""
        original_output_integration = fgru.output_integration
        
        def additive_only_integration(h1, c2, g2, h_prev):
            """Output integration with only additive excitation."""
            # Original: h_new = kappa * (c2 + h1) + omega * (c2 * h1)
            # Modified: h_new = kappa * (c2 + h1)  (omega = 0)
            
            # Get kappa parameter
            kappa = torch.sigmoid(fgru.kappa(h1))
            
            # Additive only
            h_new = kappa * (c2 + h1)
            
            # Apply output gate
            h_out = g2 * h_new + (1 - g2) * h_prev
            
            return h_out
        
        fgru.output_integration = additive_only_integration


class GammaNetMultiplicativeOnly(ParameterAblationMixin, GammaNet):
    """GammaNet with only multiplicative interactions (no additive term).
    
    This ablation removes additive interactions (κ=0) to test whether
    multiplicative gating alone is sufficient.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": True,
            "additive_excitation": False,
            "divisive_normalization": True,
            "gates": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable additive interactions."""
        self._modify_fgru_params(additive=False)
        
        # Modify output integration
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._modify_output_integration_multiplicative(layer.fgru)
    
    def _modify_output_integration_multiplicative(self, fgru: nn.Module) -> None:
        """Modify output integration to remove additive term."""
        original_output_integration = fgru.output_integration
        
        def multiplicative_only_integration(h1, c2, g2, h_prev):
            """Output integration with only multiplicative excitation."""
            # Original: h_new = kappa * (c2 + h1) + omega * (c2 * h1)
            # Modified: h_new = omega * (c2 * h1)  (kappa = 0)
            
            # Get omega parameter
            omega = torch.sigmoid(fgru.omega(h1))
            
            # Multiplicative only
            h_new = omega * (c2 * h1)
            
            # Apply output gate
            h_out = g2 * h_new + (1 - g2) * h_prev
            
            return h_out
        
        fgru.output_integration = multiplicative_only_integration


class GammaNetNoGates(ParameterAblationMixin, GammaNet):
    """GammaNet with fixed gates (no learned gating).
    
    This ablation uses fixed gate values (g1=1, g2=0.5) to test
    the importance of dynamic gating.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": True,
            "additive_excitation": True,
            "divisive_normalization": True,
            "gates": False,
            "fixed_g1": 1.0,
            "fixed_g2": 0.5
        }
    
    def _apply_ablation(self) -> None:
        """Replace dynamic gates with fixed values."""
        self._modify_fgru_params(gates=False)
        
        # Override gate computation
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._set_fixed_gates(layer.fgru)
                
    def _set_fixed_gates(self, fgru: nn.Module) -> None:
        """Set fixed gate values."""
        # Override gate computation methods
        def fixed_input_gate(h_prev, td_input=None):
            # g1 = 1.0 (full mixing)
            return torch.ones_like(h_prev)
        
        def fixed_output_gate(h1):
            # g2 = 0.5 (balanced mixing)
            return torch.ones_like(h1) * 0.5
            
        fgru.compute_input_gate = fixed_input_gate
        fgru.compute_output_gate = fixed_output_gate


class GammaNetNoDivisive(ParameterAblationMixin, GammaNet):
    """GammaNet without divisive normalization.
    
    This ablation removes divisive normalization (α=0) to test its
    importance for gain control.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": True,
            "additive_excitation": True,
            "divisive_normalization": False,
            "gates": True
        }
    
    def _apply_ablation(self) -> None:
        """Disable divisive normalization."""
        self._modify_fgru_params(divisive=False)
        
        # Modify circuit input to remove divisive term
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._modify_circuit_input(layer.fgru)
                
    def _modify_circuit_input(self, fgru: nn.Module) -> None:
        """Remove divisive normalization from circuit input."""
        original_circuit_input = fgru.circuit_input
        
        def circuit_input_no_divisive(h_prev, g1):
            """Circuit input without divisive normalization."""
            # Original: c1 = g1 * gamma(h_prev) - mu(h_prev) / (alpha(h_prev) + eps)
            # Modified: c1 = g1 * gamma(h_prev) - mu(h_prev)
            
            # Inhibitory components
            if hasattr(fgru, 'gamma') and fgru.gamma is not None:
                gamma_h = fgru.gamma(h_prev)
            else:
                gamma_h = h_prev
                
            # Subtractive inhibition
            mu_h = fgru.mu(h_prev)
            
            # No divisive term
            c1 = g1 * gamma_h - mu_h
            
            return c1
            
        fgru.circuit_input = circuit_input_no_divisive


class GammaNetLinearGates(ParameterAblationMixin, GammaNet):
    """GammaNet with linear (non-sigmoid) gates.
    
    This tests whether the nonlinearity in gate computation is important.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": True,
            "additive_excitation": True,
            "divisive_normalization": True,
            "gates": True,
            "gate_activation": "linear"
        }
    
    def _apply_ablation(self) -> None:
        """Use linear activation for gates."""
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._make_gates_linear(layer.fgru)
                
    def _make_gates_linear(self, fgru: nn.Module) -> None:
        """Replace sigmoid with linear activation for gates."""
        original_compute_input_gate = fgru.compute_input_gate
        original_compute_output_gate = fgru.compute_output_gate
        
        def linear_input_gate(h_prev, td_input=None):
            """Compute input gate with linear activation."""
            # Compute gate inputs
            gate_input = fgru.g1_conv(h_prev)
            
            if td_input is not None and hasattr(fgru, 'td_gate') and fgru.td_gate:
                gate_input = gate_input + td_input
                
            # Linear activation (clipped to [0, 1])
            g1 = torch.clamp(gate_input, 0, 1)
            
            return g1
            
        def linear_output_gate(h1):
            """Compute output gate with linear activation."""
            gate_input = fgru.g2_conv(h1)
            
            # Linear activation (clipped to [0, 1])
            g2 = torch.clamp(gate_input, 0, 1)
            
            return g2
            
        fgru.compute_input_gate = linear_input_gate
        fgru.compute_output_gate = linear_output_gate


class GammaNetSymmetricGates(ParameterAblationMixin, GammaNet):
    """GammaNet with symmetric input and output gates (g1 = g2).
    
    This tests whether having different input/output gates is important.
    """
    
    def _get_ablation_config(self) -> Dict:
        return {
            "multiplicative_excitation": True,
            "additive_excitation": True,
            "divisive_normalization": True,
            "gates": True,
            "symmetric_gates": True
        }
    
    def _apply_ablation(self) -> None:
        """Make output gate same as input gate."""
        for layer in self.encoder_layers:
            if hasattr(layer, 'fgru'):
                self._make_gates_symmetric(layer.fgru)
                
    def _make_gates_symmetric(self, fgru: nn.Module) -> None:
        """Make g2 = g1."""
        original_compute_output_gate = fgru.compute_output_gate
        
        # Store reference to input gate computation
        compute_input_gate_ref = fgru.compute_input_gate
        
        def symmetric_output_gate(h1):
            """Output gate equals input gate."""
            # Use the same computation as input gate
            # Note: this is a simplification - in practice might want
            # to use the stored g1 value
            g2 = compute_input_gate_ref(h1, None)
            return g2
            
        fgru.compute_output_gate = symmetric_output_gate