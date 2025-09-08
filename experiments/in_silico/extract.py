"""Response extraction utilities for in silico experiments.

This module provides functions to extract neural-like responses from trained
GammaNet models for comparison with neurophysiology data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from torch.utils.hooks import RemovableHandle


@dataclass
class LayerResponse:
    """Container for layer responses."""
    layer_name: str
    timestep: int
    activation: torch.Tensor  # Shape: [batch, channels, height, width]
    receptive_field_size: Optional[int] = None
    layer_type: str = "unknown"  # "encoder", "decoder", "hgru", "tdgru"


class ResponseExtractor:
    """Extract responses from GammaNet layers."""
    
    def __init__(self, model: nn.Module):
        """Initialize extractor with model.
        
        Args:
            model: Trained GammaNet model
        """
        self.model = model
        self.model.eval()
        self.hooks = []
        self.responses = {}
        self.layer_info = self._analyze_model_structure()
        
    def _analyze_model_structure(self) -> Dict[str, Dict]:
        """Analyze model to identify layers and their properties."""
        layer_info = {}
        
        # Identify encoder layers
        for i, layer in enumerate(self.model.encoder_layers):
            name = f"encoder_{i}"
            layer_info[name] = {
                "module": layer,
                "type": "encoder",
                "depth": i,
                "receptive_field": self._estimate_rf_size(i, "encoder")
            }
            
        # Identify decoder layers
        for i, layer in enumerate(self.model.decoder_layers):
            name = f"decoder_{i}"
            layer_info[name] = {
                "module": layer,
                "type": "decoder", 
                "depth": i,
                "receptive_field": self._estimate_rf_size(i, "decoder")
            }
            
        return layer_info
    
    def _estimate_rf_size(self, layer_idx: int, layer_type: str) -> int:
        """Estimate receptive field size for a layer.
        
        Simple estimation based on layer depth and kernel sizes.
        """
        # Base RF from input convolution
        rf = 7  # Assuming 7x7 input conv
        
        # Add contribution from each layer
        if layer_type == "encoder":
            # Each encoder layer adds based on fGRU kernel size
            # Rough estimate: each 3x3 kernel adds 2 pixels per side
            rf += layer_idx * 2 * 3  # 3 timesteps
        else:
            # Decoder layers have varying impact
            rf += (len(self.model.encoder_layers) + layer_idx) * 2 * 3
            
        return rf
    
    def register_hooks(self, 
                      target_layers: Optional[List[str]] = None,
                      extract_hidden_states: bool = True,
                      extract_gates: bool = False) -> None:
        """Register forward hooks on target layers.
        
        Args:
            target_layers: List of layer names to extract from. If None, extracts all.
            extract_hidden_states: Whether to extract hidden states
            extract_gates: Whether to extract gate values
        """
        # Clear existing hooks
        self.clear_hooks()
        
        if target_layers is None:
            target_layers = list(self.layer_info.keys())
            
        for layer_name in target_layers:
            if layer_name not in self.layer_info:
                warnings.warn(f"Layer {layer_name} not found in model")
                continue
                
            layer = self.layer_info[layer_name]["module"]
            
            # Hook for main layer output
            if extract_hidden_states:
                hook = layer.register_forward_hook(
                    self._create_hook(layer_name, "hidden_state")
                )
                self.hooks.append(hook)
                
            # Hook for fGRU gates if requested
            if extract_gates and hasattr(layer, 'fgru'):
                # Would need to modify fGRU to expose gates
                pass
                
    def _create_hook(self, layer_name: str, response_type: str):
        """Create a forward hook function."""
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                activation = output[0]  # Assume first element is activation
            else:
                activation = output
                
            # Store response
            key = f"{layer_name}_{response_type}"
            if key not in self.responses:
                self.responses[key] = []
                
            self.responses[key].append(activation.detach().cpu())
            
        return hook_fn
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.responses = {}
        
    @torch.no_grad()
    def extract_responses(self, 
                         stimuli: torch.Tensor,
                         timesteps: Optional[List[int]] = None) -> Dict[str, List[LayerResponse]]:
        """Extract responses from model for given stimuli.
        
        Args:
            stimuli: Input stimuli tensor [batch, channels, height, width]
            timesteps: Which timesteps to extract. If None, extracts all.
            
        Returns:
            Dictionary mapping layer names to list of responses per timestep
        """
        self.model.eval()
        self.responses = {}  # Clear previous responses
        
        # Move stimuli to model device
        device = next(self.model.parameters()).device
        stimuli = stimuli.to(device)
        
        # Forward pass
        _ = self.model(stimuli)
        
        # Organize responses by layer and timestep
        organized_responses = {}
        
        for key, activations in self.responses.items():
            layer_name = key.split('_')[0]
            
            if layer_name not in organized_responses:
                organized_responses[layer_name] = []
                
            # Create LayerResponse objects for each timestep
            for t, activation in enumerate(activations):
                if timesteps is None or t in timesteps:
                    response = LayerResponse(
                        layer_name=layer_name,
                        timestep=t,
                        activation=activation,
                        receptive_field_size=self.layer_info[layer_name]["receptive_field"],
                        layer_type=self.layer_info[layer_name]["type"]
                    )
                    organized_responses[layer_name].append(response)
                    
        return organized_responses
    
    def get_population_response(self,
                               responses: Dict[str, List[LayerResponse]],
                               layer_name: str,
                               spatial_pool: str = "center",
                               pool_size: Optional[int] = None) -> np.ndarray:
        """Extract population response from a specific layer.
        
        Args:
            responses: Dictionary of responses from extract_responses
            layer_name: Which layer to extract from
            spatial_pool: How to pool spatially - "center", "mean", "max"
            pool_size: Size of pooling region (for "center")
            
        Returns:
            Population response array [batch, timesteps, channels]
        """
        if layer_name not in responses:
            raise ValueError(f"Layer {layer_name} not found in responses")
            
        layer_responses = responses[layer_name]
        
        # Organize by timestep
        timestep_responses = {}
        for resp in layer_responses:
            if resp.timestep not in timestep_responses:
                timestep_responses[resp.timestep] = []
            timestep_responses[resp.timestep].append(resp.activation)
            
        # Process each timestep
        pooled_responses = []
        
        for t in sorted(timestep_responses.keys()):
            activations = torch.cat(timestep_responses[t], dim=0)
            
            if spatial_pool == "center":
                # Extract center region
                if pool_size is None:
                    pool_size = min(activations.shape[-2:]) // 4
                    
                center_h = activations.shape[-2] // 2
                center_w = activations.shape[-1] // 2
                half_size = pool_size // 2
                
                pooled = activations[
                    :, :,
                    center_h - half_size:center_h + half_size,
                    center_w - half_size:center_w + half_size
                ].mean(dim=(-2, -1))
                
            elif spatial_pool == "mean":
                pooled = activations.mean(dim=(-2, -1))
                
            elif spatial_pool == "max":
                pooled = activations.amax(dim=(-2, -1))
                
            else:
                raise ValueError(f"Unknown pooling method: {spatial_pool}")
                
            pooled_responses.append(pooled.numpy())
            
        # Stack timesteps
        return np.stack(pooled_responses, axis=1)  # [batch, timesteps, channels]
    
    def extract_temporal_dynamics(self,
                                 responses: Dict[str, List[LayerResponse]],
                                 layer_name: str,
                                 channel_idx: Optional[int] = None) -> np.ndarray:
        """Extract temporal dynamics of responses.
        
        Args:
            responses: Dictionary of responses
            layer_name: Which layer to analyze
            channel_idx: Specific channel to extract (if None, averages all)
            
        Returns:
            Temporal response array [batch, timesteps]
        """
        population_response = self.get_population_response(
            responses, layer_name, spatial_pool="mean"
        )
        
        if channel_idx is not None:
            temporal_response = population_response[:, :, channel_idx]
        else:
            temporal_response = population_response.mean(axis=-1)
            
        return temporal_response
    
    def compute_receptive_field_size(self,
                                    layer_name: str,
                                    method: str = "gradient") -> int:
        """Compute empirical receptive field size.
        
        Args:
            layer_name: Layer to analyze
            method: Method to use - "gradient" or "occlusion"
            
        Returns:
            Estimated receptive field size in pixels
        """
        if method == "gradient":
            return self._rf_size_gradient(layer_name)
        elif method == "occlusion":
            return self._rf_size_occlusion(layer_name)
        else:
            # Return pre-computed estimate
            return self.layer_info[layer_name]["receptive_field"]
    
    def _rf_size_gradient(self, layer_name: str) -> int:
        """Estimate RF size using gradient method."""
        # Create test stimulus
        size = 256
        test_input = torch.randn(1, 3, size, size, requires_grad=True)
        device = next(self.model.parameters()).device
        test_input = test_input.to(device)
        
        # Register hook to capture target layer
        activation = None
        def capture_hook(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
                
        hook = self.layer_info[layer_name]["module"].register_forward_hook(capture_hook)
        
        # Forward pass
        _ = self.model(test_input)
        
        # Get gradient at center location
        center_h = activation.shape[-2] // 2
        center_w = activation.shape[-1] // 2
        center_response = activation[0, :, center_h, center_w].mean()
        
        # Backward pass
        center_response.backward()
        
        # Analyze gradient magnitude
        grad = test_input.grad[0].abs().mean(dim=0).cpu().numpy()
        
        # Find extent of significant gradients
        threshold = 0.1 * grad.max()
        significant = grad > threshold
        
        # Find bounding box
        rows = np.any(significant, axis=1)
        cols = np.any(significant, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            rf_size = max(rmax - rmin, cmax - cmin)
        else:
            rf_size = self.layer_info[layer_name]["receptive_field"]
            
        hook.remove()
        return int(rf_size)
    
    def _rf_size_occlusion(self, layer_name: str) -> int:
        """Estimate RF size using occlusion method."""
        # Simplified implementation
        # In practice, would systematically occlude regions
        return self.layer_info[layer_name]["receptive_field"]
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.clear_hooks()


def extract_layer_responses(model: nn.Module,
                          stimuli: torch.Tensor,
                          target_layers: List[str],
                          timestep: int = -1) -> Dict[str, torch.Tensor]:
    """Convenience function to extract responses from specific layers.
    
    Args:
        model: GammaNet model
        stimuli: Input stimuli
        target_layers: List of layer names
        timestep: Which timestep to extract (-1 for last)
        
    Returns:
        Dictionary mapping layer names to responses
    """
    extractor = ResponseExtractor(model)
    extractor.register_hooks(target_layers)
    
    responses = extractor.extract_responses(stimuli)
    
    # Extract specific timestep
    extracted = {}
    for layer_name, layer_responses in responses.items():
        if timestep == -1:
            response = layer_responses[-1].activation
        else:
            response = layer_responses[timestep].activation
        extracted[layer_name] = response
        
    extractor.clear_hooks()
    return extracted


def get_v1_like_layers(model: nn.Module) -> List[str]:
    """Identify V1-like layers in the model.
    
    Returns layers that are likely to have V1-like properties based on
    their position in the hierarchy.
    """
    # Typically layers 2-3 of encoder are most V1-like
    return ["encoder_1", "encoder_2", "encoder_3"]