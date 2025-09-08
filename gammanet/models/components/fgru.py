"""fGRU (flexible Gated Recurrent Unit) implementation.

This module implements both horizontal (h-fGRU) and top-down (td-fGRU) variants.
Based on the circuit operations from the original TensorFlow implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
try:
    from typing import Literal
except ImportError:
    # Python 3.7 compatibility
    from typing_extensions import Literal
from .attention import SEBlock, GALABlock


class DynamicParameters(nn.Module):
    """Compute circuit parameters as dynamic functions of neural states.
    
    Instead of static parameters, this module computes alpha, mu, kappa, and omega
    as functions of the current neural activations, making them input-dependent.
    """
    
    def __init__(self, channels: int, use_softplus: bool = True):
        """Initialize dynamic parameter computation networks.
        
        Args:
            channels: Number of channels in the fGRU
            use_softplus: Whether to use softplus activation (True) or sigmoid (False)
        """
        super().__init__()
        self.channels = channels
        self.use_softplus = use_softplus
        
        # Network for computing inhibition parameters (alpha, mu)
        # Takes concatenated [h, c1, ff_input] as input
        self.inhibition_net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1)  # Output raw alpha and mu
        )
        
        # Network for computing excitation parameters (kappa, omega)
        # Takes concatenated [h1, c2] as input
        self.excitation_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1)  # Output raw kappa and omega
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize networks to output reasonable starting values."""
        # Initialize to output small positive values after activation
        for module in [self.inhibition_net, self.excitation_net]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def compute_inhibition_params(self, h: torch.Tensor, c1: torch.Tensor, 
                                 ff_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic inhibition parameters.
        
        Args:
            h: Hidden state [B, C, H, W]
            c1: Inhibitory circuit output [B, C, H, W]
            ff_input: Feedforward input [B, C, H, W]
            
        Returns:
            alpha: Dynamic divisive inhibition parameter [B, C, H, W]
            mu: Dynamic subtractive inhibition parameter [B, C, H, W]
        """
        # Concatenate inputs
        combined = torch.cat([h, c1, ff_input], dim=1)
        
        # Compute parameters
        params = self.inhibition_net(combined)
        alpha, mu = torch.chunk(params, 2, dim=1)
        
        # Apply activation for positive values
        if self.use_softplus:
            alpha = F.softplus(alpha)
            mu = F.softplus(mu)
        else:
            alpha = torch.sigmoid(alpha)
            mu = torch.sigmoid(mu)
        
        return alpha, mu
    
    def compute_excitation_params(self, h1: torch.Tensor, c2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic excitation parameters.
        
        Args:
            h1: Intermediate hidden state [B, C, H, W]
            c2: Excitatory circuit output [B, C, H, W]
            
        Returns:
            kappa: Dynamic additive excitation parameter [B, C, H, W]
            omega: Dynamic multiplicative excitation parameter [B, C, H, W]
        """
        # Concatenate inputs
        combined = torch.cat([h1, c2], dim=1)
        
        # Compute parameters
        params = self.excitation_net(combined)
        kappa, omega = torch.chunk(params, 2, dim=1)
        
        # Apply activation for positive values
        if self.use_softplus:
            kappa = F.softplus(kappa)
            omega = F.softplus(omega)
        else:
            kappa = torch.sigmoid(kappa)
            omega = torch.sigmoid(omega)
        
        return kappa, omega


class SymmetricConv2d(torch.autograd.Function):
    """Custom autograd function for symmetric weight constraints."""
    
    @staticmethod
    def forward(ctx, input, weight, bias, symmetry_type):
        if symmetry_type == 'channel':
            # Make channels symmetric
            weight_symmetric = 0.5 * (weight + weight.transpose(0, 1))
        elif symmetry_type == 'spatial':
            # Make spatial dimensions symmetric
            weight_symmetric = 0.5 * (weight + weight.transpose(-2, -1))
        elif symmetry_type == 'spatial_channel':
            # Both spatial and channel symmetry
            weight_symmetric = 0.25 * (weight + weight.transpose(0, 1) + 
                                      weight.transpose(-2, -1) + 
                                      weight.transpose(0, 1).transpose(-2, -1))
        else:
            weight_symmetric = weight
            
        ctx.save_for_backward(input, weight_symmetric, bias)
        ctx.symmetry_type = symmetry_type
        return F.conv2d(input, weight_symmetric, bias, padding='same')
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_symmetric, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Calculate proper padding for 'same' behavior
            padding = weight_symmetric.shape[2] // 2
            # Ensure grad_output and weight have same dtype
            grad_input = F.conv_transpose2d(
                grad_output, 
                weight_symmetric.to(grad_output.dtype), 
                padding=padding
            )
            
        if ctx.needs_input_grad[1]:
            # Calculate proper padding for 'same' behavior
            padding = weight_symmetric.shape[2] // 2
            # Ensure input and grad_output have same dtype
            grad_weight = F.conv2d(
                input.transpose(0, 1).to(grad_output.dtype), 
                grad_output.transpose(0, 1), 
                padding=padding
            ).transpose(0, 1)
            
            # Apply symmetry constraint to gradients
            if ctx.symmetry_type == 'channel':
                grad_weight = 0.5 * (grad_weight + grad_weight.transpose(0, 1))
            elif ctx.symmetry_type == 'spatial':
                grad_weight = 0.5 * (grad_weight + grad_weight.transpose(-2, -1))
            elif ctx.symmetry_type == 'spatial_channel':
                grad_weight = 0.25 * (grad_weight + grad_weight.transpose(0, 1) + 
                                    grad_weight.transpose(-2, -1) + 
                                    grad_weight.transpose(0, 1).transpose(-2, -1))
                
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))
            
        return grad_input, grad_weight, grad_bias, None


class fGRU(nn.Module):
    """Flexible Gated Recurrent Unit for horizontal and top-down connections.
    
    Implements the core circuit operations:
    1. Circuit input: gain control on horizontal connections
    2. Input integration: subtractive inhibition
    3. Circuit output: mixing gate for output
    4. Output integration: additive/multiplicative excitation
    """
    
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: Tuple[int, int] = (7, 7),
        use_attention: Optional[Literal['se', 'gala']] = None,
        attention_layers: int = 1,
        symmetric_weights: Optional[Literal['channel', 'spatial', 'spatial_channel']] = 'channel',
        multiplicative_excitation: bool = True,
        force_alpha_divisive: bool = True,
        force_omega_nonnegative: bool = True,
        normalization_type: Optional[str] = 'layer',
        normalize_c1_c2: bool = True,
        normalize_circuit_outputs: bool = True,  # NEW: control circuit output normalization
        use_dynamic_parameters: bool = False,  # NEW: use dynamic parameter computation
        dynamic_param_activation: str = 'softplus',  # NEW: activation for dynamic params
        partial_padding: bool = False,
        use_symmetric_conv: bool = True,
        # v2 parameters (ignored in v1 for backward compatibility)
        use_separate_ei_states: bool = False,  # Ignored in v1
        gate_norm_position: str = 'pre',  # Ignored in v1
        **kwargs  # Catch any other new parameters
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.attention_layers = attention_layers
        self.symmetric_weights = symmetric_weights
        self.multiplicative_excitation = multiplicative_excitation
        self.force_alpha_divisive = force_alpha_divisive
        self.force_omega_nonnegative = force_omega_nonnegative
        self.normalization_type = normalization_type
        self.normalize_c1_c2 = normalize_c1_c2
        self.normalize_circuit_outputs = normalize_circuit_outputs
        self.use_dynamic_parameters = use_dynamic_parameters
        self.dynamic_param_activation = dynamic_param_activation
        self.partial_padding = partial_padding
        self.use_symmetric_conv = use_symmetric_conv
        
        # Horizontal kernels (excitatory and inhibitory)
        self.W_exc = nn.Parameter(torch.empty(hidden_channels, hidden_channels, *kernel_size))
        self.W_inh = nn.Parameter(torch.empty(hidden_channels, hidden_channels, *kernel_size))
        
        # Gate parameters
        if use_attention:
            if use_attention == 'se':
                self.attention = SEBlock(hidden_channels)
            elif use_attention == 'gala':
                self.attention = GALABlock(hidden_channels, attention_layers)
        else:
            # W_gain takes concatenated input [h, ff_input]
            self.W_gain = nn.Parameter(torch.empty(hidden_channels, 2*hidden_channels, 1, 1))
            
        # W_mix takes concatenated input [h1, h2]
        self.W_mix = nn.Parameter(torch.empty(hidden_channels, 2*hidden_channels, 1, 1))
        
        # Biases
        self.b_gain = nn.Parameter(torch.zeros(hidden_channels))
        self.b_mix = nn.Parameter(torch.zeros(hidden_channels))
        
        # Integration parameters - static or dynamic
        if use_dynamic_parameters:
            # Use dynamic parameter computation
            use_softplus = (dynamic_param_activation == 'softplus')
            self.dynamic_params = DynamicParameters(hidden_channels, use_softplus=use_softplus)
            # No static parameters needed
            self.alpha = None
            self.mu = None
            self.kappa = None
            self.omega = None
        else:
            # Use static parameters (original behavior)
            self.alpha = nn.Parameter(torch.zeros(hidden_channels))  # Divisive inhibition
            self.mu = nn.Parameter(torch.zeros(hidden_channels))     # Subtractive inhibition
            
            if multiplicative_excitation:
                self.kappa = nn.Parameter(torch.ones(hidden_channels) * 0.1)   # Additive excitation
                self.omega = nn.Parameter(torch.ones(hidden_channels) * 0.1)   # Multiplicative excitation
            else:
                self.kappa = None
                self.omega = None
        
        # Normalization layers
        if normalization_type == 'layer':
            # For gates: use normalization without learnable parameters (like TF layer_norm with center=False, scale=False)
            self.norm_g1 = nn.GroupNorm(1, hidden_channels, affine=False)  # No learnable params
            self.norm_g2 = nn.GroupNorm(1, hidden_channels, affine=False)  # No learnable params
            # For circuit outputs: use standard normalization if enabled
            if normalize_c1_c2 and normalize_circuit_outputs:
                self.norm_c1 = nn.GroupNorm(1, hidden_channels)
                self.norm_c2 = nn.GroupNorm(1, hidden_channels)
            else:
                self.norm_c1 = self.norm_c2 = nn.Identity()
        elif normalization_type == 'instance':
            # For gates: use normalization without learnable parameters
            self.norm_g1 = nn.InstanceNorm2d(hidden_channels, affine=False)
            self.norm_g2 = nn.InstanceNorm2d(hidden_channels, affine=False)
            # For circuit outputs: use standard normalization if enabled
            if normalize_c1_c2 and normalize_circuit_outputs:
                self.norm_c1 = nn.InstanceNorm2d(hidden_channels, affine=True)
                self.norm_c2 = nn.InstanceNorm2d(hidden_channels, affine=True)
            else:
                self.norm_c1 = self.norm_c2 = nn.Identity()
        else:
            self.norm_g1 = self.norm_g2 = self.norm_c1 = self.norm_c2 = nn.Identity()
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights following the original implementation."""
        # Orthogonal initialization for recurrent weights
        nn.init.orthogonal_(self.W_exc)
        nn.init.orthogonal_(self.W_inh)
        
        # Xavier/Glorot for gates
        if hasattr(self, 'W_gain'):
            nn.init.xavier_uniform_(self.W_gain)
        nn.init.xavier_uniform_(self.W_mix)
        
        # Zero initialization for biases
        nn.init.zeros_(self.b_gain)
        nn.init.zeros_(self.b_mix)
        
        # Integration parameters following original TF implementation
        if not self.use_dynamic_parameters:
            if self.alpha is not None:
                nn.init.ones_(self.alpha)  # Changed from zeros to ones
            if self.mu is not None:
                nn.init.zeros_(self.mu)    # Keep at zero
            
            if self.multiplicative_excitation and self.kappa is not None:
                nn.init.zeros_(self.kappa)  # Changed from 0.1 to 0
                nn.init.ones_(self.omega)   # Changed from 0.1 to 1
    
    def _symmetric_conv2d(self, input: torch.Tensor, weight: torch.Tensor, 
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply convolution with optional symmetric weight constraints."""
        if self.use_symmetric_conv and self.symmetric_weights:
            return SymmetricConv2d.apply(input, weight, bias, self.symmetric_weights)
        else:
            return F.conv2d(input, weight, bias, padding='same')
    
    def compute_input_gate(self, h: torch.Tensor, ff_input: torch.Tensor, norm="pre") -> torch.Tensor:
        """Compute input gate (g1) from hidden state and feedforward input."""
        # Prepare gate input by concatenating hidden state and feedforward input
        # This matches the original: gate_activity = tf.concat([h2, ff_drive], -1)
        gate_input = torch.cat([h, ff_input], dim=1)
            
        # Apply attention or standard gating
        if self.use_attention:
            g1_intermediate = self.attention(gate_input)
        else:
            g1_intermediate = F.conv2d(gate_input, self.W_gain, padding='same')
            
        # Normalize and apply sigmoid
        g1_intermediate = self.norm_g1(g1_intermediate)
        g1 = torch.sigmoid(g1_intermediate + self.b_gain.view(1, -1, 1, 1))
        
        return g1
    
    def circuit_input(self, h: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
        """Apply circuit input: gain-modulated horizontal connections."""
        # Compute inhibitory horizontal connections
        # Following original implementation: use h directly, not h*g1
        c1 = self._symmetric_conv2d(h, self.W_inh)
        c1 = self.norm_c1(c1)
        
        return c1
    
    def input_integration(self, ff_input: torch.Tensor, c1: torch.Tensor, h: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
        """Input integration with subtractive inhibition and gated mixing.
        
        Following the original implementation which uses gated mixing between
        previous state and new inhibited state.
        """
        # Get integration parameters - either static or dynamic
        if self.use_dynamic_parameters:
            # Compute dynamic parameters based on current states
            alpha, mu = self.dynamic_params.compute_inhibition_params(h, c1, ff_input)
        else:
            # Use static parameters
            alpha = self.alpha.view(1, -1, 1, 1)
            mu = self.mu.view(1, -1, 1, 1)
            
            if self.force_alpha_divisive:
                alpha = torch.sigmoid(alpha)
            
        # Subtractive inhibition
        inhibition = F.relu((alpha * h + mu) * c1)
        inhibited = F.relu(ff_input - inhibition)
        
        # Gated mixing between previous hidden state and inhibited state
        # This matches the original: (1 - g1) * h + g1 * inhibited
        h1 = (1 - g1) * h + g1 * inhibited
        
        return h1
    
    def compute_output_gate(self, h1: torch.Tensor, h2: torch.Tensor, norm="pre") -> torch.Tensor:
        """Compute output/mix gate (g2)."""
        # Concatenate h1 and h2 for gate computation
        # This matches original: gate_activity = tf.concat([h1, h2], -1)
        # Norm stabalizes gates. But empirical for pre vs. post
        gate_input = torch.cat([h1, h2], dim=1)
        if norm == "pre":
            g2_intermediate = F.conv2d(self.norm_g2(gate_input), self.W_mix, padding='same')
        elif norm == "post":
            g2_intermediate = F.conv2d(gate_input, self.W_mix, padding='same')
            g2_intermediate = self.norm_g2(g2_intermediate)
        g2 = torch.sigmoid(g2_intermediate + self.b_mix.view(1, -1, 1, 1))
        return g2
    
    def circuit_output(self, h1: torch.Tensor) -> torch.Tensor:
        """Apply circuit output: excitatory horizontal connections."""
        c2 = self._symmetric_conv2d(h1, self.W_exc)
        c2 = self.norm_c2(c2)
        
        return c2
    
    def output_integration(self, h1: torch.Tensor, c2: torch.Tensor, g2: torch.Tensor, 
                          h_prev: torch.Tensor) -> torch.Tensor:
        """Output integration with additive/multiplicative excitation."""
        if self.multiplicative_excitation:
            # Get integration parameters - either static or dynamic
            if self.use_dynamic_parameters:
                # Compute dynamic parameters based on current states
                kappa, omega = self.dynamic_params.compute_excitation_params(h1, c2)
            else:
                # Use static parameters
                kappa = self.kappa.view(1, -1, 1, 1)
                omega = self.omega.view(1, -1, 1, 1)
                
                if self.force_omega_nonnegative:
                    omega = F.relu(omega)
                
            # Following original: exc = (omega * h1 + kappa) * c2
            exc = (omega * h1 + kappa) * c2
            exc = F.relu(exc)
        else:
            # Simple additive
            exc = F.relu(h1 + c2)
            
        # Mix with previous hidden state
        h_next = (1 - g2) * h_prev + g2 * exc
        
        return h_next
    
    def forward(self, ff_input: torch.Tensor, h_prev: torch.Tensor, 
                td_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of fGRU.
        
        Args:
            ff_input: Feedforward input [B, C, H, W]
            h_prev: Previous hidden state [B, C, H, W]
            td_input: Optional top-down input for gating
            
        Returns:
            h_next: Next hidden state [B, C, H, W]
            h1: Intermediate state (error signal) [B, C, H, W]
        """
        # 1. Compute input gate using h_prev and ff_input
        g1 = self.compute_input_gate(h_prev, ff_input)
        
        # 2. Circuit input (inhibitory horizontal connections)
        c1 = self.circuit_input(h_prev, g1)
        
        # 3. Input integration
        h1 = self.input_integration(ff_input, c1, h_prev, g1)
        
        # 4. Compute output gate
        g2 = self.compute_output_gate(h1, h_prev)
        
        # 5. Circuit output (excitatory horizontal connections)
        c2 = self.circuit_output(h1)
        
        # 6. Output integration
        h_next = self.output_integration(h1, c2, g2, h_prev)
        
        return h_next, h1
