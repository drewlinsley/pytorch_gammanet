"""fGRU v2 with separate E/I states and improved normalization.

This module implements both horizontal (h-fGRU) and top-down (td-fGRU) variants
with biologically-inspired separate excitatory and inhibitory populations.
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
from .normalization import LayerNorm2d


class DynamicParametersV2(nn.Module):
    """Compute circuit parameters as dynamic functions of E/I neural states.

    This version correctly uses E/I states for inhibition and post-inhibition
    signal for excitation parameters.
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
        # Takes concatenated [h_exc, h_inh, ff_input] as input (3*channels)
        self.inhibition_net = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=1),
            # No normalization in checkpoint version!
        )

        # Network for computing excitation parameters (kappa, omega)
        # Takes concatenated [inhibited_signal, h_exc] as input (2*channels)
        self.excitation_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            # No normalization in checkpoint version!
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

    def compute_inhibition_params(self, h_exc: torch.Tensor, h_inh: torch.Tensor,
                                 ff_input: torch.Tensor, use_separate_ei_states=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic inhibition parameters from E/I states.

        Args:
            h_exc: Excitatory hidden state [B, C, H, W]
            h_inh: Inhibitory hidden state [B, C, H, W]
            ff_input: Feedforward input [B, C, H, W]

        Returns:
            alpha: Dynamic divisive inhibition parameter [B, C, H, W]
            mu: Dynamic subtractive inhibition parameter [B, C, H, W]
        """
        # Concatenate E, I states and FF input
        combined = torch.cat([h_exc, h_inh, ff_input], dim=1)

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

    def compute_excitation_params(self, inhibited_signal: torch.Tensor,
                                 h_exc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic excitation parameters.

        Args:
            inhibited_signal: Post-inhibition feedforward signal [B, C, H, W]
            h_exc: Excitatory hidden state [B, C, H, W]

        Returns:
            kappa: Dynamic additive excitation parameter [B, C, H, W]
            omega: Dynamic multiplicative excitation parameter [B, C, H, W]
        """
        # Concatenate inhibited signal and excitatory state
        combined = torch.cat([inhibited_signal, h_exc], dim=1)

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


class fGRUv2(nn.Module):
    """Flexible Gated Recurrent Unit v2 with E/I states.

    Implements the core circuit operations with separate excitatory and
    inhibitory populations:
    1. Circuit input: gain control from excitatory state
    2. Input integration: inhibition from both E/I states
    3. Circuit output: excitatory horizontal connections
    4. Output integration: excitation modulation
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
        normalize_circuit_outputs: bool = True,
        use_dynamic_parameters: bool = False,
        dynamic_param_activation: str = 'softplus',
        use_separate_ei_states: bool = True,  # NEW
        gate_norm_position: str = 'pre',  # NEW: 'pre' or 'post'
        partial_padding: bool = False,
        use_symmetric_conv: bool = True,
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
        self.use_separate_ei_states = use_separate_ei_states
        self.gate_norm_position = gate_norm_position
        self.partial_padding = partial_padding
        self.use_symmetric_conv = use_symmetric_conv

        # Horizontal kernels (excitatory and inhibitory)
        self.W_exc = nn.Parameter(torch.empty(hidden_channels, hidden_channels, *kernel_size))
        self.W_inh = nn.Parameter(torch.empty(hidden_channels, hidden_channels, *kernel_size))

        # Gate parameters - adjusted for concatenated inputs
        if use_attention:
            # When using attention, we still need a projection from 3C to C before attention
            self.W_gain = nn.Parameter(torch.empty(hidden_channels, 3*hidden_channels, 1, 1))
            if use_attention == 'se':
                self.attention = SEBlock(hidden_channels)
            elif use_attention == 'gala':
                self.attention = GALABlock(hidden_channels, attention_layers)
        else:
            # W_gain takes concatenated input [h_exc, h_inh, ff_input] -> 3*channels input
            self.W_gain = nn.Parameter(torch.empty(hidden_channels, 3*hidden_channels, 1, 1))

        # W_mix takes concatenated input [h_exc_new, h_exc_prev] -> 2*channels input
        self.W_mix = nn.Parameter(torch.empty(hidden_channels, 3*hidden_channels, 1, 1))

        # Biases
        self.b_gain = nn.Parameter(torch.zeros(hidden_channels))
        self.b_mix = nn.Parameter(torch.zeros(hidden_channels))

        # Integration parameters - static or dynamic
        if use_dynamic_parameters:
            # Use dynamic parameter computation
            use_softplus = (dynamic_param_activation == 'softplus')
            self.dynamic_params = DynamicParametersV2(hidden_channels, use_softplus=use_softplus)
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

        # Normalization layers for gates (with pre/post options)
        if normalization_type == 'layer':
            # For pre-normalization (normalize concatenated inputs)
            self.norm_g1_pre = LayerNorm2d(3*hidden_channels, affine=False)  # 3C for h_exc, h_inh, ff_input
            self.norm_g2_pre = LayerNorm2d(3*hidden_channels, affine=False)  # 2C for h_inh_new, h_exc_prev
            # For post-normalization (normalize C outputs)
            self.norm_g1_post = LayerNorm2d(hidden_channels, affine=False)
            self.norm_g2_post = LayerNorm2d(hidden_channels, affine=False)

            # For circuit outputs: use standard normalization if enabled
            if normalize_c1_c2 and normalize_circuit_outputs:
                self.norm_c1 = LayerNorm2d(hidden_channels)
                self.norm_c2 = LayerNorm2d(hidden_channels)
            else:
                self.norm_c1 = self.norm_c2 = nn.Identity()

        elif normalization_type == 'instance':
            # For pre-normalization
            self.norm_g1_pre = nn.InstanceNorm2d(3*hidden_channels, affine=False)  # 3C for h_exc, h_inh, ff_input
            self.norm_g2_pre = nn.InstanceNorm2d(3*hidden_channels, affine=False)  # 2C for h_inh_new, h_exc_prev
            # For post-normalization
            self.norm_g1_post = nn.InstanceNorm2d(hidden_channels, affine=False)
            self.norm_g2_post = nn.InstanceNorm2d(hidden_channels, affine=False)

            # For circuit outputs
            if normalize_c1_c2 and normalize_circuit_outputs:
                self.norm_c1 = nn.InstanceNorm2d(hidden_channels, affine=True)
                self.norm_c2 = nn.InstanceNorm2d(hidden_channels, affine=True)
            else:
                self.norm_c1 = self.norm_c2 = nn.Identity()

        elif normalization_type == 'group':
            num_groups = min(8, hidden_channels)
            # For pre-normalization
            self.norm_g1_pre = nn.GroupNorm(min(num_groups*3, 3*hidden_channels), 3*hidden_channels, affine=False)  # 3C for h_exc, h_inh, ff_input
            self.norm_g2_pre = nn.GroupNorm(min(num_groups*3, 3*hidden_channels), 3*hidden_channels, affine=False)  # 2C for h_inh_new, h_exc_prev
            # For post-normalization
            self.norm_g1_post = nn.GroupNorm(num_groups, hidden_channels, affine=False)
            self.norm_g2_post = nn.GroupNorm(num_groups, hidden_channels, affine=False)

            # For circuit outputs
            if normalize_c1_c2 and normalize_circuit_outputs:
                self.norm_c1 = nn.GroupNorm(num_groups, hidden_channels)
                self.norm_c2 = nn.GroupNorm(num_groups, hidden_channels)
            else:
                self.norm_c1 = self.norm_c2 = nn.Identity()
        else:
            # No normalization
            self.norm_g1_pre = self.norm_g2_pre = nn.Identity()
            self.norm_g1_post = self.norm_g2_post = nn.Identity()
            self.norm_c1 = self.norm_c2 = nn.Identity()

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

    def compute_input_gate(self, inh_input: torch.Tensor, h_inh_next: torch.Tensor, h_inh_prev: torch.Tensor) -> torch.Tensor:
        """Compute input gate (g1) from excitatory state, inhibitory state, and feedforward input.

        Uses configurable pre/post normalization.
        """
        gate_input = torch.cat([inh_input, h_inh_next, h_inh_prev], dim=1)  # [B, 3C, H, W]

        # Apply attention or standard gating
        if self.use_attention:
            raise RuntimeError("We need to disable attention for now.")
            # Attention modules expect single channel input, need to project first
            gate_proj = F.conv2d(gate_input, self.W_gain, padding='same')
            g1_intermediate = self.attention(gate_proj)
        else:
            if self.gate_norm_position == 'pre':
                # Normalize concatenated input, then project
                gate_input_normed = self.norm_g1_pre(gate_input)
                g1_intermediate = F.conv2d(gate_input_normed, self.W_gain, padding='same')
            else:  # 'post'
                # Project first, then normalize
                g1_intermediate = F.conv2d(gate_input, self.W_gain, padding='same')
                g1_intermediate = self.norm_g1_post(g1_intermediate)

        g1 = torch.sigmoid(g1_intermediate + self.b_gain.view(1, -1, 1, 1))
        return g1

    def compute_output_gate(self, exc_input: torch.Tensor, h_exc_next: torch.Tensor, h_exc_prev: torch.Tensor) -> torch.Tensor:
        """Compute output/mix gate (g2) from excitatory states.

        Uses configurable pre/post normalization.
        exc_input, h_exc_next, h_exc_prev
        """
        gate_input = torch.cat([h_exc_prev, h_exc_next, exc_input], dim=1)  # [B, 2C, H, W]

        if self.gate_norm_position == 'pre':
            # Normalize concatenated input, then project
            gate_input_normed = self.norm_g2_pre(gate_input)
            g2_intermediate = F.conv2d(gate_input_normed, self.W_mix, padding='same')
        else:  # 'post'
            # Project first, then normalize
            g2_intermediate = F.conv2d(gate_input, self.W_mix, padding='same')
            g2_intermediate = self.norm_g2_post(g2_intermediate)

        g2 = torch.sigmoid(g2_intermediate + self.b_mix.view(1, -1, 1, 1))
        return g2

    def circuit_input(self, h_inh: torch.Tensor) -> torch.Tensor:
        """Apply circuit input: inhibitory horizontal connections."""
        c1 = self.norm_c1(h_inh)
        c1 = self._symmetric_conv2d(c1, self.W_inh)
        c1 = F.relu(c1)
        return c1

    def circuit_output(self, h_exc: torch.Tensor) -> torch.Tensor:
        """Apply circuit output: excitatory horizontal connections."""
        c2 = self.norm_c2(h_exc)
        c2 = self._symmetric_conv2d(c2, self.W_exc)
        c2 = F.relu(c2)
        return c2

    def apply_inhibition(self, ff_input: torch.Tensor, h_inh: torch.Tensor, inh_horizontals: torch.Tensor) -> torch.Tensor:
        """Apply inhibition using E/I states.

        Args:
            ff_input: Feedforward input [B, C, H, W]
            h_exc: Excitatory state [B, C, H, W]
            h_inh: Inhibitory state [B, C, H, W]
            c1: Inhibitory circuit output [B, C, H, W]

        Returns:
            inhibited_signal: Post-inhibition signal [B, C, H, W]
        """
        # Get integration parameters - either static or dynamic
        if self.use_dynamic_parameters:
            # Compute dynamic parameters based on E/I states
            alpha, mu = self.dynamic_params.compute_inhibition_params(h_inh, inh_horizontals, ff_input)
        else:
            # Use static parameters
            alpha = self.alpha.view(1, -1, 1, 1)
            mu = self.mu.view(1, -1, 1, 1)

            if self.force_alpha_divisive:
                alpha = torch.sigmoid(alpha)

        # Apply inhibition using both E and I states
        inhibition = (alpha * h_inh + mu) * inh_horizontals
        inhibited_signal = F.relu(ff_input - inhibition)  # Rectify to ensure these are rates
        return inhibited_signal

    def apply_excitation(self, exc_input: torch.Tensor, exc_horizontals: torch.Tensor) -> torch.Tensor:
        """Apply excitation and output gating.

                h_exc_new = self.apply_excitation(exc_input, exc_horizontals)
        Args:
            inhibited_signal: Post-inhibition signal [B, C, H, W]
            h_exc: Current excitatory state [B, C, H, W]
            c2: Excitatory circuit output [B, C, H, W]
            g2: Output gate [B, C, H, W]
            h_exc_prev: Previous excitatory state [B, C, H, W]

        Returns:
            h_exc_next: Next excitatory state [B, C, H, W]
        """
        if self.multiplicative_excitation:
            # Get integration parameters - either static or dynamic
            if self.use_dynamic_parameters:
                # Compute dynamic parameters based on inhibited signal and excitatory state
                kappa, omega = self.dynamic_params.compute_excitation_params(exc_input, exc_horizontals)
            else:
                # Use static parameters
                kappa = self.kappa.view(1, -1, 1, 1)
                omega = self.omega.view(1, -1, 1, 1)

                if self.force_omega_nonnegative:
                    omega = F.relu(omega)

            # Apply excitation
            # exc = (omega * exc_input + kappa) * exc_horizontals
            exc = omega * exc_horizontals + kappa
            # exc = F.relu(exc)  # Inputs are non-negative. No need for relu
        else:
            # Simple additive
            raise NotImplementedError("Not using the additive route.")
            exc = F.relu(h_exc + c2)
        return exc

    def forward(self, ff_input: torch.Tensor, h_exc_prev: torch.Tensor,
                h_inh_prev: Optional[torch.Tensor] = None, use_separate_ei_states=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with separate E/I states.

        Args:
            ff_input: Feedforward input [B, C, H, W]
            h_exc_prev: Previous excitatory hidden state [B, C, H, W]
            h_inh_prev: Previous inhibitory hidden state [B, C, H, W] (optional)

        Returns:
            h_exc_next: Next excitatory state [B, C, H, W]
            h_inh_next: Next inhibitory state [B, C, H, W]
            inhibited_signal: Post-inhibition signal [B, C, H, W]
        """
        if not self.use_separate_ei_states:
            raise NotImplementedError("Disallowing this.")

        # Initialize inhibitory state if not provided
        if h_inh_prev is None:
            raise RuntimeError("You must track the I hidden states.")
            # h_inh_prev = h_exc_prev

        #### Inputs to Inh cells are h_inh_prev, h_exc_prev, and inh_input
        # 1. Inhibitory circuit from inhibitory population
        inh_horizontals = self.circuit_input(h_exc_prev)  # Asymmetry: this is a function of exc hidden state.

        # 2. Apply inhibition
        inh_input = ff_input  # Renaming to make it easier to understand
        inhibited_signal = self.apply_inhibition(inh_input, h_inh_prev, inh_horizontals)  # Integrate with Inh states

        # 3. Compute input gate from excitatory and inhibitory states
        inh_gate = self.compute_input_gate(h_exc_prev, h_inh_prev, ff_input)

        # 4. Update inhibitory state with gating
        h_inh_new = (1 - inh_gate) * h_inh_prev + inh_gate * inhibited_signal

        #### Inputs to Inh cells are h_exc_prev and h_inh_new (functions as the FF drive)
        # 5. Excitatory circuit from excitatory population
        exc_input = h_inh_new  # Renaming to make it easier to understand
        exc_horizontals = self.circuit_output(exc_input)

        # 6. Apply excitation and output gating
        h_exc_new = self.apply_excitation(exc_input, exc_horizontals)

        # 7. Compute output gate from updated inhibitory state and previous excitatory state
        exc_gate = self.compute_output_gate(exc_input, h_exc_new, h_exc_prev)

        # 8. Apply output gating
        h_exc_new = (1 - exc_gate) * h_exc_prev + exc_gate * h_exc_new
        return h_exc_new, h_inh_new, inhibited_signal
