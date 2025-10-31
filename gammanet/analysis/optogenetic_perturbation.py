"""In silico optogenetics - computational analog of photoperturbation experiments.

This module implements perturbation-based analysis inspired by:
Keller et al. (2020) "A Disinhibitory Circuit for Contextual Modulation in Primary Visual Cortex"
https://pmc.ncbi.nlm.nih.gov/articles/PMC6682407/

Connection to Keller et al.:
- **Biology**: Photo-stimulate a neuron, measure effect on neighbors via calcium imaging
- **Computation**: Perturb hidden state units, optimize to find idealized circuit response

Unlike static parameter extraction, this performs dynamic perturbation:
1. "Photostimulate" specific units by increasing their activity (perturb_factor > 1.0)
2. Optimize other units' activity to find the ideal compensatory response
3. This reveals what recurrent interactions SHOULD occur given the circuit's learned function

The optimization discovers the idealized excitatory/inhibitory effects on neighboring units
that would optimally maintain the network's function despite the perturbation.
This is analogous to measuring how real neurons adjust their firing in response to
optogenetic stimulation of their neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def compute_orientation_from_components(sin_comp: torch.Tensor, cos_comp: torch.Tensor) -> torch.Tensor:
    """Compute orientation from sin/cos components via atan2 inversion.

    This inverts the 2D grating representation to extract orientation, matching
    the neurophysiology approach where population activity must be decoded.

    Args:
        sin_comp: Sin(2θ) component of grating [batch]
        cos_comp: Cos(2θ) component of grating [batch]

    Returns:
        Orientation in degrees [batch], range [0, 180)
    """
    # atan2 gives angle in radians [-π, π]
    # For orientation (not direction): θ = atan2(sin, cos) / 2
    orientation_rad = torch.atan2(sin_comp, cos_comp) / 2.0

    # Convert to degrees and map to [0, 180)
    orientation_deg = torch.rad2deg(orientation_rad) % 180.0

    return orientation_deg


def mirror_invariant_l2_grating(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mirror-invariant L2 loss for grating components.

    Gratings have 180° rotational symmetry, so [sin(2θ), cos(2θ)] and
    [-sin(2θ), -cos(2θ)] represent the same orientation. This loss function
    accounts for this symmetry by taking the minimum of the two possibilities.

    Matches the TensorFlow reference:
        tf.minimum(tf.nn.l2_loss(labels - logits), tf.nn.l2_loss(-labels - logits))

    Args:
        predictions: Predicted grating components [batch, 2]
        targets: Target grating components [batch, 2]

    Returns:
        Scalar loss value
    """
    # Compute both possible L2 losses
    # tf.nn.l2_loss computes sum(x^2) / 2
    loss_normal = ((predictions - targets) ** 2).sum() / 2.0
    loss_flipped = ((predictions + targets) ** 2).sum() / 2.0  # predictions - (-targets)

    # Return minimum (handles 180° symmetry)
    return torch.minimum(loss_normal, loss_flipped)


class OptogeneticPerturbation:
    """Perform in silico optogenetic perturbation experiments.

    This class implements computational analogs of optogenetic experiments where
    specific neurons are photo-stimulated and the effects on the circuit are measured.

    Example:
        >>> perturber = OptogeneticPerturbation(model, device='cuda')
        >>> influence_map = perturber.measure_influence_map(
        ...     input_image,
        ...     layer_name='fgru_0',
        ...     perturb_location=(80, 80),
        ...     perturb_factor=1.2
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        patch_size: int = 4
    ):
        """Initialize perturbation analyzer.

        Args:
            model: GammaNet model to analyze
            device: Device to run on ('cuda' or 'cpu')
            patch_size: Size of spatial patch to perturb (default: 4x4)
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.to(device)
        self.model.eval()

        # Storage for intermediate activations
        self.activations = {}
        self.hooks = []

        # Orientation decoder (will be trained when needed)
        self.decoder = None
        self.decoder_location = None  # Spatial location for decoder readout

    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks to capture activations.

        Args:
            layer_names: Names of layers to capture
        """
        self.remove_hooks()

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def find_optimal_stimulus_parameters(
        self,
        layer_name: str,
        decoder_location: Tuple[int, int],
        stimulus_generator,
        spatial_frequencies: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0],
        contrasts: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
        stimulus_diameter: int = 5,
        orientations: List[float] = None,
        decoder_epochs: int = 20,
    ) -> Dict[str, float]:
        """Find optimal spatial frequency and contrast based on orientation decoder fit quality.

        For each SF/contrast combination, trains a quick decoder and measures how well
        it can discriminate orientations. Selects parameters with best decoder performance.

        Args:
            layer_name: Target layer to extract activities from
            decoder_location: (h, w) location to read from
            stimulus_generator: OrientedGratingStimuli instance
            spatial_frequencies: List of SFs to test (cycles per image)
            contrasts: List of contrasts to test [0, 1]
            stimulus_diameter: Grating diameter in pixels (CRF size)
            orientations: Orientations to test (default: 0-165° in 15° steps)
            decoder_epochs: Number of training epochs per decoder (default: 20)

        Returns:
            Dictionary with:
                'optimal_sf': Best spatial frequency
                'optimal_contrast': Best contrast
                'best_r2': R² at optimal params
                'results': Full grid of (sf, contrast) -> R²
        """
        if orientations is None:
            orientations = np.arange(0, 180, 15)

        print(f"\nSearching for optimal stimulus parameters based on decoder fit quality...")
        print(f"  Testing {len(spatial_frequencies)} SFs × {len(contrasts)} contrasts = {len(spatial_frequencies) * len(contrasts)} combinations")
        print(f"  Training quick decoder ({decoder_epochs} epochs) for each combination")
        print(f"  Measuring decoder R² (coefficient of determination)")

        # Register hook to extract activities
        self.register_hooks([layer_name])

        results = {}
        best_r2 = -np.inf
        optimal_sf = None
        optimal_contrast = None

        for sf in spatial_frequencies:
            for contrast in contrasts:
                # Generate gratings at this SF/contrast for all orientations
                activities = []
                grating_components = []  # Ground truth [sin(2θ), cos(2θ)]

                for ori in orientations:
                    stimuli = stimulus_generator.generate_stimulus_set(
                        orientations=[ori],
                        spatial_frequencies=[sf],
                        contrasts=[contrast],
                        stimulus_diameter=stimulus_diameter
                    )
                    stim, meta = stimuli[0]
                    stim_tensor = torch.from_numpy(stim).unsqueeze(0).unsqueeze(0).float()
                    stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(self.device)

                    # Forward pass
                    with torch.no_grad():
                        _ = self.model(stim_tensor)

                    # Extract activity at decoder location
                    activation = self.activations[layer_name]
                    if isinstance(activation, tuple):
                        activation = activation[0]  # Use excitatory state

                    h_loc, w_loc = decoder_location
                    if len(activation.shape) == 4:
                        activity_at_loc = activation[:, :, h_loc, w_loc]  # [B, C]
                    else:
                        activity_at_loc = activation

                    activities.append(activity_at_loc.cpu())

                    # Compute target grating components
                    ori_rad = np.radians(ori)
                    sin_comp = np.sin(2 * ori_rad)
                    cos_comp = np.cos(2 * ori_rad)
                    grating_components.append([sin_comp, cos_comp])

                # Stack activities [N_orientations, C] and targets [N_orientations, 2]
                X = torch.cat(activities, dim=0)  # [N, C]
                y = torch.tensor(grating_components, dtype=torch.float32)  # [N, 2]

                # Train quick decoder
                input_dim = X.shape[1]
                temp_decoder = nn.Linear(input_dim, 2).to(self.device)
                X_device = X.to(self.device)
                y_device = y.to(self.device)

                optimizer = torch.optim.Adam(temp_decoder.parameters(), lr=0.01)
                criterion = nn.MSELoss()

                # Quick training loop
                temp_decoder.train()
                for epoch in range(decoder_epochs):
                    optimizer.zero_grad()
                    pred = temp_decoder(X_device)
                    loss = criterion(pred, y_device)
                    loss.backward()
                    optimizer.step()

                # Evaluate decoder: compute R² on training set
                temp_decoder.eval()
                with torch.no_grad():
                    pred = temp_decoder(X_device)
                    ss_res = ((y_device - pred) ** 2).sum().item()
                    ss_tot = ((y_device - y_device.mean(dim=0)) ** 2).sum().item()
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

                results[(sf, contrast)] = r2

                if r2 > best_r2:
                    best_r2 = r2
                    optimal_sf = sf
                    optimal_contrast = contrast

                print(f"    SF={sf:5.1f}, Contrast={contrast:.2f} → R²={r2:.4f}")

        self.remove_hooks()

        print(f"\n  ✓ Optimal parameters for CRF orientation decoding:")
        print(f"    Spatial Frequency: {optimal_sf} cycles/image")
        print(f"    Contrast: {optimal_contrast}")
        print(f"    Decoder R²: {best_r2:.4f}")

        return {
            'optimal_sf': optimal_sf,
            'optimal_contrast': optimal_contrast,
            'best_r2': best_r2,
            'results': results
        }

    def train_decoder(
        self,
        grating_stimuli: List[Tuple[torch.Tensor, float]],
        layer_name: str = 'fgru_0',
        decoder_location: Optional[Tuple[int, int]] = None,
        num_epochs: int = 100,
        learning_rate: float = 0.01
    ):
        """Train linear decoder to map fGRU activities to orientation.

        Args:
            grating_stimuli: List of (stimulus_tensor, orientation_degrees) tuples
            layer_name: Which fGRU layer to decode from (default: fgru_0)
            decoder_location: (h, w) location in feature space to decode from.
                            If None, uses center of feature map.
            num_epochs: Number of training epochs
            learning_rate: Learning rate for decoder training
        """
        print(f"Training orientation decoder on {layer_name}...")

        # Register hook to capture target layer
        self.register_hooks([layer_name])

        # Collect fGRU activities and labels
        activities = []
        labels = []

        with torch.no_grad():
            for stimulus, orientation in tqdm(grating_stimuli, desc="Extracting activities"):
                # Reset hidden states
                if hasattr(self.model, 'reset_hidden_states'):
                    self.model.reset_hidden_states()

                # Forward pass
                _ = self.model(stimulus.to(self.device))

                # Extract fGRU excitatory state at final timestep
                # The output is a tuple (h_exc, h_inh, ...)
                if layer_name in self.activations:
                    layer_output = self.activations[layer_name]
                    if isinstance(layer_output, tuple):
                        h_exc = layer_output[0]  # Excitatory state
                    else:
                        h_exc = layer_output

                    # Determine decoder location if not specified
                    if decoder_location is None:
                        _, _, h, w = h_exc.shape
                        decoder_location = (h // 2, w // 2)

                    # Extract at specific spatial location
                    h_loc, w_loc = decoder_location
                    h_at_location = h_exc[:, :, h_loc, w_loc]  # [B, C]

                    activities.append(h_at_location.cpu())

                    # Convert orientation to one-hot (12 bins: 0, 15, 30, ..., 165)
                    orientation_bin = int(orientation / 15) % 12
                    labels.append(orientation_bin)

        # Stack into tensors
        activities_tensor = torch.cat(activities, dim=0)  # [N, C]
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Store decoder location for later use
        self.decoder_location = decoder_location
        print(f"  Decoder location: {decoder_location} in feature space")

        # Compute sin/cos components for each orientation (labels)
        # For orientation θ, we use: sin(2θ), cos(2θ)
        orientations_rad = torch.deg2rad(torch.tensor([ori for _, ori in grating_stimuli], dtype=torch.float32))
        sin_targets = torch.sin(2 * orientations_rad)  # [N]
        cos_targets = torch.cos(2 * orientations_rad)  # [N]
        grating_components = torch.stack([sin_targets, cos_targets], dim=1)  # [N, 2]

        # Create decoder: maps activities to 2D grating components (sin, cos)
        num_channels = activities_tensor.shape[1]
        self.decoder = nn.Linear(num_channels, 2).to(self.device)

        # Train decoder
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward
            activities_batch = activities_tensor.to(self.device)
            predictions = self.decoder(activities_batch)  # [N, 2]

            # Loss: Mirror-invariant L2 loss (handles 180° symmetry)
            loss = mirror_invariant_l2_grating(predictions, grating_components.to(self.device))

            # Backward
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                # Calculate orientation error by inverting predictions
                pred_sin, pred_cos = predictions[:, 0], predictions[:, 1]
                pred_orientations = compute_orientation_from_components(pred_sin, pred_cos)
                true_orientations = torch.tensor([ori for _, ori in grating_stimuli], dtype=torch.float32).to(self.device)

                # Angular error (accounting for 180° periodicity)
                angular_error = torch.abs(pred_orientations - true_orientations)
                angular_error = torch.min(angular_error, 180 - angular_error).mean()

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Angular Error: {angular_error.item():.2f}°")

        self.remove_hooks()
        print("Decoder training complete!")

    def create_perturbation_mask(
        self,
        feature_shape: Tuple[int, ...],
        perturb_location: Tuple[int, int],
    ) -> torch.Tensor:
        """Create binary mask for perturbation.

        Args:
            feature_shape: Shape of feature map [B, C, H, W]
            perturb_location: (h, w) center of perturbation

        Returns:
            Binary mask [B, C, H, W] where 1 = perturb, 0 = leave alone

        Note:
            Uses circular mask with radius = patch_size / 2
            (e.g., patch_size=5 means 5px diameter = 2.5px radius)
        """
        b, c, h, w = feature_shape
        h_center, w_center = perturb_location

        # Create circular mask based on patch_size (diameter)
        radius = self.patch_size / 2.0

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )

        # Compute distances from center
        distances = torch.sqrt(
            (y_coords - h_center).float()**2 +
            (x_coords - w_center).float()**2
        )

        # Create circular mask
        mask = (distances <= radius).float()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(feature_shape)

        return mask

    def perturb_hidden_state(
        self,
        hidden_state: torch.Tensor,
        perturb_mask: torch.Tensor,
        perturb_factor: float = 1.2
    ) -> torch.Tensor:
        """Apply perturbation to hidden state.

        This simulates "photo-stimulating" specific units by multiplying
        their activity.

        Args:
            hidden_state: Original hidden state [B, C, H, W]
            perturb_mask: Binary mask indicating which units to perturb
            perturb_factor: Multiplicative factor (e.g., 1.2 = 20% increase)

        Returns:
            Perturbed hidden state
        """
        # Apply perturbation: perturbed units get multiplied, others unchanged
        perturbed = hidden_state * (1.0 + perturb_mask * (perturb_factor - 1.0))
        return perturbed

    def forward_with_perturbation(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        perturb_location: Tuple[int, int],
        perturb_factor: float = 1.2
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass with perturbation at specific layer.

        This modifies the model's internal state to apply perturbation
        during the forward pass.

        Args:
            input_image: Input image [B, C, H, W]
            layer_name: Name of layer to perturb (e.g., 'fgru_0')
            perturb_location: (h, w) location to perturb
            perturb_factor: Strength of perturbation

        Returns:
            Dictionary with:
                - 'output': Model output
                - 'activations': All captured activations
                - 'perturbation_mask': The mask used
        """
        # Register hooks to capture activations
        self.register_hooks([layer_name])

        # First forward pass to get activation shape
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()
        with torch.no_grad():
            _ = self.model(input_image)

        # Get the activation shape for creating mask
        target_activation = self.activations[layer_name]
        perturb_mask = self.create_perturbation_mask(
            target_activation.shape,
            perturb_location
        )

        # Create a hook that applies perturbation
        def perturb_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                # Apply perturbation
                h_perturbed = self.perturb_hidden_state(h, perturb_mask, perturb_factor)
                return (h_perturbed,) + output[1:]
            else:
                return self.perturb_hidden_state(output, perturb_mask, perturb_factor)

        # Find the target module and register perturbation hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                perturb_handle = module.register_forward_hook(perturb_hook)
                break

        # Forward pass with perturbation
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()
        with torch.no_grad():
            output = self.model(input_image)

        # Clean up
        perturb_handle.remove()

        return {
            'output': output,
            'activations': dict(self.activations),
            'perturbation_mask': perturb_mask
        }

    def measure_influence_map(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        perturb_locations: Optional[List[Tuple[int, int]]] = None,
        perturb_factor: float = 1.2,
        grid_spacing: int = 16
    ) -> Dict[str, np.ndarray]:
        """Measure influence map by perturbing multiple locations.

        This creates a map showing how perturbation at each location
        affects the rest of the circuit - revealing causal interactions.

        Args:
            input_image: Input image [B, C, H, W]
            layer_name: Layer to perturb
            perturb_locations: Specific locations to test, or None for grid
            perturb_factor: Strength of perturbation
            grid_spacing: If perturb_locations is None, spacing of grid

        Returns:
            Dictionary containing:
                - 'influence_map': [H, W] map of influence strength
                - 'excitatory_influence': [H, W] facilitatory effects
                - 'inhibitory_influence': [H, W] suppressive effects
        """
        input_image = input_image.to(self.device)

        # Get baseline activation
        self.register_hooks([layer_name])
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()
        with torch.no_grad():
            baseline_output = self.model(input_image)
            baseline_activation = self.activations[layer_name].clone()

        b, c, h, w = baseline_activation.shape

        # Define grid of perturbation locations if not provided
        if perturb_locations is None:
            h_locs = range(self.patch_size, h - self.patch_size, grid_spacing)
            w_locs = range(self.patch_size, w - self.patch_size, grid_spacing)
            perturb_locations = [(hh, ww) for hh in h_locs for ww in w_locs]

        # Measure influence for each location
        influence_map = np.zeros((h, w))
        excitatory_map = np.zeros((h, w))
        inhibitory_map = np.zeros((h, w))

        for loc in tqdm(perturb_locations, desc="Measuring influence"):
            # Perturb at this location
            result = self.forward_with_perturbation(
                input_image,
                layer_name,
                loc,
                perturb_factor
            )

            perturbed_activation = result['activations'][layer_name]

            # Compute change in activation
            delta = (perturbed_activation - baseline_activation).cpu().numpy()
            delta_mean = np.mean(np.abs(delta[0]), axis=0)  # Average over channels

            # Decompose into excitatory and inhibitory
            excitatory = np.mean(np.maximum(delta[0], 0), axis=0)
            inhibitory = np.mean(np.maximum(-delta[0], 0), axis=0)

            # Store in maps
            hh, ww = loc
            influence_map[hh, ww] = delta_mean.sum()
            excitatory_map[hh, ww] = excitatory.sum()
            inhibitory_map[hh, ww] = inhibitory.sum()

        self.remove_hooks()

        return {
            'influence_map': influence_map,
            'excitatory_influence': excitatory_map,
            'inhibitory_influence': inhibitory_map,
            'perturb_locations': perturb_locations
        }

    def optimize_circuit_response(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        perturb_location: Tuple[int, int],
        perturb_factor: float = 1.2,
        num_steps: int = 100,
        learning_rate: float = 0.01,
        ecrf_radius: int = 20,
        stimulus_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Optimize initial hidden states to compensate for perturbation.

        Instead of modifying states during forward pass, this optimizes what
        the circuit's initial hidden states should be to maintain function
        despite the perturbation that will be applied during processing.

        Args:
            input_image: Input image
            layer_name: Layer to perturb
            perturb_location: Location to perturb
            perturb_factor: Perturbation strength
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimization

        Returns:
            Dictionary with:
                - 'adjustment': Learned adjustment pattern
                - 'loss_curve': Optimization loss over time
        """
        input_image = input_image.to(self.device)

        # Check decoder is trained
        if self.decoder is None:
            raise ValueError("Decoder not trained! Call train_decoder() first.")

        # Get baseline fGRU activities and orientation WITHOUT any hooks
        # This ensures we capture the true baseline
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()

        # Temporarily register hook just to capture baseline
        baseline_activation = {}
        def capture_baseline(module, input, output):
            if isinstance(output, tuple):
                baseline_activation['h_exc'] = output[0].detach()
            else:
                baseline_activation['h_exc'] = output.detach()

        # Get target fGRU module
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found in model")

        # Get feedforward conv module that feeds into this fGRU
        # Map fGRU layers to their input conv blocks
        ff_conv_mapping = {
            'fgru_0': 'block1_conv',
            'fgru_1': 'block2_conv',
            'fgru_2': 'block3_conv',
            'fgru_3': 'block4_conv',
            'fgru_4': 'block5_conv',
        }

        ff_conv_name = ff_conv_mapping.get(layer_name)
        if ff_conv_name is None:
            raise ValueError(f"No feedforward conv block found for layer {layer_name}")

        ff_conv_module = None
        for name, module in self.model.named_modules():
            if name == ff_conv_name:
                ff_conv_module = module
                break
        if ff_conv_module is None:
            raise ValueError(f"Feedforward conv block {ff_conv_name} not found in model")

        # Register temporary hook for baseline
        temp_handle = target_module.register_forward_hook(capture_baseline)

        with torch.no_grad():
            _ = self.model(input_image)
            baseline_h_exc = baseline_activation['h_exc']

            # Extract at decoder location
            if self.decoder_location is not None:
                h_loc, w_loc = self.decoder_location
                baseline_h_exc_at_loc = baseline_h_exc[:, :, h_loc, w_loc]  # [B, C]
            else:
                # Fallback to global average pooling
                baseline_h_exc_at_loc = baseline_h_exc.mean(dim=(-2, -1))  # [B, C]

            # Decode to grating components, then invert to orientation
            baseline_components = self.decoder(baseline_h_exc_at_loc).detach()  # [B, 2]
            baseline_sin, baseline_cos = baseline_components[:, 0], baseline_components[:, 1]
            baseline_orientation = compute_orientation_from_components(baseline_sin, baseline_cos)  # [B]

        # Remove temporary hook
        temp_handle.remove()

        # Store activation shape for masks
        activation_shape = baseline_h_exc.shape

        # Create perturbation mask
        perturb_mask = self.create_perturbation_mask(
            activation_shape,
            perturb_location
        )

        # Create optimization mask - only optimize within eCRF, excluding perturbed region
        b, c, h, w = activation_shape
        h_center, w_center = perturb_location

        # Create eCRF mask (circular region around perturbation)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )
        distances = torch.sqrt(
            (y_coords - h_center).float()**2 +
            (x_coords - w_center).float()**2
        )
        ecrf_mask = (distances <= ecrf_radius).float()
        ecrf_mask = ecrf_mask.unsqueeze(0).unsqueeze(0).expand_as(perturb_mask)

        # If stimulus mask provided, further constrain to stimulus region
        if stimulus_mask is not None:
            # Ensure stimulus mask has right shape
            if stimulus_mask.dim() == 2:
                stimulus_mask = stimulus_mask.unsqueeze(0).unsqueeze(0)

            # Resize stimulus mask to match activation shape if needed
            if stimulus_mask.shape[-2:] != (h, w):
                stimulus_mask = F.interpolate(
                    stimulus_mask,
                    size=(h, w),
                    mode='nearest'
                )

            stimulus_mask = stimulus_mask.expand_as(perturb_mask)
            ecrf_mask = ecrf_mask * stimulus_mask

        # Final optimization mask: within eCRF but not in perturbed region
        optimization_mask = ecrf_mask * (1.0 - perturb_mask)

        # Create learnable INITIAL STATE adjustments for both E and I states
        # These are ADDITIVE adjustments applied to initial hidden states before forward pass
        # Initialize with small random values to break symmetry
        init_adjustment_exc = nn.Parameter(
            torch.randn_like(perturb_mask) * 0.01 * optimization_mask
        )
        init_adjustment_inh = nn.Parameter(
            torch.randn_like(perturb_mask) * 0.01 * optimization_mask
        )
        optimizer = torch.optim.Adam([init_adjustment_exc, init_adjustment_inh], lr=learning_rate)

        # Optimization loop
        loss_curve = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Reset model hidden states to zero/baseline
            if hasattr(self.model, 'reset_hidden_states'):
                self.model.reset_hidden_states()

            # Track timestep for initial state modification
            timestep_counter = {'count': 0}
            # Store perturbed output from last timestep
            perturbed_output = {'h_exc': None}

            # Hook 1: Perturb FEEDFORWARD input (conv output before fGRU)
            def ff_perturbation_hook(module, input, output):
                """Perturb feedforward drive at center location.

                This lesions the LGN/FF input before it reaches the recurrent circuit.
                """
                # output is [B, C, H, W] from conv layer
                # Apply multiplicative perturbation at center region
                output_perturbed = output * (perturb_factor * perturb_mask + (1 - perturb_mask))
                return output_perturbed

            # Hook 2: Apply learned t=0 adjustments to fGRU hidden states
            def fgru_adjustment_hook(module, input, output):
                """Apply learned initial state adjustments + capture output.

                This adjusts the recurrent hidden states at t=0 to rescue the center.
                """
                current_timestep = timestep_counter['count']
                timestep_counter['count'] += 1

                if isinstance(output, tuple) and len(output) >= 2:
                    # Separate E/I states from fGRU output
                    h_exc, h_inh = output[0], output[1]
                    rest = output[2:] if len(output) > 2 else ()

                    # Apply learned ADDITIVE adjustments ONLY at t=0
                    # This optimizes the initial hidden state in eCRF surround
                    if current_timestep == 0:
                        h_exc_adjusted = h_exc + init_adjustment_exc * optimization_mask
                        h_inh_adjusted = h_inh + init_adjustment_inh * optimization_mask
                    else:
                        h_exc_adjusted = h_exc
                        h_inh_adjusted = h_inh

                    # Capture final timestep output (gradients still flow!)
                    perturbed_output['h_exc'] = h_exc_adjusted

                    return (h_exc_adjusted, h_inh_adjusted) + rest
                else:
                    # Single state (fallback)
                    h = output[0] if isinstance(output, tuple) else output

                    # Apply learned adjustment only at t=0
                    if current_timestep == 0:
                        h_adjusted = h + init_adjustment_exc * optimization_mask
                    else:
                        h_adjusted = h

                    # Capture output
                    perturbed_output['h_exc'] = h_adjusted

                    if isinstance(output, tuple):
                        return (h_adjusted,) + output[1:]
                    else:
                        return h_adjusted

            # Register both hooks
            ff_handle = ff_conv_module.register_forward_hook(ff_perturbation_hook)
            fgru_handle = target_module.register_forward_hook(fgru_adjustment_hook)

            # Forward pass
            _ = self.model(input_image)

            # Extract perturbed fGRU activities (captured from hook with gradients!)
            h_exc_perturbed = perturbed_output['h_exc']

            # Extract at decoder location
            if self.decoder_location is not None:
                h_loc, w_loc = self.decoder_location
                h_exc_perturbed_at_loc = h_exc_perturbed[:, :, h_loc, w_loc]  # [B, C]
            else:
                # Fallback to global average pooling
                h_exc_perturbed_at_loc = h_exc_perturbed.mean(dim=(-2, -1))  # [B, C]

            # Decode perturbed grating components
            perturbed_components = self.decoder(h_exc_perturbed_at_loc)  # [B, 2]

            # Loss: Mirror-invariant L2 between perturbed and baseline components
            # This accounts for 180° grating symmetry: [sin, cos] ≈ [-sin, -cos]
            loss = mirror_invariant_l2_grating(perturbed_components, baseline_components)

            # Backward pass
            loss.backward()

            # CRITICAL: Mask gradients BEFORE optimizer step to prevent updates outside eCRF
            with torch.no_grad():
                if init_adjustment_exc.grad is not None:
                    init_adjustment_exc.grad.mul_(optimization_mask)
                if init_adjustment_inh.grad is not None:
                    init_adjustment_inh.grad.mul_(optimization_mask)

            # Optimizer step (only updates unmasked regions now)
            optimizer.step()

            # Additional safety: zero out any remaining values outside mask
            with torch.no_grad():
                init_adjustment_exc.mul_(optimization_mask)
                init_adjustment_inh.mul_(optimization_mask)

            loss_curve.append(loss.item())

            # Clean up both hooks
            ff_handle.remove()
            fgru_handle.remove()

        self.remove_hooks()

        # Zero out masked regions for clean visualization
        adjustment_exc_final = (init_adjustment_exc.detach() * optimization_mask).cpu()
        adjustment_inh_final = (init_adjustment_inh.detach() * optimization_mask).cpu()

        return {
            'adjustment_exc': adjustment_exc_final,
            'adjustment_inh': adjustment_inh_final,
            'optimization_mask': optimization_mask.cpu(),
            'perturb_mask': perturb_mask.cpu(),
            'ecrf_mask': ecrf_mask.cpu(),
            'loss_curve': np.array(loss_curve),
            'final_loss': loss_curve[-1]
        }

    def optimize_circuit_response_multi_orientation(
        self,
        grating_stimuli: List[Tuple[torch.Tensor, float]],
        layer_name: str,
        perturb_location: Tuple[int, int],
        perturb_factor: float = 1.2,
        num_steps: int = 100,
        learning_rate: float = 0.01,
        ecrf_radius: int = 20,
        stimulus_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Optimize circuit response across multiple orientations, then rotate and average.

        This reveals orientation-invariant recurrent RF structure by:
        1. Optimizing E/I adjustments for each orientation independently
        2. Rotating each adjustment map to align to 0° reference
        3. Averaging rotated maps to get canonical recurrent structure

        Args:
            grating_stimuli: List of (stimulus_tensor, orientation) tuples
            layer_name: Layer to perturb
            perturb_location: Location to perturb
            perturb_factor: Perturbation strength
            num_steps: Number of optimization steps per orientation
            learning_rate: Learning rate
            ecrf_radius: Radius of eCRF for optimization
            stimulus_mask: Optional mask constraining optimization region

        Returns:
            Dictionary with:
                - 'adjustment_exc_avg': Averaged excitatory adjustments (orientation-invariant)
                - 'adjustment_inh_avg': Averaged inhibitory adjustments (orientation-invariant)
                - 'adjustment_exc_individual': List of adjustments per orientation (before rotation)
                - 'adjustment_inh_individual': List of adjustments per orientation (before rotation)
                - 'adjustment_exc_rotated': List of rotated adjustments per orientation
                - 'adjustment_inh_rotated': List of rotated adjustments per orientation
                - 'orientations': List of orientations tested
                - 'optimization_mask': Mask used for optimization
                - 'loss_curves': List of loss curves per orientation
        """
        from scipy.ndimage import rotate as scipy_rotate

        print(f"\n{'='*70}")
        print(f"MULTI-ORIENTATION OPTIMIZATION")
        print(f"{'='*70}")
        print(f"  Optimizing for {len(grating_stimuli)} orientations")
        print(f"  Each orientation: {num_steps} optimization steps")
        print(f"  After optimization: rotate to 0° reference and average")

        # Store results for each orientation
        orientations = []
        adjustments_exc_individual = []
        adjustments_inh_individual = []
        adjustments_exc_rotated = []
        adjustments_inh_rotated = []
        loss_curves = []
        optimization_mask = None

        # Optimize for each orientation
        for idx, (stim, ori) in enumerate(grating_stimuli):
            print(f"\n  [{idx+1}/{len(grating_stimuli)}] Optimizing for orientation {ori}°...")

            # Run optimization for this orientation
            result = self.optimize_circuit_response(
                stim,
                layer_name=layer_name,
                perturb_location=perturb_location,
                perturb_factor=perturb_factor,
                num_steps=num_steps,
                learning_rate=learning_rate,
                ecrf_radius=ecrf_radius,
                stimulus_mask=stimulus_mask
            )

            orientations.append(ori)
            loss_curves.append(result['loss_curve'])

            # Get adjustments [B, C, H, W] - squeeze batch dim
            adj_exc = result['adjustment_exc'][0].numpy()  # [C, H, W]
            adj_inh = result['adjustment_inh'][0].numpy()  # [C, H, W]

            adjustments_exc_individual.append(adj_exc)
            adjustments_inh_individual.append(adj_inh)

            # Store mask (same for all orientations)
            if optimization_mask is None:
                optimization_mask = result['optimization_mask']

            # Rotate each channel map to align to 0° reference
            # Rotate by -ori to align to 0°
            adj_exc_rotated = np.zeros_like(adj_exc)
            adj_inh_rotated = np.zeros_like(adj_inh)

            for c in range(adj_exc.shape[0]):
                # Rotate spatial maps (H, W) by -ori degrees
                # reshape=False keeps output size same, order=1 is bilinear interpolation
                adj_exc_rotated[c] = scipy_rotate(
                    adj_exc[c],
                    angle=-ori,
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                adj_inh_rotated[c] = scipy_rotate(
                    adj_inh[c],
                    angle=-ori,
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=0.0
                )

            adjustments_exc_rotated.append(adj_exc_rotated)
            adjustments_inh_rotated.append(adj_inh_rotated)

            print(f"      Final loss: {result['final_loss']:.6f}")

        # Average rotated adjustments
        print(f"\n  Averaging rotated adjustments across {len(orientations)} orientations...")
        adj_exc_avg = np.mean(adjustments_exc_rotated, axis=0)  # [C, H, W]
        adj_inh_avg = np.mean(adjustments_inh_rotated, axis=0)  # [C, H, W]

        print(f"  ✓ Multi-orientation optimization complete")
        print(f"\n  Averaged adjustment statistics:")
        print(f"    Exc mean: {adj_exc_avg.mean():.6f}, std: {adj_exc_avg.std():.6f}")
        print(f"    Inh mean: {adj_inh_avg.mean():.6f}, std: {adj_inh_avg.std():.6f}")

        return {
            'adjustment_exc_avg': torch.from_numpy(adj_exc_avg).unsqueeze(0),  # Add batch dim back
            'adjustment_inh_avg': torch.from_numpy(adj_inh_avg).unsqueeze(0),
            'adjustment_exc_individual': adjustments_exc_individual,
            'adjustment_inh_individual': adjustments_inh_individual,
            'adjustment_exc_rotated': adjustments_exc_rotated,
            'adjustment_inh_rotated': adjustments_inh_rotated,
            'orientations': orientations,
            'optimization_mask': optimization_mask,
            'loss_curves': loss_curves
        }

    def measure_recurrent_contributions(
        self,
        input_image: torch.Tensor,
        layer_name: str = 'fgru_0',
        perturb_location: Tuple[int, int] = (128, 128),
        perturb_factor: float = 0.0,  # 0.0 = complete lesion
    ) -> Dict[str, np.ndarray]:
        """Measure where recurrent E/I flows from to rescue perturbed center.

        This is **Approach A**: Measure the actual recurrent flow in the learned model.

        Args:
            input_image: Input tensor [B, 3, H, W]
            layer_name: fGRU layer to analyze
            perturb_location: (h, w) center location to perturb
            perturb_factor: Perturbation strength (0.0 = complete lesion)

        Returns:
            Dictionary containing:
            - 'exc_recurrent_baseline': Exc recurrent input to center [timesteps, channels]
            - 'inh_recurrent_baseline': Inh recurrent input to center [timesteps, channels]
            - 'exc_recurrent_perturbed': Exc recurrent with FF perturbation [timesteps, channels]
            - 'inh_recurrent_perturbed': Inh recurrent with FF perturbation [timesteps, channels]
            - 'exc_spatial_baseline': Spatial map of exc contributions [timesteps, h, w]
            - 'inh_spatial_baseline': Spatial map of inh contributions [timesteps, h, w]
            - 'exc_spatial_perturbed': Spatial map with perturbation [timesteps, h, w]
            - 'inh_spatial_perturbed': Spatial map with perturbation [timesteps, h, w]
        """
        self.model.eval()
        input_image = input_image.to(self.device)

        # Get target fGRU module and FF conv module
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found")

        ff_conv_mapping = {
            'fgru_0': 'block1_conv',
            'fgru_1': 'block2_conv',
            'fgru_2': 'block3_conv',
            'fgru_3': 'block4_conv',
            'fgru_4': 'block5_conv',
        }
        ff_conv_name = ff_conv_mapping.get(layer_name)
        if ff_conv_name is None:
            raise ValueError(f"No feedforward conv block found for layer {layer_name}")

        ff_conv_module = None
        for name, module in self.model.named_modules():
            if name == ff_conv_name:
                ff_conv_module = module
                break
        if ff_conv_module is None:
            raise ValueError(f"Feedforward conv block {ff_conv_name} not found")

        # Get fGRU horizontal connection weights
        W_exc = target_module.W_exc  # [C, C, K, K]
        W_inh = target_module.W_inh  # [C, C, K, K]

        # Create perturbation mask
        # First do a forward pass to get feature shape
        with torch.no_grad():
            _ = self.model(input_image)

        # Get activation from target layer
        temp_activation = {'output': None}
        def capture_shape(module, input, output):
            if isinstance(output, tuple):
                temp_activation['output'] = output[0]
            else:
                temp_activation['output'] = output

        temp_handle = target_module.register_forward_hook(capture_shape)
        with torch.no_grad():
            _ = self.model(input_image)
        temp_handle.remove()

        activation_shape = temp_activation['output'].shape
        perturb_mask = self.create_perturbation_mask(activation_shape, perturb_location)

        def measure_flow(apply_perturbation=False):
            """Run forward pass and measure recurrent flow."""
            # Reset hidden states
            if hasattr(self.model, 'reset_hidden_states'):
                self.model.reset_hidden_states()

            # Storage for recurrent contributions
            exc_recurrent_to_center = []
            inh_recurrent_to_center = []
            exc_spatial_maps = []
            inh_spatial_maps = []
            h_states = []

            # Hook to capture hidden states at each timestep
            def capture_states(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    h_exc, h_inh = output[0], output[1]
                    h_states.append({
                        'h_exc': h_exc.detach().clone(),
                        'h_inh': h_inh.detach().clone()
                    })

            # Hook to perturb FF input if requested
            def ff_perturb(module, input, output):
                if apply_perturbation:
                    return output * (perturb_factor * perturb_mask + (1 - perturb_mask))
                return output

            # Register hooks
            fgru_handle = target_module.register_forward_hook(capture_states)
            ff_handle = ff_conv_module.register_forward_hook(ff_perturb)

            with torch.no_grad():
                _ = self.model(input_image)

            # Clean up
            fgru_handle.remove()
            ff_handle.remove()

            # Now compute recurrent contributions from each timestep
            for t, state in enumerate(h_states):
                h_exc = state['h_exc']
                h_inh = state['h_inh']

                # Compute horizontal recurrent inputs
                # Use same symmetric conv as fGRU
                if target_module.use_symmetric_conv and target_module.symmetric_weights:
                    exc_recurrent = target_module._symmetric_conv2d(h_exc, W_exc)
                    inh_recurrent = target_module._symmetric_conv2d(h_inh, W_inh)
                else:
                    exc_recurrent = F.conv2d(h_exc, W_exc, padding='same')
                    inh_recurrent = F.conv2d(h_inh, W_inh, padding='same')

                # Extract at center location
                h_center, w_center = perturb_location
                exc_to_center = exc_recurrent[:, :, h_center, w_center]  # [B, C]
                inh_to_center = inh_recurrent[:, :, h_center, w_center]  # [B, C]

                exc_recurrent_to_center.append(exc_to_center.detach().cpu().numpy())
                inh_recurrent_to_center.append(inh_to_center.detach().cpu().numpy())

                # Store full spatial maps
                exc_spatial_maps.append(exc_recurrent.squeeze(0).detach().cpu().numpy())  # [C, H, W]
                inh_spatial_maps.append(inh_recurrent.squeeze(0).detach().cpu().numpy())

            return {
                'exc_to_center': np.array(exc_recurrent_to_center),  # [T, B, C]
                'inh_to_center': np.array(inh_recurrent_to_center),
                'exc_spatial': np.array(exc_spatial_maps),  # [T, C, H, W]
                'inh_spatial': np.array(inh_spatial_maps),
            }

        # Measure baseline (no perturbation)
        print("Measuring baseline recurrent flow...")
        baseline_flow = measure_flow(apply_perturbation=False)

        # Measure with perturbation
        print(f"Measuring recurrent flow with {perturb_factor} FF perturbation...")
        perturbed_flow = measure_flow(apply_perturbation=True)

        return {
            'exc_recurrent_baseline': baseline_flow['exc_to_center'],
            'inh_recurrent_baseline': baseline_flow['inh_to_center'],
            'exc_recurrent_perturbed': perturbed_flow['exc_to_center'],
            'inh_recurrent_perturbed': perturbed_flow['inh_to_center'],
            'exc_spatial_baseline': baseline_flow['exc_spatial'],
            'inh_spatial_baseline': baseline_flow['inh_spatial'],
            'exc_spatial_perturbed': perturbed_flow['exc_spatial'],
            'inh_spatial_perturbed': perturbed_flow['inh_spatial'],
            'perturb_location': perturb_location,
            'perturb_factor': perturb_factor,
        }


def visualize_influence_map(
    influence_data: Dict[str, np.ndarray],
    save_path: Optional[Path] = None
):
    """Visualize influence map results.

    Args:
        influence_data: Output from measure_influence_map()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total influence - use percentile scaling
    total_map = influence_data['influence_map']
    vmin_total = np.percentile(total_map[total_map != 0], 2.5) if np.any(total_map != 0) else 0
    vmax_total = np.percentile(total_map[total_map != 0], 97.5) if np.any(total_map != 0) else 1
    im0 = axes[0].imshow(total_map, cmap='RdBu_r', vmin=vmin_total, vmax=vmax_total)
    axes[0].set_title('Total Influence')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    # Excitatory influence - use percentile scaling
    exc_map = influence_data['excitatory_influence']
    vmax_exc = np.percentile(exc_map[exc_map > 0], 97.5) if np.any(exc_map > 0) else 1
    im1 = axes[1].imshow(exc_map, cmap='Reds', vmin=0, vmax=vmax_exc)
    axes[1].set_title('Excitatory (Facilitatory)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Inhibitory influence - use percentile scaling
    inh_map = influence_data['inhibitory_influence']
    vmax_inh = np.percentile(inh_map[inh_map > 0], 97.5) if np.any(inh_map > 0) else 1
    im2 = axes[2].imshow(inh_map, cmap='Blues', vmin=0, vmax=vmax_inh)
    axes[2].set_title('Inhibitory (Suppressive)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved influence map to {save_path}")

    return fig


def visualize_optimization(
    optimization_data: Dict[str, torch.Tensor],
    save_path: Optional[Path] = None
):
    """Visualize optimization results with separate E/I adjustments.

    Args:
        optimization_data: Output from optimize_circuit_response()
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_loss = fig.add_subplot(gs[1, :])

    # Get adjustments (already zeroed in masked regions)
    adjustment_exc = optimization_data['adjustment_exc'].cpu().numpy()[0]
    adjustment_inh = optimization_data['adjustment_inh'].cpu().numpy()[0]

    # Average over channels
    adj_exc_mean = np.mean(adjustment_exc, axis=0)
    adj_inh_mean = np.mean(adjustment_inh, axis=0)
    adj_diff_mean = adj_exc_mean - adj_inh_mean  # E - I

    # Excitatory adjustment (red colormap)
    non_zero_exc = adj_exc_mean[adj_exc_mean != 0]
    if len(non_zero_exc) > 0:
        vmax_exc = np.percentile(np.abs(non_zero_exc), 97.5)
    else:
        vmax_exc = 0.1

    im0 = axes[0].imshow(adj_exc_mean, cmap='Reds', vmin=0, vmax=vmax_exc, interpolation='nearest')
    axes[0].set_title('Excitatory Adjustment')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    # Inhibitory adjustment (blue colormap)
    non_zero_inh = adj_inh_mean[adj_inh_mean != 0]
    if len(non_zero_inh) > 0:
        vmax_inh = np.percentile(np.abs(non_zero_inh), 97.5)
    else:
        vmax_inh = 0.1

    im1 = axes[1].imshow(adj_inh_mean, cmap='Blues', vmin=0, vmax=vmax_inh, interpolation='nearest')
    axes[1].set_title('Inhibitory Adjustment')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # E - I difference (red-white-blue diverging colormap)
    non_zero_diff = adj_diff_mean[adj_diff_mean != 0]
    if len(non_zero_diff) > 0:
        vmin_diff = np.percentile(non_zero_diff, 2.5)
        vmax_diff = np.percentile(non_zero_diff, 97.5)
        # Make symmetric around zero
        vmax_abs = max(abs(vmin_diff), abs(vmax_diff))
        vmin_diff, vmax_diff = -vmax_abs, vmax_abs
    else:
        vmin_diff, vmax_diff = -0.1, 0.1

    im2 = axes[2].imshow(adj_diff_mean, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, interpolation='nearest')
    axes[2].set_title('E - I Balance')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    # Loss curve
    ax_loss.plot(optimization_data['loss_curve'], linewidth=2)
    ax_loss.set_xlabel('Optimization Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Optimization Convergence')
    ax_loss.grid(True, alpha=0.3)

    plt.suptitle('Optogenetic Perturbation Optimization: Learned E/I Adjustments',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved optimization results to {save_path}")

    return fig


def visualize_recurrent_flow(
    flow_data: Dict[str, np.ndarray],
    save_path: Optional[Path] = None
):
    """Visualize recurrent E/I flow to perturbed center.

    Args:
        flow_data: Output from measure_recurrent_contributions()
        save_path: Optional path to save figure
    """
    # Extract data
    exc_baseline = flow_data['exc_recurrent_baseline']  # [T, B, C]
    inh_baseline = flow_data['inh_recurrent_baseline']
    exc_perturbed = flow_data['exc_recurrent_perturbed']
    inh_perturbed = flow_data['inh_recurrent_perturbed']

    exc_spatial_baseline = flow_data['exc_spatial_baseline']  # [T, C, H, W]
    inh_spatial_baseline = flow_data['inh_spatial_baseline']
    exc_spatial_perturbed = flow_data['exc_spatial_perturbed']
    inh_spatial_perturbed = flow_data['inh_spatial_perturbed']

    perturb_loc = flow_data['perturb_location']
    perturb_factor = flow_data['perturb_factor']

    num_timesteps = exc_baseline.shape[0]

    # Average across channels and batch
    exc_baseline_mean = exc_baseline.mean(axis=(1, 2))  # [T]
    inh_baseline_mean = inh_baseline.mean(axis=(1, 2))
    exc_perturbed_mean = exc_perturbed.mean(axis=(1, 2))
    inh_perturbed_mean = inh_perturbed.mean(axis=(1, 2))

    # Compute change due to perturbation
    exc_change = exc_perturbed_mean - exc_baseline_mean
    inh_change = inh_perturbed_mean - inh_baseline_mean

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Time courses
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])

    # Excitatory flow over time
    ax1.plot(exc_baseline_mean, label='Baseline', linewidth=2, color='red', alpha=0.7)
    ax1.plot(exc_perturbed_mean, label=f'Perturbed ({perturb_factor})', linewidth=2, color='darkred')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Excitatory Recurrent Input')
    ax1.set_title('Excitatory Flow to Center')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Inhibitory flow over time
    ax2.plot(inh_baseline_mean, label='Baseline', linewidth=2, color='blue', alpha=0.7)
    ax2.plot(inh_perturbed_mean, label=f'Perturbed ({perturb_factor})', linewidth=2, color='darkblue')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Inhibitory Recurrent Input')
    ax2.set_title('Inhibitory Flow to Center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Change in E/I flow
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(exc_change, label='Exc change', linewidth=2, color='red')
    ax3.plot(inh_change, label='Inh change', linewidth=2, color='blue')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Change in Recurrent Input')
    ax3.set_title('Rescue Response: Change in E/I Flow Due to Perturbation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: E-I balance
    ax4 = fig.add_subplot(gs[1, 2:])
    ei_baseline = exc_baseline_mean - inh_baseline_mean
    ei_perturbed = exc_perturbed_mean - inh_perturbed_mean
    ax4.plot(ei_baseline, label='Baseline E-I', linewidth=2, color='purple', alpha=0.7)
    ax4.plot(ei_perturbed, label='Perturbed E-I', linewidth=2, color='darkviolet')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('E - I Balance')
    ax4.set_title('Net E/I Balance at Center')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 3: Spatial maps at final timestep
    final_t = -1

    # Average across channels for visualization
    exc_spatial_final_baseline = exc_spatial_baseline[final_t].mean(axis=0)  # [H, W]
    inh_spatial_final_baseline = inh_spatial_baseline[final_t].mean(axis=0)
    exc_spatial_final_perturbed = exc_spatial_perturbed[final_t].mean(axis=0)
    inh_spatial_final_perturbed = inh_spatial_perturbed[final_t].mean(axis=0)

    # Compute change
    exc_change_spatial = exc_spatial_final_perturbed - exc_spatial_final_baseline
    inh_change_spatial = inh_spatial_final_perturbed - inh_spatial_final_baseline

    ax5 = fig.add_subplot(gs[2, 0])
    im5 = ax5.imshow(exc_change_spatial, cmap='Reds', interpolation='nearest')
    ax5.plot(perturb_loc[1], perturb_loc[0], 'kx', markersize=15, markeredgewidth=3)
    ax5.set_title(f'Exc Change (t={final_t})')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[2, 1])
    im6 = ax6.imshow(inh_change_spatial, cmap='Blues', interpolation='nearest')
    ax6.plot(perturb_loc[1], perturb_loc[0], 'kx', markersize=15, markeredgewidth=3)
    ax6.set_title(f'Inh Change (t={final_t})')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[2, 2:])
    ei_change_spatial = exc_change_spatial - inh_change_spatial
    vmax = np.percentile(np.abs(ei_change_spatial), 98)
    im7 = ax7.imshow(ei_change_spatial, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax7.plot(perturb_loc[1], perturb_loc[0], 'kx', markersize=15, markeredgewidth=3)
    ax7.set_title(f'E-I Balance Change (t={final_t})')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    plt.suptitle(f'Recurrent Flow Analysis: {perturb_factor} FF Perturbation',
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved recurrent flow visualization to {save_path}")

    return fig


def visualize_multi_orientation_optimization(
    multi_ori_data: Dict,
    save_path: Optional[Path] = None,
    show_individual: bool = True,
    aggregation: str = 'mean'
):
    """Visualize multi-orientation optimization results.

    Shows averaged (orientation-invariant) adjustments plus individual orientations.

    Args:
        multi_ori_data: Output from optimize_circuit_response_multi_orientation()
        save_path: Optional path to save figure
        show_individual: If True, show individual orientations + rotated (default: True)
        aggregation: How to aggregate across channels: 'mean', 'max', or 'both'
    """
    # Extract averaged adjustments
    adj_exc_avg = multi_ori_data['adjustment_exc_avg'].cpu().numpy()[0]  # [C, H, W]
    adj_inh_avg = multi_ori_data['adjustment_inh_avg'].cpu().numpy()[0]

    # Aggregate over channels based on method
    if aggregation == 'mean':
        adj_exc_avg_spatial = np.mean(adj_exc_avg, axis=0)  # [H, W]
        adj_inh_avg_spatial = np.mean(adj_inh_avg, axis=0)
    elif aggregation == 'max':
        adj_exc_avg_spatial = np.max(adj_exc_avg, axis=0)  # [H, W]
        adj_inh_avg_spatial = np.max(adj_inh_avg, axis=0)
    else:
        raise ValueError(f"aggregation must be 'mean' or 'max', got {aggregation}")

    adj_diff_avg_spatial = adj_exc_avg_spatial - adj_inh_avg_spatial

    orientations = multi_ori_data['orientations']
    loss_curves = multi_ori_data['loss_curves']

    if show_individual:
        # Create figure with averaged + individual orientations
        num_ori = len(orientations)
        fig = plt.figure(figsize=(20, 4 + 4 * ((num_ori + 2) // 3)))

        # Top row: Averaged results
        gs = fig.add_gridspec(1 + (num_ori + 2) // 3, 3, hspace=0.4, wspace=0.3)

        axes_avg = [fig.add_subplot(gs[0, i]) for i in range(3)]

        # Plot averaged E, I, E-I
        non_zero_exc = adj_exc_avg_spatial[adj_exc_avg_spatial != 0]
        vmax_exc = np.percentile(np.abs(non_zero_exc), 97.5) if len(non_zero_exc) > 0 else 0.1

        im0 = axes_avg[0].imshow(adj_exc_avg_spatial, cmap='Reds', vmin=0, vmax=vmax_exc, interpolation='nearest')
        axes_avg[0].set_title(f'Averaged Excitatory ({aggregation.capitalize()})\n(Orientation-Invariant)', fontweight='bold')
        axes_avg[0].axis('off')
        plt.colorbar(im0, ax=axes_avg[0], fraction=0.046)

        non_zero_inh = adj_inh_avg_spatial[adj_inh_avg_spatial != 0]
        vmax_inh = np.percentile(np.abs(non_zero_inh), 97.5) if len(non_zero_inh) > 0 else 0.1

        im1 = axes_avg[1].imshow(adj_inh_avg_spatial, cmap='Blues', vmin=0, vmax=vmax_inh, interpolation='nearest')
        axes_avg[1].set_title(f'Averaged Inhibitory ({aggregation.capitalize()})\n(Orientation-Invariant)', fontweight='bold')
        axes_avg[1].axis('off')
        plt.colorbar(im1, ax=axes_avg[1], fraction=0.046)

        non_zero_diff = adj_diff_avg_spatial[adj_diff_avg_spatial != 0]
        if len(non_zero_diff) > 0:
            vmin_diff = np.percentile(non_zero_diff, 2.5)
            vmax_diff = np.percentile(non_zero_diff, 97.5)
            vmax_abs = max(abs(vmin_diff), abs(vmax_diff))
            vmin_diff, vmax_diff = -vmax_abs, vmax_abs
        else:
            vmin_diff, vmax_diff = -0.1, 0.1

        im2 = axes_avg[2].imshow(adj_diff_avg_spatial, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, interpolation='nearest')
        axes_avg[2].set_title(f'Averaged E-I Balance ({aggregation.capitalize()})\n(Orientation-Invariant)', fontweight='bold')
        axes_avg[2].axis('off')
        plt.colorbar(im2, ax=axes_avg[2], fraction=0.046)

        # Bottom rows: Individual orientations (rotated to 0°)
        adjustments_exc_rotated = multi_ori_data['adjustment_exc_rotated']
        adjustments_inh_rotated = multi_ori_data['adjustment_inh_rotated']

        for idx, ori in enumerate(orientations):
            row = 1 + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            # Get rotated adjustments for this orientation
            adj_exc_rot = adjustments_exc_rotated[idx].mean(axis=0)  # Average over channels
            adj_inh_rot = adjustments_inh_rotated[idx].mean(axis=0)
            adj_diff_rot = adj_exc_rot - adj_inh_rot

            im = ax.imshow(adj_diff_rot, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, interpolation='nearest')
            ax.set_title(f'{ori}° (rotated to 0°)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'Multi-Orientation Optimization ({len(orientations)} orientations)\nAveraged = Orientation-Invariant Recurrent RF Structure',
                     fontsize=14, fontweight='bold')

    else:
        # Simpler figure: just averaged results + loss curves
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_loss = fig.add_subplot(gs[1, :])

        # Plot averaged E, I, E-I
        non_zero_exc = adj_exc_avg_spatial[adj_exc_avg_spatial != 0]
        vmax_exc = np.percentile(np.abs(non_zero_exc), 97.5) if len(non_zero_exc) > 0 else 0.1

        im0 = axes[0].imshow(adj_exc_avg_spatial, cmap='Reds', vmin=0, vmax=vmax_exc, interpolation='nearest')
        axes[0].set_title(f'Averaged Excitatory ({aggregation.capitalize()})')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])

        non_zero_inh = adj_inh_avg_spatial[adj_inh_avg_spatial != 0]
        vmax_inh = np.percentile(np.abs(non_zero_inh), 97.5) if len(non_zero_inh) > 0 else 0.1

        im1 = axes[1].imshow(adj_inh_avg_spatial, cmap='Blues', vmin=0, vmax=vmax_inh, interpolation='nearest')
        axes[1].set_title(f'Averaged Inhibitory ({aggregation.capitalize()})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        non_zero_diff = adj_diff_avg_spatial[adj_diff_avg_spatial != 0]
        if len(non_zero_diff) > 0:
            vmin_diff = np.percentile(non_zero_diff, 2.5)
            vmax_diff = np.percentile(non_zero_diff, 97.5)
            vmax_abs = max(abs(vmin_diff), abs(vmax_diff))
            vmin_diff, vmax_diff = -vmax_abs, vmax_abs
        else:
            vmin_diff, vmax_diff = -0.1, 0.1

        im2 = axes[2].imshow(adj_diff_avg_spatial, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, interpolation='nearest')
        axes[2].set_title(f'Averaged E-I Balance ({aggregation.capitalize()})')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])

        # Loss curves for each orientation
        for idx, (ori, loss_curve) in enumerate(zip(orientations, loss_curves)):
            ax_loss.plot(loss_curve, label=f'{ori}°', alpha=0.7, linewidth=1.5)

        ax_loss.set_xlabel('Optimization Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Optimization Convergence (All Orientations)')
        ax_loss.legend(ncol=len(orientations)//2, fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        plt.suptitle(f'Multi-Orientation Optimization ({aggregation.capitalize()} aggregation)\n({len(orientations)} orientations averaged)',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-orientation optimization results to {save_path}")

    return fig
