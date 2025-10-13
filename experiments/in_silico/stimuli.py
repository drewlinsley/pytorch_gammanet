"""Stimulus generation for in silico neurophysiology experiments.

This module provides classes to generate visual stimuli used in classic
neurophysiology experiments including Kapadia, Kinoshita, Trott & Born.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy import ndimage


@dataclass
class StimulusMetadata:
    """Metadata for generated stimuli."""
    stimulus_type: str
    orientation: float
    contrast: float
    spatial_frequency: float
    position: Tuple[int, int]
    size: Tuple[int, int]
    parameters: Dict


class BaseStimulus:
    """Base class for stimulus generation."""
    
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        self.size = size
        self.center = (size[0] // 2, size[1] // 2)
        
    def create_grating(self, 
                      orientation: float,
                      spatial_frequency: float,
                      phase: float = 0,
                      contrast: float = 1.0) -> np.ndarray:
        """Create oriented sinusoidal grating.
        
        Args:
            orientation: Orientation in degrees (0-180)
            spatial_frequency: Cycles per image
            phase: Phase offset in radians
            contrast: Contrast (0-1)
            
        Returns:
            Grating array normalized to [-1, 1]
        """
        # Create coordinate grids
        y, x = np.mgrid[0:self.size[0], 0:self.size[1]]
        x = x - self.center[1]
        y = y - self.center[0]
        
        # Convert orientation to radians
        theta = np.radians(orientation)
        
        # Rotate coordinates
        xr = x * np.cos(theta) + y * np.sin(theta)
        
        # Generate sinusoidal grating
        wavelength = self.size[0] / spatial_frequency
        grating = np.sin(2 * np.pi * xr / wavelength + phase)
        
        # Apply contrast
        grating = grating * contrast
        
        return grating
    
    def create_gabor(self,
                    orientation: float,
                    spatial_frequency: float,
                    sigma: float,
                    phase: float = 0,
                    contrast: float = 1.0) -> np.ndarray:
        """Create Gabor patch.
        
        Args:
            orientation: Orientation in degrees
            spatial_frequency: Cycles per image
            sigma: Standard deviation of Gaussian envelope
            phase: Phase offset
            contrast: Contrast (0-1)
            
        Returns:
            Gabor patch array
        """
        # Create grating
        grating = self.create_grating(orientation, spatial_frequency, phase, contrast)
        
        # Create Gaussian envelope
        y, x = np.mgrid[0:self.size[0], 0:self.size[1]]
        x = x - self.center[1]
        y = y - self.center[0]
        
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Combine grating and envelope
        gabor = grating * gaussian
        
        return gabor
    
    def create_circular_mask(self, radius: int) -> np.ndarray:
        """Create circular mask."""
        y, x = np.ogrid[:self.size[0], :self.size[1]]
        dist_from_center = np.sqrt((x - self.center[1])**2 + (y - self.center[0])**2)
        mask = dist_from_center <= radius
        return mask.astype(np.float32)
    
    def create_annular_mask(self, inner_radius: int, outer_radius: int) -> np.ndarray:
        """Create annular (ring) mask."""
        inner_mask = self.create_circular_mask(inner_radius)
        outer_mask = self.create_circular_mask(outer_radius)
        return outer_mask - inner_mask


class OrientedGratingStimuli(BaseStimulus):
    """Generate oriented grating stimuli for basic orientation tuning."""
    
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        
    def generate_stimulus_set(self,
                            orientations: List[float],
                            spatial_frequencies: List[float] = [10.0, 20.0, 30.0, 40.0],
                            contrasts: List[float] = [1.0],
                            stimulus_diameter: Optional[int] = None) -> List[Tuple[np.ndarray, StimulusMetadata]]:
        """Generate set of oriented gratings.

        Args:
            orientations: List of orientations in degrees
            spatial_frequencies: List of spatial frequencies
            contrasts: List of contrasts
            stimulus_diameter: Diameter of circular aperture in pixels (at input resolution).
                             If None, uses full image (no masking).

        Returns:
            List of (stimulus, metadata) tuples
        """
        stimuli = []

        # Create mask if diameter specified
        if stimulus_diameter is not None:
            radius = stimulus_diameter // 2
            mask = self.create_circular_mask(radius)
        else:
            mask = np.ones(self.size, dtype=np.float32)
            radius = min(self.size) // 2  # For metadata

        for ori in orientations:
            for sf in spatial_frequencies:
                for contrast in contrasts:
                    # Create grating (already has contrast applied, in [-contrast, +contrast] range)
                    grating = self.create_grating(ori, sf, contrast=contrast)

                    # Apply mask and normalize to [0, 1] with mean=0.5 (grey background)
                    # grating is in [-contrast, +contrast], multiply by 0.5 to get [-0.5*contrast, +0.5*contrast]
                    # then add 0.5 to center at grey: [0.5 - 0.5*contrast, 0.5 + 0.5*contrast]
                    # Outside mask: set to 0.5 (grey background)
                    stimulus = grating * mask * 0.5 + 0.5

                    # Create metadata
                    metadata = StimulusMetadata(
                        stimulus_type="oriented_grating",
                        orientation=ori,
                        contrast=contrast,
                        spatial_frequency=sf,
                        position=self.center,
                        size=(radius*2, radius*2) if stimulus_diameter else self.size,
                        parameters={
                            "aperture_diameter": stimulus_diameter,
                            "aperture_radius": radius
                        }
                    )

                    stimuli.append((stimulus, metadata))

        return stimuli


class KapadiaStimuli(BaseStimulus):
    """Generate stimuli for Kapadia et al. (1995) collinear facilitation experiments."""

    def __init__(self, size: Tuple[int, int] = (256, 256),
                 bar_length: int = 20, bar_width: int = 5):
        super().__init__(size)
        self.bar_length = bar_length  # pixels (default 20)
        self.bar_width = bar_width    # pixels (default 5)
        
    def create_bar(self, orientation: float, length: int, width: int) -> np.ndarray:
        """Create oriented bar stimulus."""
        # Create horizontal bar
        bar = np.zeros((width*3, length*3))
        bar[width:width*2, length:length*2] = 1.0
        
        # Rotate to desired orientation
        bar = ndimage.rotate(bar, -orientation, reshape=True, order=1)
        
        # Threshold to clean up
        bar = (bar > 0.5).astype(np.float32)
        
        return bar
    
    def generate_stimulus_set(self,
                            orientations: List[float] = [0, 45, 90, 135],
                            flanker_distances: List[int] = [15, 20, 25, 30, 35, 40],
                            flanker_angles: List[float] = [0, 15, 30, 45, 60, 75, 90],
                            contrasts: List[float] = [0.2, 0.6],
                            center_only: bool = False) -> List[Tuple[np.ndarray, StimulusMetadata]]:
        """Generate Kapadia-style collinear facilitation stimuli.

        Args:
            orientations: Target orientations
            flanker_distances: Distances between target and flankers (pixels)
            flanker_angles: Relative angles of flankers
            contrasts: [target_contrast, flanker_contrast]
            center_only: If True, only generate center bar without flankers

        Returns:
            List of (stimulus, metadata) tuples
        """
        stimuli = []
        target_contrast, flanker_contrast = contrasts

        for ori in orientations:
            # Create target bar
            target = self.create_bar(ori, self.bar_length, self.bar_width)

            if center_only:
                # Generate only center bar without flankers
                stimulus = np.zeros(self.size)

                # Place target at center
                target_y = self.center[0] - target.shape[0] // 2
                target_x = self.center[1] - target.shape[1] // 2

                # Add target with contrast
                stimulus[target_y:target_y+target.shape[0],
                       target_x:target_x+target.shape[1]] = target * target_contrast

                # Create metadata
                metadata = StimulusMetadata(
                    stimulus_type="kapadia_center_only",
                    orientation=ori,
                    contrast=target_contrast,
                    spatial_frequency=None,
                    position=self.center,
                    size=(self.bar_length, self.bar_width),
                    parameters={
                        "flanker_distance": 0,
                        "flanker_angle": 0,
                        "flanker_contrast": 0,
                        "bar_length": self.bar_length,
                        "bar_width": self.bar_width
                    }
                )

                stimuli.append((stimulus, metadata))
            else:
                # Generate stimuli with flankers
                for distance in flanker_distances:
                    for angle in flanker_angles:
                        # Create blank canvas
                        stimulus = np.zeros(self.size)

                        # Place target at center
                        target_y = self.center[0] - target.shape[0] // 2
                        target_x = self.center[1] - target.shape[1] // 2

                        # Add target with contrast
                        stimulus[target_y:target_y+target.shape[0],
                               target_x:target_x+target.shape[1]] = target * target_contrast

                        # Calculate flanker positions
                        # Flankers are placed along the axis of the target orientation
                        dx = np.cos(np.radians(ori)) * distance
                        dy = np.sin(np.radians(ori)) * distance

                        # Create flankers with relative angle
                        flanker = self.create_bar(ori + angle, self.bar_length, self.bar_width)

                        # Place flankers
                        for sign in [-1, 1]:  # Both sides
                            flanker_y = int(self.center[0] + sign * dy - flanker.shape[0] // 2)
                            flanker_x = int(self.center[1] + sign * dx - flanker.shape[1] // 2)

                            # Check bounds
                            if (0 <= flanker_y < self.size[0] - flanker.shape[0] and
                                0 <= flanker_x < self.size[1] - flanker.shape[1]):
                                stimulus[flanker_y:flanker_y+flanker.shape[0],
                                       flanker_x:flanker_x+flanker.shape[1]] = flanker * flanker_contrast

                        # Create metadata
                        metadata = StimulusMetadata(
                            stimulus_type="kapadia_collinear",
                            orientation=ori,
                            contrast=target_contrast,
                            spatial_frequency=1.0 / self.bar_length,
                            position=self.center,
                            size=(self.bar_length, self.bar_width),
                            parameters={
                                "flanker_distance": distance,
                                "flanker_angle": angle,
                                "flanker_contrast": flanker_contrast
                            }
                        )

                        stimuli.append((stimulus, metadata))

        return stimuli


class KinoshitaStimuli(BaseStimulus):
    """Generate stimuli for Kinoshita & Gilbert (2008) surround modulation experiments."""
    
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        
    def generate_stimulus_set(self,
                            center_orientations: List[float] = [0, 45, 90, 135],
                            surround_orientations: List[float] = None,
                            center_radius: int = 25,
                            surround_inner_radius: int = 30,
                            surround_outer_radius: int = 80,
                            spatial_frequency: float = 4.0,
                            contrasts: List[float] = [0.5, 0.5]) -> List[Tuple[np.ndarray, StimulusMetadata]]:
        """Generate center-surround stimuli.
        
        Args:
            center_orientations: Orientations for center
            surround_orientations: Orientations for surround (if None, uses relative angles)
            center_radius: Radius of center region
            surround_inner_radius: Inner radius of surround
            surround_outer_radius: Outer radius of surround
            spatial_frequency: Spatial frequency of gratings
            contrasts: [center_contrast, surround_contrast]
            
        Returns:
            List of (stimulus, metadata) tuples
        """
        stimuli = []
        center_contrast, surround_contrast = contrasts
        
        if surround_orientations is None:
            # Use relative orientations
            relative_angles = [0, 30, 60, 90]  # Relative to center
        
        # Create masks
        center_mask = self.create_circular_mask(center_radius)
        surround_mask = self.create_annular_mask(surround_inner_radius, surround_outer_radius)
        
        for center_ori in center_orientations:
            # Center-only condition
            center_grating = self.create_grating(center_ori, spatial_frequency, contrast=center_contrast)
            stimulus = center_grating * center_mask
            
            metadata = StimulusMetadata(
                stimulus_type="kinoshita_center_only",
                orientation=center_ori,
                contrast=center_contrast,
                spatial_frequency=spatial_frequency,
                position=self.center,
                size=(center_radius*2, center_radius*2),
                parameters={"condition": "center_only"}
            )
            
            stimuli.append((stimulus, metadata))
            
            # Center-surround conditions
            if surround_orientations is None:
                surround_oris = [center_ori + angle for angle in relative_angles]
            else:
                surround_oris = surround_orientations
                
            for surround_ori in surround_oris:
                # Create surround grating
                surround_grating = self.create_grating(surround_ori, spatial_frequency, contrast=surround_contrast)
                
                # Combine center and surround
                stimulus = (center_grating * center_mask + 
                          surround_grating * surround_mask)
                
                # Create metadata
                metadata = StimulusMetadata(
                    stimulus_type="kinoshita_center_surround",
                    orientation=center_ori,
                    contrast=center_contrast,
                    spatial_frequency=spatial_frequency,
                    position=self.center,
                    size=(center_radius*2, center_radius*2),
                    parameters={
                        "surround_orientation": surround_ori,
                        "surround_contrast": surround_contrast,
                        "orientation_difference": surround_ori - center_ori,
                        "center_radius": center_radius,
                        "surround_inner_radius": surround_inner_radius,
                        "surround_outer_radius": surround_outer_radius
                    }
                )
                
                stimuli.append((stimulus, metadata))
                
        return stimuli


class TextureBoundaryStimuli(BaseStimulus):
    """Generate texture boundary stimuli similar to Trott & Born experiments."""
    
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        
    def create_texture_patch(self, 
                           orientation: float,
                           spatial_frequency: float,
                           patch_size: int,
                           jitter: float = 0) -> np.ndarray:
        """Create texture patch with oriented elements."""
        # For simplicity, using gratings as texture elements
        # In full implementation, would use line segments or Gabors
        grating = self.create_grating(orientation, spatial_frequency)
        
        # Add orientation jitter if specified
        if jitter > 0:
            angle_offset = np.random.uniform(-jitter, jitter)
            grating = ndimage.rotate(grating, angle_offset, reshape=False)
            
        return grating
    
    def generate_stimulus_set(self,
                            orientations: List[float] = [0, 45, 90, 135],
                            orientation_differences: List[float] = [0, 30, 60, 90],
                            boundary_positions: List[str] = ["vertical", "horizontal"],
                            spatial_frequency: float = 8.0) -> List[Tuple[np.ndarray, StimulusMetadata]]:
        """Generate texture boundary stimuli.
        
        Args:
            orientations: Base orientations
            orientation_differences: Differences across boundary
            boundary_positions: Boundary orientations
            spatial_frequency: Texture element frequency
            
        Returns:
            List of (stimulus, metadata) tuples
        """
        stimuli = []
        
        for base_ori in orientations:
            for ori_diff in orientation_differences:
                for boundary_pos in boundary_positions:
                    # Create blank canvas
                    stimulus = np.zeros(self.size)
                    
                    if boundary_pos == "vertical":
                        # Left half
                        left_texture = self.create_texture_patch(base_ori, spatial_frequency, self.size[1]//2)
                        stimulus[:, :self.size[1]//2] = left_texture[:, :self.size[1]//2]
                        
                        # Right half
                        right_texture = self.create_texture_patch(base_ori + ori_diff, spatial_frequency, self.size[1]//2)
                        stimulus[:, self.size[1]//2:] = right_texture[:, self.size[1]//2:]
                        
                    else:  # horizontal
                        # Top half
                        top_texture = self.create_texture_patch(base_ori, spatial_frequency, self.size[0]//2)
                        stimulus[:self.size[0]//2, :] = top_texture[:self.size[0]//2, :]
                        
                        # Bottom half
                        bottom_texture = self.create_texture_patch(base_ori + ori_diff, spatial_frequency, self.size[0]//2)
                        stimulus[self.size[0]//2:, :] = bottom_texture[self.size[0]//2:, :]
                    
                    # Create metadata
                    metadata = StimulusMetadata(
                        stimulus_type="texture_boundary",
                        orientation=base_ori,
                        contrast=1.0,
                        spatial_frequency=spatial_frequency,
                        position=self.center,
                        size=self.size,
                        parameters={
                            "orientation_difference": ori_diff,
                            "boundary_position": boundary_pos,
                            "texture_orientation_1": base_ori,
                            "texture_orientation_2": base_ori + ori_diff
                        }
                    )
                    
                    stimuli.append((stimulus, metadata))
                    
        return stimuli


class TiltIllusionStimuli(BaseStimulus):
    """Generate tilt illusion stimuli."""
    
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        
    def generate_stimulus_set(self,
                            center_orientations: List[float] = [45, 90, 135],
                            surround_tilts: List[float] = [-30, -15, 0, 15, 30],
                            center_radius: int = 20,
                            surround_radius: int = 60,
                            spatial_frequency: float = 4.0) -> List[Tuple[np.ndarray, StimulusMetadata]]:
        """Generate tilt illusion stimuli.
        
        Args:
            center_orientations: Center grating orientations
            surround_tilts: Surround tilts relative to center
            center_radius: Center region radius
            surround_radius: Surround outer radius
            spatial_frequency: Grating frequency
            
        Returns:
            List of (stimulus, metadata) tuples
        """
        stimuli = []
        
        # Create masks
        center_mask = self.create_circular_mask(center_radius)
        surround_mask = self.create_annular_mask(center_radius + 5, surround_radius)
        
        for center_ori in center_orientations:
            for tilt in surround_tilts:
                # Create gratings
                center_grating = self.create_grating(center_ori, spatial_frequency)
                surround_grating = self.create_grating(center_ori + tilt, spatial_frequency)
                
                # Combine
                stimulus = (center_grating * center_mask + 
                          surround_grating * surround_mask)
                
                # Create metadata
                metadata = StimulusMetadata(
                    stimulus_type="tilt_illusion",
                    orientation=center_ori,
                    contrast=1.0,
                    spatial_frequency=spatial_frequency,
                    position=self.center,
                    size=(center_radius*2, center_radius*2),
                    parameters={
                        "surround_tilt": tilt,
                        "surround_orientation": center_ori + tilt,
                        "center_radius": center_radius,
                        "surround_radius": surround_radius
                    }
                )
                
                stimuli.append((stimulus, metadata))
                
        return stimuli


def create_stimulus_batch(stimuli: List[Tuple[np.ndarray, StimulusMetadata]], 
                         normalize: bool = True) -> Tuple[torch.Tensor, List[StimulusMetadata]]:
    """Convert list of stimuli to batch tensor.
    
    Args:
        stimuli: List of (stimulus, metadata) tuples
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Batch tensor and list of metadata
    """
    arrays = []
    metadata = []
    
    for stim, meta in stimuli:
        if normalize:
            # Normalize to [-1, 1]
            stim = (stim - stim.mean()) / (stim.std() + 1e-8)
            stim = np.clip(stim, -3, 3) / 3  # Clip outliers
            
        arrays.append(stim)
        metadata.append(meta)
    
    # Stack into batch
    batch = np.stack(arrays, axis=0)
    
    # Convert to torch tensor and add channel dimension
    batch = torch.from_numpy(batch).float().unsqueeze(1)
    
    # Repeat to 3 channels if needed (for compatibility with RGB models)
    batch = batch.repeat(1, 3, 1, 1)
    
    return batch, metadata