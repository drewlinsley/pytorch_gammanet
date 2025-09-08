"""VGG16-GammaNet v2 with E/I states and improved fGRU modules.

This implementation supports both the original fGRU and the new fGRUv2 with
separate excitatory/inhibitory states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torchvision.models as models

from .components import fGRUv2
from .components.normalization import InstanceNorm2d
from .components.alignment import DistributionAlignment


class VGG16Block(nn.Module):
    """A block of VGG16 layers."""
    
    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.layers = layers
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VGG16GammaNetV2(nn.Module):
    """VGG16 with integrated fGRUv2 modules supporting E/I states.
    
    Architecture follows the original implementation:
    - Input → conv1_1 → conv1_2 → fGRU_block1 → pool1
    - → conv2_1 → conv2_2 → fGRU_0 → pool2  
    - → conv3_1 → conv3_2 → conv3_3 → fGRU_1 → pool3
    - → conv4_1 → conv4_2 → conv4_3 → fGRU_2 → pool4
    - → conv5_1 → conv5_2 → conv5_3 → fGRU_3
    
    Then top-down pass modulates encoder states back up to full resolution.
    Output is from td_h_block1 at full input resolution (no upsampling needed).
    """
    
    def __init__(
        self,
        config: Dict,
        input_channels: int = 3,
        output_channels: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.timesteps = config.get('timesteps', 8)
        self.fgru_config = config.get('fgru', {})
        self.skip_connections = config.get('skip_connections', True)
        self.use_distribution_alignment = config.get('use_distribution_alignment', False)
        self.use_separate_ei_states = self.fgru_config.get('use_separate_ei_states', False)
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Split VGG16 into blocks for fGRU insertion
        # Block 1: conv1_1, conv1_2 (indices 0-4, but we separate pool)
        self.block1_conv = VGG16Block(vgg16.features[0:4])  # Just conv layers
        self.pool1 = vgg16.features[4]  # MaxPool2d
        
        # Block 2: conv2_1, conv2_2 (indices 5-9)
        self.block2_conv = VGG16Block(vgg16.features[5:9])
        self.pool2 = vgg16.features[9]  # MaxPool2d
        
        # Block 3: conv3_1, conv3_2, conv3_3 (indices 10-16)
        self.block3_conv = VGG16Block(vgg16.features[10:16])
        self.pool3 = vgg16.features[16]  # MaxPool2d
        
        # Block 4: conv4_1, conv4_2, conv4_3 (indices 17-23)
        self.block4_conv = VGG16Block(vgg16.features[17:23])
        self.pool4 = vgg16.features[23]  # MaxPool2d
        
        # Block 5: conv5_1, conv5_2, conv5_3 (indices 24-30)
        self.block5_conv = VGG16Block(vgg16.features[24:30])
        
        # Create fGRU modules at insertion points
        # NEW: fGRU after Block 1 (64 channels)
        self.fgru_block1 = fGRUv2(
            hidden_channels=64,
            kernel_size=(3, 3),
            **self.fgru_config
        )
        
        # fGRU_0: after conv2_2 (128 channels)
        self.fgru_0 = fGRUv2(
            hidden_channels=128,
            kernel_size=(3, 3),
            **self.fgru_config
        )
        
        # fGRU_1: after conv3_3 (256 channels)
        self.fgru_1 = fGRUv2(
            hidden_channels=256,
            kernel_size=(3, 3),
            **self.fgru_config
        )
        
        # fGRU_2: after conv4_3 (512 channels)
        self.fgru_2 = fGRUv2(
            hidden_channels=512,
            kernel_size=(3, 3),
            **self.fgru_config
        )
        
        # fGRU_3: after conv5_3 (512 channels)
        self.fgru_3 = fGRUv2(
            hidden_channels=512,
            kernel_size=(3, 3),
            **self.fgru_config
        )
        
        # Top-down fGRUs for modulation
        self.td_fgru_3_to_2 = nn.Sequential(
            nn.Conv2d(512, 512, 1),  # Channel projection if needed
            InstanceNorm2d(512),
            nn.ELU(inplace=True)
        )
        self.td_fgru_2 = fGRUv2(hidden_channels=512, kernel_size=(1, 1), **self.fgru_config)
        
        self.td_fgru_2_to_1 = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # Project 512 → 256
            InstanceNorm2d(256),
            nn.ELU(inplace=True)
        )
        self.td_fgru_1 = fGRUv2(hidden_channels=256, kernel_size=(1, 1), **self.fgru_config)
        
        self.td_fgru_1_to_0 = nn.Sequential(
            nn.Conv2d(256, 128, 1),  # Project 256 → 128
            InstanceNorm2d(128),
            nn.ELU(inplace=True)
        )
        self.td_fgru_0 = fGRUv2(hidden_channels=128, kernel_size=(1, 1), **self.fgru_config)
        
        # NEW: Top-down to Block 1 level (full resolution)
        self.td_fgru_0_to_block1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # Project 128 → 64
            InstanceNorm2d(64),
            nn.ELU(inplace=True)
        )
        self.td_fgru_block1 = fGRUv2(hidden_channels=64, kernel_size=(1, 1), **self.fgru_config)
        
        # Distribution alignment layers (after each fGRU before passing to VGG/pooling)
        if self.use_distribution_alignment:
            self.align_block1 = DistributionAlignment(64)
            self.align_0 = DistributionAlignment(128)
            self.align_1 = DistributionAlignment(256)
            self.align_2 = DistributionAlignment(512)
            self.align_3 = DistributionAlignment(512)
        else:
            # Identity operations when not using alignment
            self.align_block1 = nn.Identity()
            self.align_0 = nn.Identity()
            self.align_1 = nn.Identity()
            self.align_2 = nn.Identity()
            self.align_3 = nn.Identity()
        
        # Direct output projection from full-resolution td_h_block1 features
        self.output_projection = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            InstanceNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, output_channels, 1)
        )
        
        # Initialize hidden states buffers
        # Excitatory states
        self.register_buffer('h_block1_exc', None)
        self.register_buffer('h0_exc', None)
        self.register_buffer('h1_exc', None)
        self.register_buffer('h2_exc', None)
        self.register_buffer('h3_exc', None)
        
        # Inhibitory states (if using separate E/I)
        if self.use_separate_ei_states:
            self.register_buffer('h_block1_inh', None)
            self.register_buffer('h0_inh', None)
            self.register_buffer('h1_inh', None)
            self.register_buffer('h2_inh', None)
            self.register_buffer('h3_inh', None)
        
        # Top-down excitatory hidden states
        self.register_buffer('td_h_block1_exc', None)
        self.register_buffer('td_h0_exc', None)
        self.register_buffer('td_h1_exc', None)
        self.register_buffer('td_h2_exc', None)
        
        # Top-down inhibitory states (if using separate E/I)
        if self.use_separate_ei_states:
            self.register_buffer('td_h_block1_inh', None)
            self.register_buffer('td_h0_inh', None)
            self.register_buffer('td_h1_inh', None)
            self.register_buffer('td_h2_inh', None)
        
    def init_hidden_states(self, batch_size: int, height: int, width: int, device: torch.device):
        """Initialize hidden states for all fGRU modules."""
        # Calculate feature map sizes at each level
        h0, w0 = height, width                # Full resolution: 320x320
        h1, w1 = height // 2, width // 2      # After pool1: 160x160 for 320x320 input
        h2, w2 = height // 4, width // 4      # After pool2: 80x80
        h3, w3 = height // 8, width // 8      # After pool3: 40x40
        h4, w4 = height // 16, width // 16    # After pool4: 20x20
        
        # Encoder excitatory hidden states
        self.h_block1_exc = torch.zeros(batch_size, 64, h0, w0).to(device)
        self.h0_exc = torch.zeros(batch_size, 128, h1, w1).to(device)
        self.h1_exc = torch.zeros(batch_size, 256, h2, w2).to(device)
        self.h2_exc = torch.zeros(batch_size, 512, h3, w3).to(device)
        self.h3_exc = torch.zeros(batch_size, 512, h4, w4).to(device)
        
        # Encoder inhibitory hidden states (if using separate E/I)
        if self.use_separate_ei_states:
            self.h_block1_inh = torch.zeros(batch_size, 64, h0, w0).to(device)
            self.h0_inh = torch.zeros(batch_size, 128, h1, w1).to(device)
            self.h1_inh = torch.zeros(batch_size, 256, h2, w2).to(device)
            self.h2_inh = torch.zeros(batch_size, 512, h3, w3).to(device)
            self.h3_inh = torch.zeros(batch_size, 512, h4, w4).to(device)
        
        # Top-down excitatory hidden states
        self.td_h_block1_exc = torch.zeros(batch_size, 64, h0, w0).to(device)
        self.td_h2_exc = torch.zeros(batch_size, 512, h3, w3).to(device)
        self.td_h1_exc = torch.zeros(batch_size, 256, h2, w2).to(device)
        self.td_h0_exc = torch.zeros(batch_size, 128, h1, w1).to(device)
        
        # Top-down inhibitory hidden states (if using separate E/I)
        if self.use_separate_ei_states:
            self.td_h_block1_inh = torch.zeros(batch_size, 64, h0, w0).to(device)
            self.td_h2_inh = torch.zeros(batch_size, 512, h3, w3).to(device)
            self.td_h1_inh = torch.zeros(batch_size, 256, h2, w2).to(device)
            self.td_h0_inh = torch.zeros(batch_size, 128, h1, w1).to(device)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep iterations.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Output tensor [B, 1, H, W]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize hidden states if needed
        if self.h_block1_exc is None or self.h_block1_exc.shape[0] != batch_size:
            self.init_hidden_states(batch_size, x.shape[2], x.shape[3], device)
            
        # Process through timesteps
        for t in range(self.timesteps):
            # === Bottom-up pass ===
            
            # Block 1: conv1 layers + fGRU
            x1 = self.block1_conv(x)
            # Apply fGRU after conv1_2
            if self.use_separate_ei_states:
                self.h_block1_exc, self.h_block1_inh, _ = self.fgru_block1(
                    x1, self.h_block1_exc, self.h_block1_inh
                )
                aligned_block1 = self.align_block1(self.h_block1_exc)
            else:
                self.h_block1_exc, _, _ = self.fgru_block1(x1, self.h_block1_exc)
                aligned_block1 = self.align_block1(self.h_block1_exc)
            x1 = self.pool1(aligned_block1)
            
            # Block 2: conv2 layers + fGRU_0
            x2 = self.block2_conv(x1)
            if self.use_separate_ei_states:
                self.h0_exc, self.h0_inh, _ = self.fgru_0(x2, self.h0_exc, self.h0_inh)
                aligned_0 = self.align_0(self.h0_exc)
            else:
                self.h0_exc, _, _ = self.fgru_0(x2, self.h0_exc)
                aligned_0 = self.align_0(self.h0_exc)
            x2 = self.pool2(aligned_0)
            
            # Block 3: conv3 layers + fGRU_1
            x3 = self.block3_conv(x2)
            if self.use_separate_ei_states:
                self.h1_exc, self.h1_inh, _ = self.fgru_1(x3, self.h1_exc, self.h1_inh)
                aligned_1 = self.align_1(self.h1_exc)
            else:
                self.h1_exc, _, _ = self.fgru_1(x3, self.h1_exc)
                aligned_1 = self.align_1(self.h1_exc)
            x3 = self.pool3(aligned_1)
            
            # Block 4: conv4 layers + fGRU_2
            x4 = self.block4_conv(x3)
            if self.use_separate_ei_states:
                self.h2_exc, self.h2_inh, _ = self.fgru_2(x4, self.h2_exc, self.h2_inh)
                aligned_2 = self.align_2(self.h2_exc)
            else:
                self.h2_exc, _, _ = self.fgru_2(x4, self.h2_exc)
                aligned_2 = self.align_2(self.h2_exc)
            x4 = self.pool4(aligned_2)
            
            # Block 5: conv5 layers + fGRU_3
            x5 = self.block5_conv(x4)
            if self.use_separate_ei_states:
                self.h3_exc, self.h3_inh, _ = self.fgru_3(x5, self.h3_exc, self.h3_inh)
            else:
                self.h3_exc, _, _ = self.fgru_3(x5, self.h3_exc)
            
            # === Top-down pass ===
            
            # TD: fGRU_3 → fGRU_2
            td_3_to_2 = self.td_fgru_3_to_2(self.h3_exc)
            td_3_to_2 = F.interpolate(td_3_to_2, size=self.h2_exc.shape[2:], 
                                      mode='bilinear', align_corners=False)
            if self.use_separate_ei_states:
                self.td_h2_exc, self.td_h2_inh, _ = self.td_fgru_2(
                    td_3_to_2, self.td_h2_exc, self.td_h2_inh
                )
            else:
                self.td_h2_exc, _, _ = self.td_fgru_2(td_3_to_2, self.td_h2_exc)
            
            if self.skip_connections:
                self.h2_exc = self.h2_exc + self.td_h2_exc
            else:
                self.h2_exc = self.td_h2_exc
                
            # TD: fGRU_2 → fGRU_1
            td_2_to_1 = self.td_fgru_2_to_1(self.h2_exc)
            td_2_to_1 = F.interpolate(td_2_to_1, size=self.h1_exc.shape[2:], 
                                      mode='bilinear', align_corners=False)
            if self.use_separate_ei_states:
                self.td_h1_exc, self.td_h1_inh, _ = self.td_fgru_1(
                    td_2_to_1, self.td_h1_exc, self.td_h1_inh
                )
            else:
                self.td_h1_exc, _, _ = self.td_fgru_1(td_2_to_1, self.td_h1_exc)
            
            if self.skip_connections:
                self.h1_exc = self.h1_exc + self.td_h1_exc
            else:
                self.h1_exc = self.td_h1_exc
                
            # TD: fGRU_1 → fGRU_0
            td_1_to_0 = self.td_fgru_1_to_0(self.h1_exc)
            td_1_to_0 = F.interpolate(td_1_to_0, size=self.h0_exc.shape[2:], 
                                      mode='bilinear', align_corners=False)
            if self.use_separate_ei_states:
                self.td_h0_exc, self.td_h0_inh, _ = self.td_fgru_0(
                    td_1_to_0, self.td_h0_exc, self.td_h0_inh
                )
            else:
                self.td_h0_exc, _, _ = self.td_fgru_0(td_1_to_0, self.td_h0_exc)
            
            if self.skip_connections:
                self.h0_exc = self.h0_exc + self.td_h0_exc
            else:
                self.h0_exc = self.td_h0_exc
                
            # TD: fGRU_0 → Block 1 (full resolution)
            td_0_to_block1 = self.td_fgru_0_to_block1(self.h0_exc)
            td_0_to_block1 = F.interpolate(td_0_to_block1, size=self.h_block1_exc.shape[2:], 
                                          mode='bilinear', align_corners=False)
            if self.use_separate_ei_states:
                self.td_h_block1_exc, self.td_h_block1_inh, _ = self.td_fgru_block1(
                    td_0_to_block1, self.td_h_block1_exc, self.td_h_block1_inh
                )
            else:
                self.td_h_block1_exc, _, _ = self.td_fgru_block1(td_0_to_block1, self.td_h_block1_exc)
            
            if self.skip_connections:
                self.h_block1_exc = self.h_block1_exc + self.td_h_block1_exc
            else:
                self.h_block1_exc = self.td_h_block1_exc
                
        # Output from full resolution td_h_block1
        output = self.output_projection(self.td_h_block1_exc)
            
        return output
    
    def reset_hidden_states(self):
        """Reset all hidden states."""
        # Excitatory states
        self.h_block1_exc = None
        self.h0_exc = None
        self.h1_exc = None
        self.h2_exc = None
        self.h3_exc = None
        self.td_h_block1_exc = None
        self.td_h0_exc = None
        self.td_h1_exc = None
        self.td_h2_exc = None
        
        # Inhibitory states
        if self.use_separate_ei_states:
            self.h_block1_inh = None
            self.h0_inh = None
            self.h1_inh = None
            self.h2_inh = None
            self.h3_inh = None
            self.td_h_block1_inh = None
            self.td_h0_inh = None
            self.td_h1_inh = None
            self.td_h2_inh = None