#!/usr/bin/env python
"""Visualize the GammaNet architecture changes."""

import torch
import yaml
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from gammanet.models import GammaNet


def visualize_architecture():
    """Visualize the GammaNet architecture flow."""
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = GammaNet(
        config=config['model'],
        input_channels=3,
        output_channels=1
    )
    
    print("=" * 80)
    print("GammaNet Architecture Visualization")
    print("=" * 80)
    print("\nLayer Configuration:")
    for i, layer_cfg in enumerate(model.layers_config):
        layer_type = "Encoder" if i < model.encoder_count else "Decoder"
        print(f"Layer {i} ({layer_type}): {layer_cfg['features']} channels, "
              f"pool={layer_cfg.get('pool', False)}, "
              f"h_kernel={layer_cfg.get('h_kernel', 'N/A')}")
    
    print(f"\nEncoder count: {model.encoder_count}")
    print(f"Decoder count: {model.decoder_count}")
    
    print("\n" + "=" * 80)
    print("Information Flow (per timestep):")
    print("=" * 80)
    
    print("\n1. BOTTOM-UP PASS (Encoder):")
    for i in range(model.encoder_count):
        layer_cfg = model.layers_config[i]
        print(f"   Layer {i}: {layer_cfg['features']}ch")
        if i < model.encoder_count - 1:
            print("      ↓")
    
    print("\n2. TOP-DOWN PASS (Decoder):")
    print(f"   Starting from encoder layer {model.encoder_count-1}")
    
    for i in range(model.decoder_count):
        if i in model.decoder_to_encoder_mapping:
            encoder_idx = model.decoder_to_encoder_mapping[i]
            decoder_idx = model.encoder_count + i
            decoder_cfg = model.layers_config[decoder_idx]
            encoder_cfg = model.layers_config[encoder_idx]
            
            print(f"\n   Decoder {i} ({decoder_cfg['features']}ch) "
                  f"modulates Encoder {encoder_idx} ({encoder_cfg['features']}ch)")
            print(f"   - Top-down signal upsampled to match encoder resolution")
            print(f"   - No skip connections (per paper)")
            
            if i < model.decoder_count - 1:
                print("      ↓")
    
    print("\n3. OUTPUT:")
    print(f"   Final output from Encoder 0 hidden state → 1 channel edge map")
    
    print("\n" + "=" * 80)
    print("Key Architecture Changes:")
    print("=" * 80)
    print("1. Dynamic encoder/decoder split (no hardcoded :5)")
    print("2. Automatic encoder-decoder pairing")
    print("3. Removed skip connections in decoder (following paper)")
    print("4. Decoder only receives top-down signal to modulate encoder states")
    print("=" * 80)


if __name__ == '__main__':
    visualize_architecture()