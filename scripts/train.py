#!/usr/bin/env python
"""Training script for GammaNet.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --resume checkpoints/best_model.pt
"""

import argparse
import yaml
import torch
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gammanet.data import BSDS500Dataset, get_train_transforms, get_val_transforms
from gammanet.training import GammaNetTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GammaNet on BSDS500')
    
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of data workers')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Override number of epochs')
    
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load and update configuration from file and command line args."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override with command line arguments
    if args.data_dir:
        config['data']['train_path'] = Path(args.data_dir) / 'train'
        config['data']['val_path'] = Path(args.data_dir) / 'val'
        
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
        
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
        
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
        
    return config


def create_dataloaders(config: dict) -> tuple:
    """Create training and validation dataloaders."""
    # Get transforms
    train_transform = get_train_transforms(
        crop_size=config['data']['augmentation'].get('random_crop', 320),
        mean=config['data']['normalize_mean'],
        std=config['data']['normalize_std']
    )
    
    val_transform = get_val_transforms(
        crop_size=config['data']['augmentation'].get('random_crop', 320),
        mean=config['data']['normalize_mean'],
        std=config['data']['normalize_std']
    )
    
    # Create datasets
    train_dataset = BSDS500Dataset(
        root_dir=config['data']['train_path'],
        split='train',
        transform=train_transform,
        cache_data=True,
        thin_edges=config['data'].get('thin_edges', False)  # Default to False for now
    )
    
    val_dataset = BSDS500Dataset(
        root_dir=config['data']['val_path'],
        split='val',
        transform=val_transform,
        cache_data=True,
        thin_edges=config['data'].get('thin_edges', False)  # Default to False for now
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='fp16' if config['training'].get('mixed_precision', True) else 'no',
        gradient_accumulation_steps=config['training'].get('gradient_accumulation', 1),
    )
    
    # Print configuration
    if accelerator.is_main_process:
        accelerator.print("Configuration:")
        accelerator.print(yaml.dump(config, default_flow_style=False))
        
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Initialize trainer
    trainer = GammaNetTrainer(config, accelerator)
    
    # Prepare for distributed training
    trainer.prepare_for_training(train_loader, val_loader)
    
    # Load checkpoint if resuming
    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Start training
    accelerator.print(f"Starting training for {config['training']['num_epochs']} epochs")
    trainer.fit()
    
    accelerator.print("Training completed!")


if __name__ == '__main__':
    main()