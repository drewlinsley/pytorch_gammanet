#!/usr/bin/env python
"""Training script for ablation models.

This script extends the regular training script to handle ablation models.

Usage:
    python scripts/train_ablation.py --config config/ablations/ffonly.yaml
    python scripts/train_ablation.py --config config/ablations/no_gates.yaml --resume
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
from gammanet.models import get_ablation_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ablation models')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to ablation configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/ablations',
                        help='Directory to save ablation checkpoints')
    
    return parser.parse_args()


def load_ablation_config(config_path: str) -> dict:
    """Load ablation configuration and merge with base config."""
    with open(config_path, 'r') as f:
        ablation_config = yaml.safe_load(f)
    
    # Load base config if specified
    if 'base_config' in ablation_config['model']:
        base_config_path = Path(config_path).parent / ablation_config['model']['base_config']
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (ablation overrides base)
        merged_config = base_config.copy()
        
        # Deep merge
        for key, value in ablation_config.items():
            if key in merged_config and isinstance(value, dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
                
        return merged_config
    
    return ablation_config


def create_ablation_model(config: dict):
    """Create ablation model based on configuration."""
    model_class_name = config['model'].get('name', config['model'].get('class'))
    
    # Get ablation model class
    if model_class_name in ['GammaNetFFOnly', 'GammaNetHOnly', 'GammaNetTDOnly',
                           'GammaNetNoRecurrence', 'GammaNetBottomUpOnly',
                           'GammaNetAdditiveOnly', 'GammaNetMultiplicativeOnly',
                           'GammaNetNoGates', 'GammaNetNoDivisive']:
        # Extract short name
        short_name = model_class_name.replace('GammaNet', '').lower()
        short_name = ''.join(['_' + c.lower() if c.isupper() else c for c in short_name]).lstrip('_')
        
        # Get model class
        model_class = get_ablation_model(short_name)
    else:
        raise ValueError(f"Unknown ablation model: {model_class_name}")
    
    # Create model
    model = model_class(
        config=config['model'],
        input_channels=3,
        output_channels=1
    )
    
    return model


class AblationTrainer(GammaNetTrainer):
    """Extended trainer for ablation models."""
    
    def __init__(self, config: dict, accelerator: Accelerator):
        """Initialize ablation trainer."""
        super().__init__(config, accelerator)
        
        # Store ablation info
        self.ablation_info = None
        
    def _create_model(self):
        """Create ablation model instead of regular GammaNet."""
        self.model = create_ablation_model(self.config)
        
        # Get ablation info
        if hasattr(self.model, 'get_ablation_info'):
            self.ablation_info = self.model.get_ablation_info()
            self.accelerator.print(f"Ablation: {self.ablation_info['description']}")
            
    def _log_metrics(self, metrics: dict, step: int, prefix: str = "train"):
        """Log metrics with ablation info."""
        # Add ablation type to metrics
        if self.ablation_info:
            metrics['ablation_type'] = self.ablation_info['class']
            
        super()._log_metrics(metrics, step, prefix)
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint with ablation configuration."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'ablation_info': self.ablation_info
        }
        
        # Save with ablation name
        if self.ablation_info:
            ablation_name = self.ablation_info['class'].replace('GammaNet', '').lower()
            filename = f"{ablation_name}_epoch_{epoch}.pt"
            if is_best:
                best_filename = f"{ablation_name}_best.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}.pt"
            best_filename = "best_model.pt"
            
        save_path = self.checkpoint_dir / filename
        
        self.accelerator.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.checkpoint_dir / best_filename
            self.accelerator.save(checkpoint, best_path)
            
        return save_path


def main():
    """Main training function for ablations."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_ablation_config(args.config)
    
    # Override output directory
    if args.output_dir:
        config['training']['checkpoint_dir'] = args.output_dir
        
    # Override data directory if specified
    if args.data_dir:
        config['data']['train_path'] = Path(args.data_dir) / 'train'
        config['data']['val_path'] = Path(args.data_dir) / 'val'
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='fp16' if config['training'].get('mixed_precision', True) else 'no',
        gradient_accumulation_steps=config['training'].get('gradient_accumulation', 1),
    )
    
    # Print configuration
    if accelerator.is_main_process:
        accelerator.print(f"Ablation Configuration: {args.config}")
        accelerator.print(yaml.dump(config, default_flow_style=False))
    
    # Create dataloaders
    train_transform = get_train_transforms(
        crop_size=config['data']['augmentation'].get('random_crop', 320),
        mean=config['data']['normalize_mean'],
        std=config['data']['normalize_std']
    )
    
    val_transform = get_val_transforms(
        mean=config['data']['normalize_mean'],
        std=config['data']['normalize_std']
    )
    
    train_dataset = BSDS500Dataset(
        root_dir=config['data']['train_path'],
        split='train',
        transform=train_transform,
        cache_data=True
    )
    
    val_dataset = BSDS500Dataset(
        root_dir=config['data']['val_path'],
        split='val',
        transform=val_transform,
        cache_data=True
    )
    
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
    
    # Initialize trainer
    trainer = AblationTrainer(config, accelerator)
    
    # Prepare for distributed training
    trainer.prepare_for_training(train_loader, val_loader)
    
    # Load checkpoint if resuming
    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    accelerator.print(f"Starting ablation training for {config['training']['num_epochs']} epochs")
    trainer.fit()
    
    # Save final ablation info
    if accelerator.is_main_process and trainer.ablation_info:
        info_path = Path(config['training']['checkpoint_dir']) / 'ablation_info.yaml'
        with open(info_path, 'w') as f:
            yaml.dump(trainer.ablation_info, f)
    
    accelerator.print("Ablation training completed!")


if __name__ == '__main__':
    main()