#!/usr/bin/env python
"""Evaluation script for GammaNet models."""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from gammanet.models import VGG16GammaNet, VGG16GammaNetV2
from gammanet.data import BSDS500Dataset
from gammanet.utils import EdgeDetectionMetrics
from gammanet.training.losses import BiBalancedBCELoss


def load_model(config, checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    # Create model
    model_version = config['model'].get('model_version', 'v2')
    use_backbone = config['model'].get('use_backbone', True)
    
    if use_backbone:
        if model_version == 'v2':
            model = VGG16GammaNetV2(
                config=config['model'],
                input_channels=3,
                output_channels=1
            )
            print("Using VGG16GammaNetV2 with E/I states")
        else:
            model = VGG16GammaNet(
                config=config['model'],
                input_channels=3,
                output_channels=1
            )
            print("Using VGG16GammaNet (v1)")
    else:
        raise NotImplementedError("Non-backbone models not implemented for evaluation")
    
    # Load checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        # Handle numpy version compatibility issues
        import sys
        import numpy
        if not hasattr(numpy, '_core'):
            numpy._core = numpy.core
            sys.modules['numpy._core'] = numpy.core
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
            
        model.load_state_dict(new_state_dict)
        
        if 'epoch' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint loaded, using randomly initialized model")
    
    return model.to(device)


def evaluate(model, dataloader, criterion, device='cpu', accelerator=None):
    """Evaluate model on dataset."""
    model.eval()
    metrics = EdgeDetectionMetrics()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Reset hidden states
            if hasattr(model, 'reset_hidden_states'):
                model.reset_hidden_states()
            
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['edges'].to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            probs = torch.sigmoid(predictions)
            metrics.update(probs, targets)
    
    # Compute final metrics
    results = metrics.compute()
    results['loss'] = total_loss / num_batches
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GammaNet model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Override test data path')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch checkpoint to load')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # If specific epoch requested, construct checkpoint path
    if args.epoch is not None:
        checkpoint_path = f"logs/checkpoints/checkpoint_epoch_{args.epoch}.pt"
        print(f"Looking for checkpoint at epoch {args.epoch}")
    else:
        checkpoint_path = args.checkpoint
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    
    # Create test dataset
    test_path = args.data_path or config['data'].get('test_path', config['data']['val_path'])
    print(f"Loading test data from: {test_path}")
    
    test_dataset = BSDS500Dataset(
        root_dir=test_path,
        split='test',
        transform=None,  # No augmentation for testing
        target_transform=None,
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create loss function
    loss_type = config['training'].get('loss', 'bi_bce_hed')
    if loss_type == 'bi_bce_hed':
        criterion = BiBalancedBCELoss(gamma=0.5, neg_weight=1.1)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, criterion, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {results['loss']:.4f}")
    print(f"ODS F1: {results['ods_f1']:.4f}")
    print(f"ODS Precision: {results['ods_precision']:.4f}")
    print(f"ODS Recall: {results['ods_recall']:.4f}")
    print(f"OIS F1: {results['ois_f1']:.4f}")
    print(f"OIS Precision: {results['ois_precision']:.4f}")
    print(f"OIS Recall: {results['ois_recall']:.4f}")
    print(f"AP: {results['ap']:.4f}")
    print("="*50)
    
    # Save results
    import json
    results_path = Path(checkpoint_path).parent / f"eval_results_epoch_{args.epoch or 'best'}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()