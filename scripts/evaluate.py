#!/usr/bin/env python
"""Evaluation script for GammaNet.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data-dir /path/to/BSDS500
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --tta  # Test-time augmentation
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gammanet.models import GammaNet
from gammanet.data import BSDS500Dataset, get_val_transforms, get_tta_transforms
from gammanet.utils import compute_ods_ois, EdgeDetectionMetrics
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate GammaNet on BSDS500')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to BSDS500 data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction images')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model from checkpoint.
    
    Returns:
        model, config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = GammaNet(
        config=config['model'],
        input_channels=3,
        output_channels=1
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def apply_tta(model, image, transforms):
    """Apply test-time augmentation.
    
    Args:
        model: The model to use for predictions
        image: Input image [C, H, W]
        transforms: List of TTA transforms
        
    Returns:
        Average prediction across all augmentations
    """
    predictions = []
    
    for transform in transforms:
        # Apply transform
        aug_image = transform(image=image.permute(1, 2, 0).cpu().numpy())['image']
        aug_image = torch.from_numpy(aug_image).permute(2, 0, 1).unsqueeze(0).to(image.device)
        
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(aug_image))
            
        # Reverse augmentation on prediction
        # This is simplified - proper implementation would reverse specific augmentations
        predictions.append(pred.squeeze(0).cpu().numpy())
        
    # Average predictions
    return np.mean(predictions, axis=0)


@torch.no_grad()
def evaluate(model, dataloader, device, use_tta=False, save_predictions=False, output_dir=None):
    """Run evaluation on dataset.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to use
        use_tta: Whether to use test-time augmentation
        save_predictions: Whether to save prediction images
        output_dir: Directory to save predictions
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = EdgeDetectionMetrics()
    all_predictions = []
    all_targets = []
    
    # Create output directory if saving predictions
    if save_predictions and output_dir:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
    # Get TTA transforms if needed
    if use_tta:
        tta_transforms = get_tta_transforms()
    
    # Process each image
    for batch in tqdm(dataloader, desc='Evaluating'):
        images = batch['image'].to(device)
        targets = batch['edges']
        image_ids = batch['image_id']
        
        # Reset hidden states
        model.reset_hidden_states()
        
        if use_tta and dataloader.batch_size == 1:
            # Apply TTA (only works with batch size 1)
            pred = apply_tta(model, images[0], tta_transforms)
            pred = torch.from_numpy(pred).unsqueeze(0)
        else:
            # Standard prediction
            logits = model(images)
            pred = torch.sigmoid(logits).cpu()
            
        # Update metrics
        metrics.update(pred, targets)
        
        # Store for ODS/OIS computation
        for p, t in zip(pred, targets):
            all_predictions.append(p.squeeze().numpy())
            all_targets.append(t.squeeze().numpy())
            
        # Save predictions if requested
        if save_predictions:
            for i, img_id in enumerate(image_ids):
                pred_img = (pred[i, 0].numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(pred_dir / f"{img_id}.png"), pred_img)
                
    # Compute final metrics
    results = compute_ods_ois(all_predictions, all_targets)
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
        
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create dataset
    transform = get_val_transforms(
        mean=config['data']['normalize_mean'],
        std=config['data']['normalize_std']
    )
    
    dataset = BSDS500Dataset(
        root_dir=args.data_dir,
        split=args.split,
        transform=transform,
        cache_data=False  # Don't cache for evaluation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    # Run evaluation
    print(f"Evaluating on {len(dataset)} images...")
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        use_tta=args.tta,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\nResults:")
    print(f"  ODS F1: {results['ods_f1']:.4f} (threshold: {results['ods_threshold']:.3f})")
    print(f"  OIS F1: {results['ois_f1']:.4f}")
    print(f"  AP: {results['ap']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"results_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'split': args.split,
            'use_tta': args.tta,
            'results': results
        }, f, indent=2)
        
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()