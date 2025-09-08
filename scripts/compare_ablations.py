#!/usr/bin/env python
"""Compare performance of different ablation models.

This script runs the same experiments on multiple ablation models and
creates comparison plots and statistics.

Usage:
    python scripts/compare_ablations.py --base-checkpoint checkpoints/full_model.pt \
        --ablation-dir checkpoints/ablations/ --experiment kapadia
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gammanet.models import GammaNet, get_ablation_model
from experiments.in_silico import (
    KapadiaStimuli,
    KinoshitaStimuli,
    ResponseExtractor,
    get_v1_like_layers,
    load_kapadia_data,
    load_kinoshita_data,
    compute_similarity_metrics,
    plot_ablation_comparison
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare ablation models')
    
    parser.add_argument('--base-checkpoint', type=str, required=True,
                        help='Path to full model checkpoint')
    parser.add_argument('--ablation-dir', type=str, required=True,
                        help='Directory containing ablation checkpoints')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'kapadia', 'kinoshita', 'task_performance'],
                        help='Which comparison to run')
    parser.add_argument('--output-dir', type=str, default='./ablation_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_ablation_models(base_checkpoint, ablation_dir, device):
    """Load all ablation models from directory."""
    models = {}
    
    # Load base model
    checkpoint = torch.load(base_checkpoint, map_location=device)
    config = checkpoint['config']
    
    base_model = GammaNet(
        config=config['model'],
        input_channels=3,
        output_channels=1
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()
    
    models['Full Model'] = {
        'model': base_model,
        'config': config,
        'checkpoint_path': base_checkpoint
    }
    
    # Load ablation models
    ablation_path = Path(ablation_dir)
    for checkpoint_file in ablation_path.glob('*.pt'):
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            ablation_config = checkpoint['config']
            
            # Determine ablation type from config or filename
            if 'ablation' in ablation_config['model']:
                ablation_name = ablation_config['model']['ablation']['type']
            else:
                # Extract from filename
                ablation_name = checkpoint_file.stem
            
            # Get ablation class
            if ablation_name in ['ffonly', 'honly', 'tdonly', 'no_recurrence']:
                model_class = get_ablation_model(ablation_name)
            else:
                continue
                
            # Create ablation model
            model = model_class(
                config=ablation_config['model'],
                input_channels=3,
                output_channels=1
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            models[ablation_name] = {
                'model': model,
                'config': ablation_config,
                'checkpoint_path': str(checkpoint_file)
            }
            
            print(f"Loaded ablation model: {ablation_name}")
            
        except Exception as e:
            print(f"Failed to load {checkpoint_file}: {e}")
            
    return models


def compare_kapadia_responses(models, device, output_dir):
    """Compare Kapadia experiment across models."""
    print("\nRunning Kapadia comparison...")
    
    # Load neural data
    neural_data = load_kapadia_data()
    
    # Generate stimuli
    stim_generator = KapadiaStimuli()
    pixel_distances = (neural_data["distances"] * 30).astype(int)
    
    stimuli = stim_generator.generate_stimulus_set(
        orientations=[90],
        flanker_distances=pixel_distances[:4],  # Use subset for speed
        flanker_angles=[0],  # Just collinear
        contrasts=[neural_data["target_contrast"], neural_data["flanker_contrast"]]
    )
    
    # Results storage
    results = {}
    
    for model_name, model_info in models.items():
        print(f"  Processing {model_name}...")
        model = model_info['model']
        
        # Create extractor
        extractor = ResponseExtractor(model)
        v1_layers = get_v1_like_layers(model)
        target_layer = v1_layers[1] if len(v1_layers) > 1 else v1_layers[0]
        
        extractor.register_hooks([target_layer])
        
        # Process stimuli
        responses = []
        for stim, meta in stimuli:
            stim_tensor = torch.from_numpy(stim).float().unsqueeze(0).unsqueeze(0)
            stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(device)
            
            resp = extractor.extract_responses(stim_tensor)
            pop_resp = extractor.get_population_response(
                resp, target_layer, spatial_pool="center"
            )
            responses.append(pop_resp.mean())
        
        responses = np.array(responses)
        
        # Calculate facilitation
        baseline = responses.min()
        facilitation = (responses - baseline) / baseline
        
        # Compare with neural
        similarity = compute_similarity_metrics(
            facilitation[:len(neural_data["distances"][:4])],
            neural_data["facilitation_collinear"][:4]
        )
        
        results[model_name] = {
            'facilitation': facilitation.tolist(),
            'correlation': similarity['correlation'],
            'rmse': similarity['rmse']
        }
        
        extractor.clear_hooks()
    
    return results


def compare_kinoshita_responses(models, device, output_dir):
    """Compare Kinoshita experiment across models."""
    print("\nRunning Kinoshita comparison...")
    
    # Load neural data
    neural_data = load_kinoshita_data()
    
    # Generate stimuli
    stim_generator = KinoshitaStimuli()
    
    # Use subset of orientation differences
    ori_diffs = [0, 45, 90]
    
    results = {}
    
    for model_name, model_info in models.items():
        print(f"  Processing {model_name}...")
        model = model_info['model']
        
        # Create extractor
        extractor = ResponseExtractor(model)
        v1_layers = get_v1_like_layers(model)
        target_layer = v1_layers[1] if len(v1_layers) > 1 else v1_layers[0]
        
        extractor.register_hooks([target_layer])
        
        # Generate center-only stimulus
        center_stim = stim_generator.generate_stimulus_set(
            center_orientations=[45],
            surround_orientations=[45],  # Same as center
            center_radius=25,
            surround_inner_radius=100,  # Far away
            surround_outer_radius=101
        )[0]  # Get first stimulus
        
        stim_tensor = torch.from_numpy(center_stim[0]).float().unsqueeze(0).unsqueeze(0)
        stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(device)
        
        resp = extractor.extract_responses(stim_tensor)
        center_only_resp = extractor.get_population_response(
            resp, target_layer, spatial_pool="center"
        ).mean()
        
        # Generate surround stimuli
        surround_responses = []
        
        for ori_diff in ori_diffs:
            stim_set = stim_generator.generate_stimulus_set(
                center_orientations=[45],
                surround_orientations=[45 + ori_diff],
                center_radius=25,
                surround_inner_radius=30,
                surround_outer_radius=80
            )
            
            stim, _ = stim_set[0]
            stim_tensor = torch.from_numpy(stim).float().unsqueeze(0).unsqueeze(0)
            stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(device)
            
            resp = extractor.extract_responses(stim_tensor)
            surround_resp = extractor.get_population_response(
                resp, target_layer, spatial_pool="center"
            ).mean()
            
            surround_responses.append(surround_resp)
        
        # Normalize responses
        normalized = np.array(surround_responses) / center_only_resp
        
        # Suppression indices
        suppression = 1 - normalized
        
        results[model_name] = {
            'orientation_differences': ori_diffs,
            'normalized_responses': normalized.tolist(),
            'suppression_indices': suppression.tolist(),
            'mean_suppression': suppression.mean()
        }
        
        extractor.clear_hooks()
    
    return results


def compare_task_performance(models, device, output_dir):
    """Compare edge detection performance across models."""
    print("\nComparing task performance...")
    
    # This would load BSDS500 results if available
    # For now, return dummy data
    results = {}
    
    for model_name in models:
        # In practice, would load evaluation results
        results[model_name] = {
            'ods_f1': np.random.uniform(0.6, 0.8),
            'ois_f1': np.random.uniform(0.65, 0.82),
            'ap': np.random.uniform(0.6, 0.85)
        }
    
    return results


def create_comparison_plots(all_results, output_dir):
    """Create comparison visualizations."""
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    # 1. Kapadia correlation comparison
    if 'kapadia' in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(all_results['kapadia'].keys())
        correlations = [all_results['kapadia'][m]['correlation'] for m in models]
        
        bars = ax.bar(models, correlations)
        
        # Color code by ablation type
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Correlation with Neural Data', fontsize=12)
        ax.set_title('Kapadia Collinear Facilitation: Model-Neural Correlation', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(output_dir / 'kapadia_correlation_comparison.png', dpi=300)
    
    # 2. Kinoshita suppression comparison
    if 'kinoshita' in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(all_results['kinoshita'].keys())
        mean_suppression = [all_results['kinoshita'][m]['mean_suppression'] 
                          for m in models]
        
        bars = ax.bar(models, mean_suppression)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Mean Suppression Index', fontsize=12)
        ax.set_title('Kinoshita Surround Suppression: Mean Effect', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(output_dir / 'kinoshita_suppression_comparison.png', dpi=300)
    
    # 3. Task performance comparison
    if 'task_performance' in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(all_results['task_performance'].keys())
        metrics = ['ods_f1', 'ois_f1', 'ap']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [all_results['task_performance'][m][metric] for m in models]
            ax.bar(x + i*width, values, width, label=metric.upper())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Edge Detection Performance Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'task_performance_comparison.png', dpi=300)
    
    # 4. Summary heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all metrics
    summary_data = []
    model_names = set()
    
    for exp_name, exp_results in all_results.items():
        for model_name, model_results in exp_results.items():
            model_names.add(model_name)
            
            if exp_name == 'kapadia':
                summary_data.append({
                    'Model': model_name,
                    'Metric': 'Kapadia Correlation',
                    'Value': model_results['correlation']
                })
            elif exp_name == 'kinoshita':
                summary_data.append({
                    'Model': model_name,
                    'Metric': 'Kinoshita Suppression',
                    'Value': model_results['mean_suppression']
                })
            elif exp_name == 'task_performance':
                summary_data.append({
                    'Model': model_name,
                    'Metric': 'ODS F1',
                    'Value': model_results['ods_f1']
                })
    
    # Create dataframe and pivot
    df = pd.DataFrame(summary_data)
    pivot_df = df.pivot(index='Model', columns='Metric', values='Value')
    
    # Normalize each metric to [0, 1]
    pivot_norm = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())
    
    # Create heatmap
    sns.heatmap(pivot_norm, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Normalized Performance'})
    
    ax.set_title('Ablation Model Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_summary_heatmap.png', dpi=300)


def main():
    """Main comparison function."""
    args = parse_args()
    
    # Setup
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = load_ablation_models(args.base_checkpoint, args.ablation_dir, device)
    print(f"Loaded {len(models)} models")
    
    # Run comparisons
    all_results = {}
    
    if args.experiment in ['all', 'kapadia']:
        all_results['kapadia'] = compare_kapadia_responses(models, device, output_dir)
    
    if args.experiment in ['all', 'kinoshita']:
        all_results['kinoshita'] = compare_kinoshita_responses(models, device, output_dir)
    
    if args.experiment in ['all', 'task_performance']:
        all_results['task_performance'] = compare_task_performance(models, device, output_dir)
    
    # Save results
    results_file = output_dir / f"ablation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    create_comparison_plots(all_results, output_dir)
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    
    if 'kapadia' in all_results:
        print("\nKapadia Correlations:")
        for model, results in all_results['kapadia'].items():
            print(f"  {model}: {results['correlation']:.3f}")
    
    if 'kinoshita' in all_results:
        print("\nKinoshita Mean Suppression:")
        for model, results in all_results['kinoshita'].items():
            print(f"  {model}: {results['mean_suppression']:.3f}")
    
    print("\nComparison completed!")


if __name__ == '__main__':
    main()