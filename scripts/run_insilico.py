#!/usr/bin/env python
"""Run in silico neurophysiology experiments on trained GammaNet models.

This script generates stimuli, extracts model responses, and compares
with primate neurophysiology data.

Usage:
    python scripts/run_insilico.py --checkpoint checkpoints/best_model.pt --experiment all
    python scripts/run_insilico.py --checkpoint checkpoints/best_model.pt --experiment kapadia
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gammanet.models import GammaNet
from experiments.in_silico import (
    # Stimuli
    OrientedGratingStimuli,
    KapadiaStimuli,
    KinoshitaStimuli,
    TextureBoundaryStimuli,
    create_stimulus_batch,
    
    # Response extraction
    ResponseExtractor,
    get_v1_like_layers,
    
    # Analysis
    OrientationTuningAnalyzer,
    ContrastResponseAnalyzer,
    SurroundModulationAnalyzer,
    
    # Neural comparison
    load_kapadia_data,
    load_kinoshita_data,
    compute_similarity_metrics,
    create_model_neural_alignment_report,
    
    # Visualization
    plot_tuning_curves,
    plot_spatial_interactions,
    plot_model_neural_comparison,
    create_summary_figure
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run in silico experiments')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'orientation', 'contrast', 'kapadia', 
                                'kinoshita', 'texture_boundary'],
                        help='Which experiment to run')
    parser.add_argument('--layer', type=str, default=None,
                        help='Specific layer to analyze (default: auto-detect V1-like)')
    parser.add_argument('--output-dir', type=str, default='./insilico_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save raw neural responses')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for stimulus processing')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load model from checkpoint."""
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


def run_orientation_tuning(model, extractor, target_layer, device, output_dir):
    """Run orientation tuning experiment."""
    print("\nRunning orientation tuning experiment...")
    
    # Generate stimuli
    stim_generator = OrientedGratingStimuli()
    orientations = np.arange(0, 180, 15)
    stimuli = stim_generator.generate_stimulus_set(
        orientations=orientations,
        spatial_frequencies=[4.0],
        contrasts=[0.5]
    )
    
    # Create batch
    stim_batch, metadata = create_stimulus_batch(stimuli)
    stim_batch = stim_batch.to(device)
    
    # Extract responses
    extractor.register_hooks([target_layer])
    responses = extractor.extract_responses(stim_batch)
    
    # Get population response
    pop_response = extractor.get_population_response(
        responses, target_layer, spatial_pool="center"
    )
    
    # Average across neurons and timesteps
    mean_response = pop_response.mean(axis=(1, 2))
    
    # Analyze tuning
    analyzer = OrientationTuningAnalyzer()
    tuning_curve = analyzer.fit_tuning_curve(orientations, mean_response)
    
    # Save results
    results = {
        "orientations": orientations.tolist(),
        "responses": mean_response.tolist(),
        "tuning_curve": {
            "preferred_orientation": tuning_curve.preferred_value,
            "bandwidth": tuning_curve.bandwidth,
            "r_squared": tuning_curve.r_squared,
            "modulation_index": tuning_curve.modulation_index
        }
    }
    
    # Plot
    fig = plot_tuning_curves(
        [{"x_values": orientations, "y_values": mean_response,
          "preferred": tuning_curve.preferred_value}],
        labels=["Model"],
        title="Orientation Tuning",
        save_path=output_dir / "orientation_tuning.png"
    )
    
    return results


def run_contrast_response(model, extractor, target_layer, device, output_dir):
    """Run contrast response experiment."""
    print("\nRunning contrast response experiment...")
    
    # Generate stimuli at preferred orientation
    stim_generator = OrientedGratingStimuli()
    contrasts = np.logspace(-2, 0, 8)  # 1% to 100%
    
    stimuli = stim_generator.generate_stimulus_set(
        orientations=[45],  # Single orientation
        spatial_frequencies=[4.0],
        contrasts=contrasts
    )
    
    # Process stimuli
    stim_batch, metadata = create_stimulus_batch(stimuli)
    stim_batch = stim_batch.to(device)
    
    # Extract responses
    responses = extractor.extract_responses(stim_batch)
    pop_response = extractor.get_population_response(
        responses, target_layer, spatial_pool="center"
    )
    mean_response = pop_response.mean(axis=(1, 2))
    
    # Analyze contrast response
    analyzer = ContrastResponseAnalyzer()
    crf = analyzer.fit_contrast_response(contrasts, mean_response)
    
    # Save results
    results = {
        "contrasts": contrasts.tolist(),
        "responses": mean_response.tolist(),
        "contrast_response_fit": {
            "c50": crf.fit_params["c50"] if crf.fit_params else None,
            "r_max": crf.fit_params["r_max"] if crf.fit_params else None,
            "n": crf.fit_params["n"] if crf.fit_params else None,
            "r_squared": crf.r_squared
        }
    }
    
    # Plot
    fig = plot_contrast_response(
        [{"contrasts": contrasts, "responses": mean_response,
          "c50": crf.fit_params["c50"] if crf.fit_params else None}],
        labels=["Model"],
        title="Contrast Response Function",
        save_path=output_dir / "contrast_response.png"
    )
    
    return results


def run_kapadia_experiment(model, extractor, target_layer, device, output_dir):
    """Run Kapadia collinear facilitation experiment."""
    print("\nRunning Kapadia collinear facilitation experiment...")
    
    # Load neural data
    neural_data = load_kapadia_data()
    
    # Generate stimuli
    stim_generator = KapadiaStimuli()
    
    # Convert distances from visual degrees to pixels (assuming 30 pixels/degree)
    pixel_distances = (neural_data["distances"] * 30).astype(int)
    
    stimuli = stim_generator.generate_stimulus_set(
        orientations=[90],  # Vertical
        flanker_distances=pixel_distances,
        flanker_angles=[0, 90],  # Collinear and orthogonal
        contrasts=[neural_data["target_contrast"], neural_data["flanker_contrast"]]
    )
    
    # Process in batches
    results_collinear = []
    results_orthogonal = []
    
    for i, (stim, meta) in enumerate(stimuli):
        stim_tensor = torch.from_numpy(stim).float().unsqueeze(0).unsqueeze(0)
        stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(device)
        
        responses = extractor.extract_responses(stim_tensor)
        pop_response = extractor.get_population_response(
            responses, target_layer, spatial_pool="center"
        )
        mean_resp = pop_response.mean()
        
        if meta.parameters["flanker_angle"] == 0:
            results_collinear.append(mean_resp)
        else:
            results_orthogonal.append(mean_resp)
    
    # Calculate facilitation indices
    # Group by distance
    n_distances = len(pixel_distances)
    collinear_by_dist = np.array(results_collinear[:n_distances])
    orthogonal_by_dist = np.array(results_orthogonal[:n_distances])
    
    # Baseline (no flankers) - approximate as minimum response
    baseline = min(collinear_by_dist.min(), orthogonal_by_dist.min())
    
    # Facilitation index
    facilitation_collinear = (collinear_by_dist - baseline) / baseline
    facilitation_orthogonal = (orthogonal_by_dist - baseline) / baseline
    
    # Compare with neural data
    similarity = compute_similarity_metrics(
        facilitation_collinear,
        neural_data["facilitation_collinear"],
        neural_data["facilitation_se"]
    )
    
    # Save results
    results = {
        "distances": neural_data["distances"].tolist(),
        "model_facilitation_collinear": facilitation_collinear.tolist(),
        "model_facilitation_orthogonal": facilitation_orthogonal.tolist(),
        "neural_facilitation_collinear": neural_data["facilitation_collinear"].tolist(),
        "neural_facilitation_orthogonal": neural_data["facilitation_orthogonal"].tolist(),
        "similarity_metrics": similarity
    }
    
    # Plot comparison
    fig = plot_spatial_interactions({
        "collinear_facilitation": {
            "distances": neural_data["distances"],
            "facilitation": facilitation_collinear,
            "orthogonal": facilitation_orthogonal
        }
    }, title="Kapadia Collinear Facilitation",
    save_path=output_dir / "kapadia_facilitation.png")
    
    # Model vs neural scatter plot
    fig2 = plot_model_neural_comparison(
        facilitation_collinear,
        neural_data["facilitation_collinear"],
        neural_data["facilitation_se"],
        title="Kapadia: Model vs Neural",
        save_path=output_dir / "kapadia_model_neural.png"
    )
    
    return results


def run_kinoshita_experiment(model, extractor, target_layer, device, output_dir):
    """Run Kinoshita surround modulation experiment."""
    print("\nRunning Kinoshita surround modulation experiment...")
    
    # Load neural data
    neural_data = load_kinoshita_data()
    
    # Generate stimuli
    stim_generator = KinoshitaStimuli()
    
    stimuli = stim_generator.generate_stimulus_set(
        center_orientations=[45],
        surround_orientations=None,  # Use relative angles
        center_radius=25,
        surround_inner_radius=30,
        surround_outer_radius=80,
        contrasts=[neural_data["contrast"], neural_data["contrast"]]
    )
    
    # Process stimuli
    center_only_resp = None
    surround_responses = []
    
    for stim, meta in stimuli:
        stim_tensor = torch.from_numpy(stim).float().unsqueeze(0).unsqueeze(0)
        stim_tensor = stim_tensor.repeat(1, 3, 1, 1).to(device)
        
        responses = extractor.extract_responses(stim_tensor)
        pop_response = extractor.get_population_response(
            responses, target_layer, spatial_pool="center"
        )
        mean_resp = pop_response.mean()
        
        if meta.parameters.get("condition") == "center_only":
            center_only_resp = mean_resp
        else:
            surround_responses.append((
                meta.parameters["orientation_difference"],
                mean_resp
            ))
    
    # Sort by orientation difference
    surround_responses.sort(key=lambda x: x[0])
    ori_diffs = [x[0] for x in surround_responses]
    responses = [x[1] for x in surround_responses]
    
    # Normalize by center-only response
    normalized_responses = np.array(responses) / center_only_resp
    
    # Compare with neural data
    # Interpolate to match neural data points
    from scipy.interpolate import interp1d
    interp_func = interp1d(ori_diffs, normalized_responses, kind='linear', 
                          fill_value='extrapolate')
    model_at_neural = interp_func(neural_data["orientation_differences"])
    
    similarity = compute_similarity_metrics(
        model_at_neural,
        neural_data["normalized_response"],
        neural_data["response_se"]
    )
    
    # Analyze surround modulation
    analyzer = SurroundModulationAnalyzer()
    suppression_indices = []
    for resp in responses:
        si = analyzer.compute_suppression_index(center_only_resp, resp)
        suppression_indices.append(si)
    
    # Save results
    results = {
        "orientation_differences": ori_diffs,
        "model_normalized_responses": normalized_responses.tolist(),
        "model_suppression_indices": suppression_indices,
        "neural_data": {
            "orientation_differences": neural_data["orientation_differences"].tolist(),
            "normalized_responses": neural_data["normalized_response"].tolist()
        },
        "similarity_metrics": similarity
    }
    
    # Plot
    fig = plot_spatial_interactions({
        "orientation_tuning": {
            "orientation_differences": np.array(ori_diffs),
            "normalized_responses": normalized_responses,
            "model_responses": model_at_neural
        }
    }, title="Kinoshita Surround Modulation",
    save_path=output_dir / "kinoshita_surround.png")
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Setup
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create response extractor
    extractor = ResponseExtractor(model)
    
    # Determine target layer
    if args.layer:
        target_layer = args.layer
    else:
        # Auto-detect V1-like layers
        v1_layers = get_v1_like_layers(model)
        target_layer = v1_layers[1]  # Use second encoder layer
        print(f"Using layer: {target_layer}")
    
    # Run experiments
    all_results = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "target_layer": target_layer,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if args.experiment in ['all', 'orientation']:
        all_results['orientation_tuning'] = run_orientation_tuning(
            model, extractor, target_layer, device, output_dir
        )
    
    if args.experiment in ['all', 'contrast']:
        all_results['contrast_response'] = run_contrast_response(
            model, extractor, target_layer, device, output_dir
        )
    
    if args.experiment in ['all', 'kapadia']:
        all_results['kapadia'] = run_kapadia_experiment(
            model, extractor, target_layer, device, output_dir
        )
    
    if args.experiment in ['all', 'kinoshita']:
        all_results['kinoshita'] = run_kinoshita_experiment(
            model, extractor, target_layer, device, output_dir
        )
    
    # Save all results
    results_file = output_dir / f"insilico_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create summary figure if running all experiments
    if args.experiment == 'all':
        # Prepare data for summary figure
        summary_data = {}
        
        if 'orientation_tuning' in all_results:
            summary_data['orientation_tuning'] = {
                'orientations': all_results['orientation_tuning']['orientations'],
                'responses': all_results['orientation_tuning']['responses']
            }
        
        if 'contrast_response' in all_results:
            summary_data['contrast_response'] = {
                'contrasts': all_results['contrast_response']['contrasts'],
                'responses': all_results['contrast_response']['responses']
            }
        
        if 'kapadia' in all_results:
            summary_data['collinear_facilitation'] = {
                'distances': all_results['kapadia']['distances'],
                'facilitation': all_results['kapadia']['model_facilitation_collinear']
            }
        
        if 'kinoshita' in all_results:
            summary_data['surround_modulation'] = {
                'ori_diff': all_results['kinoshita']['orientation_differences'],
                'suppression': all_results['kinoshita']['model_suppression_indices']
            }
        
        # Add summary statistics
        summary_data['summary_stats'] = {
            'orientation_bandwidth': all_results.get('orientation_tuning', {}).get(
                'tuning_curve', {}).get('bandwidth', 0),
            'contrast_c50': all_results.get('contrast_response', {}).get(
                'contrast_response_fit', {}).get('c50', 0),
            'kapadia_correlation': all_results.get('kapadia', {}).get(
                'similarity_metrics', {}).get('correlation', 0),
            'kinoshita_correlation': all_results.get('kinoshita', {}).get(
                'similarity_metrics', {}).get('correlation', 0)
        }
        
        fig = create_summary_figure(
            summary_data,
            save_path=output_dir / "insilico_summary.png"
        )
    
    # Clean up
    extractor.clear_hooks()
    
    print("\nIn silico experiments completed!")


if __name__ == '__main__':
    main()