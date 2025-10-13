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


def convert_to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    else:
        return obj

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gammanet.models import GammaNet, VGG16GammaNet, VGG16GammaNetV2
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
    plot_contrast_response,
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Determine which model to use based on config
    model_config = config['model']
    use_backbone = model_config.get('use_backbone', False)
    model_version = model_config.get('model_version', 'v1')

    # Create appropriate model
    if use_backbone:
        if model_version == 'v2':
            model = VGG16GammaNetV2(config=model_config)
        else:
            model = VGG16GammaNet(config=model_config)
    else:
        model = GammaNet(
            config=model_config,
            input_channels=3,
            output_channels=1
        )

    # Load weights (filter out hidden state buffers)
    state_dict = checkpoint['model_state_dict']
    # Remove hidden state buffers that were saved during training
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if not (k.startswith('h_block') or k.startswith('h0_') or
                                   k.startswith('h1_') or k.startswith('h2_') or
                                   k.startswith('h3_') or k.startswith('td_h'))}

    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


def run_orientation_tuning(model, extractor, target_layer, device, output_dir):
    """Run orientation tuning experiment."""
    print("\nRunning orientation tuning experiment...")
    
    # Generate stimuli with multiple biologically plausible spatial frequencies
    stim_generator = OrientedGratingStimuli()
    orientations = np.arange(0, 180, 15)
    # Test multiple spatial frequencies (cycles per image)
    # This corresponds to 0.04-0.20 cycles/pixel for 256x256 images
    spatial_frequencies = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

    all_responses = []
    for sf in spatial_frequencies:
        stimuli = stim_generator.generate_stimulus_set(
            orientations=orientations,
            spatial_frequencies=[sf],
            contrasts=[0.5]
        )

        # Create batch
        stim_batch, metadata = create_stimulus_batch(stimuli)
        stim_batch = stim_batch.to(device)

        # Reset hidden states for clean forward pass
        model.reset_hidden_states()

        # Extract responses
        responses = extractor.extract_responses(stim_batch)

        # Get population response
        pop_response = extractor.get_population_response(
            responses, target_layer, spatial_pool="center"
        )

        # Average across neurons and timesteps
        mean_response = pop_response.mean(axis=(1, 2))
        all_responses.append((sf, mean_response))

    # Find optimal spatial frequency (one with highest variance/selectivity)
    best_sf = None
    best_variance = 0
    best_responses = None

    for sf, resp in all_responses:
        variance = np.var(resp)
        if variance > best_variance:
            best_variance = variance
            best_sf = sf
            best_responses = resp

    print(f"Optimal spatial frequency: {best_sf} cycles/image")

    # Use best responses for tuning curve analysis
    mean_response = best_responses
    
    # Analyze tuning
    analyzer = OrientationTuningAnalyzer()
    tuning_curve = analyzer.fit_tuning_curve(orientations, mean_response)
    
    # Save results (convert numpy types to Python types for JSON)
    results = {
        "orientations": orientations.tolist(),
        "responses": mean_response.tolist(),
        "optimal_spatial_frequency": float(best_sf),
        "spatial_frequency_responses": {
            str(sf): resp.tolist() for sf, resp in all_responses
        },
        "tuning_curve": {
            "preferred_orientation": float(tuning_curve.preferred_value) if tuning_curve.preferred_value is not None else None,
            "bandwidth": float(tuning_curve.bandwidth) if tuning_curve.bandwidth is not None else None,
            "r_squared": float(tuning_curve.r_squared) if tuning_curve.r_squared is not None else None,
            "modulation_index": float(tuning_curve.modulation_index) if tuning_curve.modulation_index is not None else None
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
    
    # Reset hidden states for clean forward pass
    model.reset_hidden_states()

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
            "c50": float(crf.fit_params["c50"]) if crf.fit_params and "c50" in crf.fit_params else None,
            "r_max": float(crf.fit_params["r_max"]) if crf.fit_params and "r_max" in crf.fit_params else None,
            "n": float(crf.fit_params["n"]) if crf.fit_params and "n" in crf.fit_params else None,
            "r_squared": float(crf.r_squared) if crf.r_squared is not None else None
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

        # Reset hidden states for clean forward pass
        model.reset_hidden_states()

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
        "similarity_metrics": convert_to_json_serializable(similarity)
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

        # Reset hidden states for clean forward pass
        model.reset_hidden_states()

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
        "model_suppression_indices": [float(si) for si in suppression_indices],
        "neural_data": {
            "orientation_differences": neural_data["orientation_differences"].tolist(),
            "normalized_responses": neural_data["normalized_response"].tolist()
        },
        "similarity_metrics": convert_to_json_serializable(similarity)
    }
    
    # Plot
    fig = plot_spatial_interactions({
        "orientation_tuning": {
            "orientation_differences": np.array(ori_diffs),
            "normalized_responses": normalized_responses,
        },
        "neural_comparison": {
            "orientation_differences": neural_data["orientation_differences"],
            "model_responses": model_at_neural,
            "neural_responses": neural_data["normalized_response"]
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
        target_layer = v1_layers[1] if len(v1_layers) > 1 else v1_layers[0]

    print(f"Using layer: {target_layer}")

    # Register hooks for the target layer
    extractor.register_hooks([target_layer])
    
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