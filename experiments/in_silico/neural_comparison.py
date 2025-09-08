"""Neural data comparison utilities for in silico experiments.

This module provides functions to load empirical neural data and compare
model responses with primate neurophysiology.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import json
import warnings


def load_kapadia_data() -> Dict[str, np.ndarray]:
    """Load digitized Kapadia et al. (1995) collinear facilitation data.
    
    Returns:
        Dictionary with experimental data arrays
    """
    # Digitized data from Kapadia et al. 1995, Figure 5
    # Shows facilitation as a function of flanker distance
    
    data = {
        # Flanker distances in degrees of visual angle
        "distances": np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        
        # Facilitation index (response with flankers / response alone - 1)
        "facilitation_collinear": np.array([0.15, 0.45, 0.65, 0.55, 0.35, 0.15, 0.05, 0.0]),
        
        # Standard errors
        "facilitation_se": np.array([0.05, 0.08, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]),
        
        # Orthogonal flankers (control)
        "facilitation_orthogonal": np.array([0.0, -0.05, -0.10, -0.08, -0.05, -0.02, 0.0, 0.0]),
        
        # Parameters from paper
        "target_contrast": 0.1,  # 10% contrast
        "flanker_contrast": 0.5,  # 50% contrast
        "bar_length": 0.2,  # degrees visual angle
        "bar_width": 0.05,  # degrees visual angle
    }
    
    return data


def load_kinoshita_data() -> Dict[str, np.ndarray]:
    """Load digitized Kinoshita & Gilbert (2008) surround modulation data.
    
    Returns:
        Dictionary with experimental data arrays
    """
    # Digitized data showing surround suppression as function of 
    # orientation difference between center and surround
    
    data = {
        # Orientation differences (degrees)
        "orientation_differences": np.array([0, 15, 30, 45, 60, 75, 90]),
        
        # Normalized responses (center+surround / center alone)
        "normalized_response": np.array([0.45, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0]),
        
        # Standard errors
        "response_se": np.array([0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06]),
        
        # Size tuning data
        "stimulus_sizes": np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  # degrees
        "size_tuning": np.array([0.3, 0.7, 1.0, 0.8, 0.6, 0.5, 0.45]),
        
        # Parameters
        "center_size": 1.0,  # Optimal center size in degrees
        "contrast": 0.5,
    }
    
    return data


def load_trott_born_data() -> Dict[str, np.ndarray]:
    """Load texture boundary response data (simplified version).
    
    Returns:
        Dictionary with boundary detection performance
    """
    # Simplified data based on texture boundary experiments
    data = {
        # Orientation differences across boundary
        "orientation_differences": np.array([0, 15, 30, 45, 60, 75, 90]),
        
        # Detection performance (d-prime)
        "detection_performance": np.array([0.0, 0.5, 1.2, 2.0, 2.5, 2.8, 3.0]),
        
        # Neural modulation index
        "boundary_modulation": np.array([0.0, 0.1, 0.25, 0.45, 0.60, 0.70, 0.75]),
        
        # Standard errors
        "performance_se": np.array([0.1, 0.15, 0.2, 0.2, 0.18, 0.15, 0.12]),
    }
    
    return data


def fit_encoding_model(model_responses: np.ndarray,
                      neural_responses: np.ndarray,
                      n_components: int = 10) -> Dict:
    """Fit PLS regression between model and neural responses.
    
    Args:
        model_responses: Model responses [n_stimuli, n_model_units]
        neural_responses: Neural responses [n_stimuli, n_neurons]
        n_components: Number of PLS components
        
    Returns:
        Dictionary with fit results
    """
    # Ensure proper shapes
    if model_responses.ndim == 1:
        model_responses = model_responses.reshape(-1, 1)
    if neural_responses.ndim == 1:
        neural_responses = neural_responses.reshape(-1, 1)
        
    # Fit PLS regression
    pls = PLSRegression(n_components=min(n_components, model_responses.shape[1]))
    
    try:
        pls.fit(model_responses, neural_responses)
        
        # Predict neural responses
        neural_pred = pls.predict(model_responses)
        
        # Calculate R-squared for each neuron
        if neural_responses.shape[1] == 1:
            r2 = r2_score(neural_responses, neural_pred)
        else:
            r2 = np.array([
                r2_score(neural_responses[:, i], neural_pred[:, i])
                for i in range(neural_responses.shape[1])
            ])
            
        # Get loadings
        loadings = pls.x_loadings_
        
        results = {
            "r2": r2,
            "mean_r2": np.mean(r2) if isinstance(r2, np.ndarray) else r2,
            "loadings": loadings,
            "n_components": pls.n_components,
            "pls_model": pls
        }
        
    except Exception as e:
        warnings.warn(f"PLS fitting failed: {e}")
        results = {
            "r2": 0,
            "mean_r2": 0,
            "loadings": None,
            "n_components": 0,
            "pls_model": None
        }
        
    return results


def compute_similarity_metrics(model_data: np.ndarray,
                             neural_data: np.ndarray,
                             error_bars: Optional[np.ndarray] = None) -> Dict:
    """Compute similarity metrics between model and neural data.
    
    Args:
        model_data: Model responses
        neural_data: Neural responses
        error_bars: Standard errors for neural data
        
    Returns:
        Dictionary with similarity metrics
    """
    # Ensure same length
    min_len = min(len(model_data), len(neural_data))
    model_data = model_data[:min_len]
    neural_data = neural_data[:min_len]
    
    # Normalize both to [0, 1] for fair comparison
    if model_data.max() > model_data.min():
        model_norm = (model_data - model_data.min()) / (model_data.max() - model_data.min())
    else:
        model_norm = model_data
        
    if neural_data.max() > neural_data.min():
        neural_norm = (neural_data - neural_data.min()) / (neural_data.max() - neural_data.min())
    else:
        neural_norm = neural_data
    
    # Compute metrics
    metrics = {}
    
    # Pearson correlation
    if len(model_norm) > 2:
        r, p_value = stats.pearsonr(model_norm, neural_norm)
        metrics["correlation"] = r
        metrics["p_value"] = p_value
    else:
        metrics["correlation"] = 0
        metrics["p_value"] = 1
    
    # R-squared
    metrics["r_squared"] = r2_score(neural_norm, model_norm)
    
    # Root mean squared error
    metrics["rmse"] = np.sqrt(np.mean((model_norm - neural_norm) ** 2))
    
    # Mean absolute error
    metrics["mae"] = np.mean(np.abs(model_norm - neural_norm))
    
    # Chi-squared if error bars provided
    if error_bars is not None and np.all(error_bars > 0):
        chi2 = np.sum(((model_data - neural_data) / error_bars) ** 2)
        metrics["chi_squared"] = chi2
        metrics["reduced_chi_squared"] = chi2 / (len(model_data) - 1)
    
    return metrics


def compare_tuning_properties(model_tuning: Dict,
                            neural_tuning: Dict) -> Dict:
    """Compare tuning properties between model and neural data.
    
    Args:
        model_tuning: Dictionary with model tuning properties
        neural_tuning: Dictionary with neural tuning properties
        
    Returns:
        Comparison metrics
    """
    comparison = {}
    
    # Compare preferred orientations
    if "preferred_orientation" in model_tuning and "preferred_orientation" in neural_tuning:
        ori_diff = np.abs(model_tuning["preferred_orientation"] - 
                         neural_tuning["preferred_orientation"])
        # Handle circular difference
        ori_diff = min(ori_diff, 180 - ori_diff)
        comparison["orientation_difference"] = ori_diff
    
    # Compare bandwidths
    if "bandwidth" in model_tuning and "bandwidth" in neural_tuning:
        comparison["bandwidth_ratio"] = model_tuning["bandwidth"] / neural_tuning["bandwidth"]
    
    # Compare modulation indices
    if "modulation_index" in model_tuning and "modulation_index" in neural_tuning:
        comparison["modulation_index_difference"] = (
            model_tuning["modulation_index"] - neural_tuning["modulation_index"]
        )
    
    # Compare selectivity
    if "selectivity_index" in model_tuning and "selectivity_index" in neural_tuning:
        comparison["selectivity_correlation"] = stats.pearsonr(
            model_tuning["selectivity_index"],
            neural_tuning["selectivity_index"]
        )[0]
    
    return comparison


def save_comparison_results(results: Dict,
                          output_path: Path,
                          experiment_name: str) -> None:
    """Save comparison results to file.
    
    Args:
        results: Dictionary with all comparison results
        output_path: Directory to save results
        experiment_name: Name of the experiment
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / f"{experiment_name}_comparison.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            json_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save summary as text
    summary_path = output_path / f"{experiment_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Neural Comparison Summary: {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        if "similarity_metrics" in results:
            f.write("Similarity Metrics:\n")
            for metric, value in results["similarity_metrics"].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
        
        if "encoding_model" in results:
            f.write("Encoding Model Results:\n")
            f.write(f"  Mean R²: {results['encoding_model']['mean_r2']:.4f}\n")
            f.write(f"  N components: {results['encoding_model']['n_components']}\n")
            f.write("\n")
        
        if "tuning_comparison" in results:
            f.write("Tuning Property Comparison:\n")
            for prop, value in results["tuning_comparison"].items():
                f.write(f"  {prop}: {value:.4f}\n")


def create_model_neural_alignment_report(
    model_responses: Dict[str, np.ndarray],
    neural_datasets: Dict[str, Dict],
    output_path: Path
) -> Dict:
    """Create comprehensive alignment report between model and neural data.
    
    Args:
        model_responses: Dictionary of model responses for each experiment
        neural_datasets: Dictionary of neural datasets
        output_path: Where to save report
        
    Returns:
        Summary statistics
    """
    report = {
        "experiments": {},
        "overall_alignment": {}
    }
    
    # Analyze each experiment
    for exp_name, model_data in model_responses.items():
        if exp_name not in neural_datasets:
            continue
            
        neural_data = neural_datasets[exp_name]
        
        # Compute similarity
        similarity = compute_similarity_metrics(
            model_data,
            neural_data.get("responses", neural_data.get("values"))
        )
        
        report["experiments"][exp_name] = {
            "similarity": similarity,
            "n_conditions": len(model_data)
        }
    
    # Compute overall alignment score
    all_correlations = [
        exp["similarity"]["correlation"] 
        for exp in report["experiments"].values()
        if "correlation" in exp["similarity"]
    ]
    
    if all_correlations:
        report["overall_alignment"] = {
            "mean_correlation": np.mean(all_correlations),
            "std_correlation": np.std(all_correlations),
            "n_experiments": len(all_correlations)
        }
    
    # Save report
    save_comparison_results(report, output_path, "overall_alignment")
    
    return report