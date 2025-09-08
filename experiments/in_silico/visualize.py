"""Visualization utilities for in silico experiments.

This module provides functions to create publication-quality figures
for neurophysiology experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_tuning_curves(tuning_data: List[Dict],
                      labels: Optional[List[str]] = None,
                      title: str = "Orientation Tuning",
                      save_path: Optional[Path] = None) -> plt.Figure:
    """Plot orientation tuning curves with fits.
    
    Args:
        tuning_data: List of dictionaries with x_values, y_values, y_err, fit
        labels: Labels for each curve
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(tuning_data)))
    
    for i, data in enumerate(tuning_data):
        label = labels[i] if labels else f"Condition {i+1}"
        color = colors[i]
        
        # Plot data points
        x = data['x_values']
        y = data['y_values']
        
        ax.scatter(x, y, color=color, s=60, alpha=0.7, label=label)
        
        # Plot error bars if available
        if 'y_err' in data and data['y_err'] is not None:
            ax.errorbar(x, y, yerr=data['y_err'], fmt='none', 
                       color=color, alpha=0.5, capsize=3)
        
        # Plot fit if available
        if 'fit' in data and data['fit'] is not None:
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = data['fit'](x_fit)
            ax.plot(x_fit, y_fit, color=color, linewidth=2, alpha=0.8)
            
        # Add preferred orientation marker
        if 'preferred' in data:
            ax.axvline(data['preferred'], color=color, linestyle='--', 
                      alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Orientation (degrees)', fontsize=12)
    ax.set_ylabel('Response', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits and ticks
    ax.set_xlim(-10, 190)
    ax.set_xticks(np.arange(0, 181, 45))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_contrast_response(crf_data: List[Dict],
                         labels: Optional[List[str]] = None,
                         title: str = "Contrast Response Function",
                         save_path: Optional[Path] = None) -> plt.Figure:
    """Plot contrast response functions.
    
    Args:
        crf_data: List of CRF data dictionaries
        labels: Labels for each curve
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(crf_data)))
    
    for i, data in enumerate(crf_data):
        label = labels[i] if labels else f"Condition {i+1}"
        color = colors[i]
        
        # Plot data
        contrasts = data['contrasts']
        responses = data['responses']
        
        ax.scatter(contrasts * 100, responses, color=color, s=60, 
                  alpha=0.7, label=label)
        
        # Plot fit if available
        if 'fit' in data and data['fit'] is not None:
            c_fit = np.logspace(-2, 0, 50)
            r_fit = data['fit'](c_fit)
            ax.plot(c_fit * 100, r_fit, color=color, linewidth=2)
            
        # Mark C50 if available
        if 'c50' in data:
            ax.axvline(data['c50'] * 100, color=color, linestyle=':', 
                      alpha=0.5)
    
    ax.set_xlabel('Contrast (%)', fontsize=12)
    ax.set_ylabel('Response', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, 100)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_spatial_interactions(interaction_data: Dict,
                            title: str = "Spatial Interactions",
                            save_path: Optional[Path] = None) -> plt.Figure:
    """Plot spatial interaction effects (e.g., surround suppression).
    
    Args:
        interaction_data: Dictionary with interaction data
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Orientation difference tuning
    ax1 = fig.add_subplot(gs[0, 0])
    if 'orientation_tuning' in interaction_data:
        data = interaction_data['orientation_tuning']
        ori_diff = data['orientation_differences']
        responses = data['normalized_responses']
        
        ax1.plot(ori_diff, responses, 'o-', markersize=8, linewidth=2)
        if 'model_responses' in data:
            ax1.plot(ori_diff, data['model_responses'], 's--', 
                    markersize=6, linewidth=2, label='Model')
            ax1.legend()
            
        ax1.set_xlabel('Center-Surround Orientation Difference (deg)')
        ax1.set_ylabel('Normalized Response')
        ax1.set_title('Surround Orientation Tuning')
        ax1.grid(True, alpha=0.3)
    
    # 2. Size tuning
    ax2 = fig.add_subplot(gs[0, 1])
    if 'size_tuning' in interaction_data:
        data = interaction_data['size_tuning']
        sizes = data['sizes']
        responses = data['responses']
        
        ax2.plot(sizes, responses, 'o-', markersize=8, linewidth=2)
        
        # Mark optimal size
        if 'optimal_size' in data:
            ax2.axvline(data['optimal_size'], color='red', 
                       linestyle='--', alpha=0.5)
            
        ax2.set_xlabel('Stimulus Size (deg)')
        ax2.set_ylabel('Response')
        ax2.set_title('Size Tuning')
        ax2.grid(True, alpha=0.3)
    
    # 3. Collinear facilitation
    ax3 = fig.add_subplot(gs[1, 0])
    if 'collinear_facilitation' in interaction_data:
        data = interaction_data['collinear_facilitation']
        distances = data['distances']
        facilitation = data['facilitation']
        
        ax3.plot(distances, facilitation * 100, 'o-', markersize=8, 
                linewidth=2, label='Collinear')
        
        if 'orthogonal' in data:
            ax3.plot(distances, data['orthogonal'] * 100, 's--', 
                    markersize=6, linewidth=2, label='Orthogonal', alpha=0.6)
            
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Flanker Distance (deg)')
        ax3.set_ylabel('Facilitation (%)')
        ax3.set_title('Collinear Facilitation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    if 'summary_matrix' in interaction_data:
        matrix = interaction_data['summary_matrix']
        im = ax4.imshow(matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax4)
        ax4.set_title('Interaction Summary')
        
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_model_neural_comparison(model_data: np.ndarray,
                                neural_data: np.ndarray,
                                error_bars: Optional[np.ndarray] = None,
                                title: str = "Model vs Neural Data",
                                xlabel: str = "Neural Response",
                                ylabel: str = "Model Response",
                                save_path: Optional[Path] = None) -> plt.Figure:
    """Create scatter plot comparing model and neural responses.
    
    Args:
        model_data: Model responses
        neural_data: Neural responses
        error_bars: Error bars for neural data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize data for plotting
    model_norm = (model_data - model_data.min()) / (model_data.max() - model_data.min())
    neural_norm = (neural_data - neural_data.min()) / (neural_data.max() - neural_data.min())
    
    # Scatter plot
    ax.scatter(neural_norm, model_norm, s=80, alpha=0.6, edgecolors='black', 
              linewidth=1)
    
    # Add error bars if provided
    if error_bars is not None:
        error_norm = error_bars / (neural_data.max() - neural_data.min())
        ax.errorbar(neural_norm, model_norm, xerr=error_norm, fmt='none', 
                   alpha=0.3, capsize=0)
    
    # Add unity line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Unity')
    
    # Fit linear regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(neural_norm.reshape(-1, 1), model_norm)
    x_fit = np.linspace(0, 1, 100)
    y_fit = reg.predict(x_fit.reshape(-1, 1))
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, label='Linear fit')
    
    # Calculate and display R²
    from sklearn.metrics import r2_score
    r2 = r2_score(neural_norm, model_norm)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_ablation_comparison(ablation_results: Dict[str, Dict],
                           metric: str = "correlation",
                           title: str = "Ablation Study Results",
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Plot comparison of different ablation models.
    
    Args:
        ablation_results: Dictionary mapping model names to results
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    model_names = list(ablation_results.keys())
    experiments = list(ablation_results[model_names[0]].keys())
    
    # Create grouped bar plot
    x = np.arange(len(experiments))
    width = 0.8 / len(model_names)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, model in enumerate(model_names):
        values = [ablation_results[model][exp].get(metric, 0) 
                 for exp in experiments]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i])
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add baseline if comparing to full model
    if 'Full Model' in model_names:
        full_idx = model_names.index('Full Model')
        baseline_values = [ablation_results['Full Model'][exp].get(metric, 0) 
                          for exp in experiments]
        ax.axhline(np.mean(baseline_values), color='black', 
                  linestyle='--', alpha=0.5, label='Full Model Mean')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def create_summary_figure(all_results: Dict,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """Create comprehensive summary figure with multiple panels.
    
    Args:
        all_results: Dictionary with all experimental results
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Orientation tuning
    ax_a = fig.add_subplot(gs[0, 0])
    if 'orientation_tuning' in all_results:
        data = all_results['orientation_tuning']
        ax_a.plot(data['orientations'], data['responses'], 'o-', linewidth=2)
        ax_a.set_xlabel('Orientation (deg)')
        ax_a.set_ylabel('Response')
        ax_a.set_title('A. Orientation Tuning')
        ax_a.grid(True, alpha=0.3)
    
    # Panel B: Contrast response
    ax_b = fig.add_subplot(gs[0, 1])
    if 'contrast_response' in all_results:
        data = all_results['contrast_response']
        ax_b.semilogx(data['contrasts'] * 100, data['responses'], 'o-', linewidth=2)
        ax_b.set_xlabel('Contrast (%)')
        ax_b.set_ylabel('Response')
        ax_b.set_title('B. Contrast Response')
        ax_b.grid(True, alpha=0.3, which='both')
    
    # Panel C: Surround modulation
    ax_c = fig.add_subplot(gs[0, 2])
    if 'surround_modulation' in all_results:
        data = all_results['surround_modulation']
        ax_c.plot(data['ori_diff'], data['suppression'], 'o-', linewidth=2)
        ax_c.set_xlabel('Center-Surround Ori Diff (deg)')
        ax_c.set_ylabel('Suppression Index')
        ax_c.set_title('C. Surround Modulation')
        ax_c.grid(True, alpha=0.3)
    
    # Panel D: Collinear facilitation
    ax_d = fig.add_subplot(gs[1, 0])
    if 'collinear_facilitation' in all_results:
        data = all_results['collinear_facilitation']
        ax_d.plot(data['distances'], data['facilitation'], 'o-', linewidth=2)
        ax_d.set_xlabel('Flanker Distance')
        ax_d.set_ylabel('Facilitation')
        ax_d.set_title('D. Collinear Facilitation')
        ax_d.grid(True, alpha=0.3)
    
    # Panel E: Model-neural correlation
    ax_e = fig.add_subplot(gs[1, 1])
    if 'model_neural_correlation' in all_results:
        data = all_results['model_neural_correlation']
        ax_e.scatter(data['neural'], data['model'], alpha=0.6)
        ax_e.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_e.set_xlabel('Neural Response')
        ax_e.set_ylabel('Model Response')
        ax_e.set_title('E. Model-Neural Correlation')
        ax_e.grid(True, alpha=0.3)
    
    # Panel F: Ablation results
    ax_f = fig.add_subplot(gs[1, 2])
    if 'ablation_summary' in all_results:
        data = all_results['ablation_summary']
        models = list(data.keys())
        scores = [data[m]['mean_score'] for m in models]
        ax_f.bar(models, scores)
        ax_f.set_ylabel('Performance')
        ax_f.set_title('F. Ablation Results')
        ax_f.tick_params(axis='x', rotation=45)
    
    # Panel G: Temporal dynamics
    ax_g = fig.add_subplot(gs[2, :2])
    if 'temporal_dynamics' in all_results:
        data = all_results['temporal_dynamics']
        timesteps = np.arange(len(data['responses']))
        ax_g.plot(timesteps, data['responses'], linewidth=2)
        ax_g.set_xlabel('Timestep')
        ax_g.set_ylabel('Response')
        ax_g.set_title('G. Temporal Dynamics')
        ax_g.grid(True, alpha=0.3)
    
    # Panel H: Summary statistics
    ax_h = fig.add_subplot(gs[2, 2])
    ax_h.axis('off')
    if 'summary_stats' in all_results:
        stats_text = "Summary Statistics:\n\n"
        for key, value in all_results['summary_stats'].items():
            stats_text += f"{key}: {value:.3f}\n"
        ax_h.text(0.1, 0.9, stats_text, transform=ax_h.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('In Silico Neurophysiology Results', fontsize=16, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig