"""In silico neurophysiology experiments for GammaNet.

This package provides tools for running computational experiments that
replicate classic neurophysiology studies, allowing comparison between
GammaNet responses and primate visual cortex data.
"""

from .stimuli import (
    OrientedGratingStimuli,
    KapadiaStimuli,
    KinoshitaStimuli,
    TextureBoundaryStimuli,
    TiltIllusionStimuli,
    create_stimulus_batch
)

from .extract import (
    ResponseExtractor,
    extract_layer_responses,
    get_v1_like_layers
)

from .analysis import (
    OrientationTuningAnalyzer,
    ContrastResponseAnalyzer,
    SurroundModulationAnalyzer,
    PopulationCodingAnalyzer,
    TuningCurve
)

from .neural_comparison import (
    load_kapadia_data,
    load_kinoshita_data,
    load_trott_born_data,
    fit_encoding_model,
    compute_similarity_metrics,
    compare_tuning_properties,
    create_model_neural_alignment_report
)

from .visualize import (
    plot_tuning_curves,
    plot_contrast_response,
    plot_spatial_interactions,
    plot_model_neural_comparison,
    plot_ablation_comparison,
    create_summary_figure
)

__all__ = [
    # Stimuli
    'OrientedGratingStimuli',
    'KapadiaStimuli', 
    'KinoshitaStimuli',
    'TextureBoundaryStimuli',
    'TiltIllusionStimuli',
    'create_stimulus_batch',
    
    # Response extraction
    'ResponseExtractor',
    'extract_layer_responses',
    'get_v1_like_layers',
    
    # Analysis
    'OrientationTuningAnalyzer',
    'ContrastResponseAnalyzer',
    'SurroundModulationAnalyzer',
    'PopulationCodingAnalyzer',
    'TuningCurve',
    
    # Neural comparison
    'load_kapadia_data',
    'load_kinoshita_data',
    'load_trott_born_data',
    'fit_encoding_model',
    'compute_similarity_metrics',
    'compare_tuning_properties',
    'create_model_neural_alignment_report',
    
    # Visualization
    'plot_tuning_curves',
    'plot_contrast_response',
    'plot_spatial_interactions',
    'plot_model_neural_comparison',
    'plot_ablation_comparison',
    'create_summary_figure'
]