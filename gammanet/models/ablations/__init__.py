"""Model ablations for studying GammaNet components.

This package provides ablated versions of GammaNet to study the contribution
of different architectural components and mechanisms.
"""

from .base import (
    AblationMixin,
    ParameterAblationMixin,
    ConnectivityAblationMixin,
    TimestepAblationMixin
)

from .connectivity import (
    GammaNetFFOnly,
    GammaNetHOnly,
    GammaNetTDOnly,
    GammaNetNoRecurrence,
    GammaNetBottomUpOnly,
    GammaNetDelayedTD
)

from .gating import (
    GammaNetAdditiveOnly,
    GammaNetMultiplicativeOnly,
    GammaNetNoGates,
    GammaNetNoDivisive,
    GammaNetLinearGates,
    GammaNetSymmetricGates
)

# Ablation registry for easy access
ABLATION_REGISTRY = {
    # Connectivity ablations
    "ffonly": GammaNetFFOnly,
    "honly": GammaNetHOnly,
    "tdonly": GammaNetTDOnly,
    "no_recurrence": GammaNetNoRecurrence,
    "bottom_up_only": GammaNetBottomUpOnly,
    "delayed_td": GammaNetDelayedTD,
    
    # Gating ablations
    "additive_only": GammaNetAdditiveOnly,
    "multiplicative_only": GammaNetMultiplicativeOnly,
    "no_gates": GammaNetNoGates,
    "no_divisive": GammaNetNoDivisive,
    "linear_gates": GammaNetLinearGates,
    "symmetric_gates": GammaNetSymmetricGates
}


def get_ablation_model(ablation_name: str):
    """Get ablation model class by name.
    
    Args:
        ablation_name: Name of the ablation
        
    Returns:
        Ablation model class
        
    Raises:
        ValueError: If ablation name not found
    """
    if ablation_name not in ABLATION_REGISTRY:
        available = ", ".join(ABLATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown ablation: {ablation_name}. "
            f"Available ablations: {available}"
        )
    
    return ABLATION_REGISTRY[ablation_name]


__all__ = [
    # Base classes
    'AblationMixin',
    'ParameterAblationMixin',
    'ConnectivityAblationMixin',
    'TimestepAblationMixin',
    
    # Connectivity ablations
    'GammaNetFFOnly',
    'GammaNetHOnly',
    'GammaNetTDOnly',
    'GammaNetNoRecurrence',
    'GammaNetBottomUpOnly',
    'GammaNetDelayedTD',
    
    # Gating ablations
    'GammaNetAdditiveOnly',
    'GammaNetMultiplicativeOnly',
    'GammaNetNoGates',
    'GammaNetNoDivisive',
    'GammaNetLinearGates',
    'GammaNetSymmetricGates',
    
    # Registry
    'ABLATION_REGISTRY',
    'get_ablation_model'
]