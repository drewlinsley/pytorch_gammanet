"""GammaNet models and components."""

from .gammanet import GammaNet
from .gammanet_backbone import GammaNetBackbone
from .vgg16_gammanet import VGG16GammaNet
from .vgg16_gammanet_v2 import VGG16GammaNetV2  # New v2 model
from .components.fgru import fGRU
from .components.fgru_v2 import fGRUv2  # New v2 fGRU
from .ablations import (
    # Connectivity ablations
    GammaNetFFOnly,
    GammaNetHOnly,
    GammaNetTDOnly,
    GammaNetNoRecurrence,
    GammaNetBottomUpOnly,
    GammaNetDelayedTD,
    
    # Gating ablations
    GammaNetAdditiveOnly,
    GammaNetMultiplicativeOnly,
    GammaNetNoGates,
    GammaNetNoDivisive,
    GammaNetLinearGates,
    GammaNetSymmetricGates,
    
    # Registry
    ABLATION_REGISTRY,
    get_ablation_model
)

__all__ = [
    "GammaNet",
    "GammaNetBackbone",
    "VGG16GammaNet",
    "VGG16GammaNetV2",
    "fGRU",
    "fGRUv2",
    # Connectivity ablations
    "GammaNetFFOnly",
    "GammaNetHOnly", 
    "GammaNetTDOnly",
    "GammaNetNoRecurrence",
    "GammaNetBottomUpOnly",
    "GammaNetDelayedTD",
    # Gating ablations
    "GammaNetAdditiveOnly",
    "GammaNetMultiplicativeOnly",
    "GammaNetNoGates",
    "GammaNetNoDivisive",
    "GammaNetLinearGates",
    "GammaNetSymmetricGates",
    # Registry
    "ABLATION_REGISTRY",
    "get_ablation_model"
]