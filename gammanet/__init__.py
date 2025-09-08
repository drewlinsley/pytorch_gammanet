"""PyTorch GammaNet - Recurrent neural networks inspired by cortical circuits."""

__version__ = "0.1.0"

from .models import GammaNet, fGRU

__all__ = ["GammaNet", "fGRU"]