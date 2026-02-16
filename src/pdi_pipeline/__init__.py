"""PDI Pipeline - Entropy-guided gap-filling for satellite imagery.

Public API exports for the core pipeline library.
"""

from pdi_pipeline.exceptions import (
    ConfigError,
    ConvergenceError,
    DimensionError,
    InsufficientDataError,
    InterpolationError,
    PDIError,
    ValidationError,
)
from pdi_pipeline.methods.base import BaseMethod
from pdi_pipeline.methods.registry import get_interpolator, list_methods

__all__ = [
    "BaseMethod",
    "ConfigError",
    "ConvergenceError",
    "DimensionError",
    "InsufficientDataError",
    "InterpolationError",
    "PDIError",
    "ValidationError",
    "get_interpolator",
    "list_methods",
]
