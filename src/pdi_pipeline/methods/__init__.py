"""Reconstruction methods for image inpainting and interpolation."""

from pdi_pipeline.methods.base import BaseMethod
from pdi_pipeline.methods.bicubic import BicubicInterpolator
from pdi_pipeline.methods.bilinear import BilinearInterpolator
from pdi_pipeline.methods.compressive import (
    L1DCTInpainting,
    L1WaveletInpainting,
)
from pdi_pipeline.methods.idw import IDWInterpolator
from pdi_pipeline.methods.kriging import KrigingInterpolator
from pdi_pipeline.methods.lanczos import LanczosInterpolator
from pdi_pipeline.methods.multi_temporal import (
    SpaceTimeKriging,
    TemporalFourierInterpolator,
    TemporalSplineInterpolator,
)
from pdi_pipeline.methods.nearest import NearestInterpolator
from pdi_pipeline.methods.patch_based import (
    ExemplarBasedInterpolator,
    NonLocalMeansInterpolator,
)
from pdi_pipeline.methods.rbf import RBFInterpolator
from pdi_pipeline.methods.spline import SplineInterpolator
from pdi_pipeline.methods.transforms import (
    DCTInpainting,
    TVInpainting,
    WaveletInpainting,
)

__all__ = [
    "BaseMethod",
    "BicubicInterpolator",
    "BilinearInterpolator",
    "DCTInpainting",
    "ExemplarBasedInterpolator",
    "IDWInterpolator",
    "KrigingInterpolator",
    "L1DCTInpainting",
    "L1WaveletInpainting",
    "LanczosInterpolator",
    "NearestInterpolator",
    "NonLocalMeansInterpolator",
    "RBFInterpolator",
    "SpaceTimeKriging",
    "SplineInterpolator",
    "TVInpainting",
    "TemporalFourierInterpolator",
    "TemporalSplineInterpolator",
    "WaveletInpainting",
]
