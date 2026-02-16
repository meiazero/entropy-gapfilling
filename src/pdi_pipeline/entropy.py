"""Local Shannon entropy computation for satellite image patches.

Entropy is computed on a uint8-quantized grayscale version of the input
(mean across all bands for multichannel images). The sliding-window
approach uses skimage.filters.rank.entropy with a disk structuring
element inscribed in a square of the requested window size.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters.rank import entropy as _rank_entropy
from skimage.morphology import footprint_rectangle

from pdi_pipeline.exceptions import DimensionError, ValidationError

logger = logging.getLogger(__name__)


def shannon_entropy(
    image: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute per-pixel local Shannon entropy.

    For multichannel images (H, W, C), entropy is computed on the mean
    of all bands - not the per-band average, which would inflate entropy
    for correlated bands.

    Args:
        image: Input image, (H, W) or (H, W, C), any dtype.
            Values are internally rescaled to uint8 [0, 255].
        window_size: Side length of the square sliding window.
            Must be a positive odd integer (e.g. 7, 15, 31).

    Returns:
        Float32 array of shape (H, W) with local entropy values.

    Raises:
        ValueError: If window_size is not a positive odd integer, or if
            image dimensions are invalid.
    """
    if window_size < 1 or window_size % 2 == 0:
        msg = f"window_size must be a positive odd integer, got {window_size}"
        raise ValidationError(msg)

    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    elif image.ndim == 2:
        gray = image
    else:
        msg = f"Image must be 2D or 3D, got ndim={image.ndim}"
        raise DimensionError(msg)

    vmin, vmax = float(gray.min()), float(gray.max())
    span = vmax - vmin
    if span < 1e-10:
        return np.zeros(gray.shape, dtype=np.float32)

    gray_u8 = ((gray - vmin) / span * 255.0).astype(np.uint8)

    selem = footprint_rectangle((window_size, window_size))
    ent = _rank_entropy(gray_u8, footprint=selem)

    return ent.astype(np.float32)
