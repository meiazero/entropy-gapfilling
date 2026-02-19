"""Local Shannon entropy via sliding-window rank filter.

Grayscale (uint8) from band-mean; uses a rectangular structuring element.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters.rank import entropy as _rank_entropy
from skimage.morphology import footprint_rectangle

from pdi_pipeline.exceptions import DimensionError, ValidationError

_MIN_SPAN = 1e-10

logger = logging.getLogger(__name__)

# Module-level cache: reuse the same footprint array for repeated window sizes.
# footprint_rectangle creates an uint8 array; caching avoids repeated
# allocations # when shannon_entropy is called thousands of times
# (e.g. 77k patches x 3 windows).
_FOOTPRINT_CACHE: dict[int, np.ndarray] = {}


def _get_footprint(window_size: int) -> np.ndarray:
    if window_size not in _FOOTPRINT_CACHE:
        _FOOTPRINT_CACHE[window_size] = footprint_rectangle((
            window_size,
            window_size,
        ))
    return _FOOTPRINT_CACHE[window_size]


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

    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 3:
        gray = np.mean(image, axis=2, dtype=np.float32)
    elif image.ndim == 2:
        gray = image
    else:
        msg = f"Image must be 2D or 3D, got ndim={image.ndim}"
        raise DimensionError(msg)

    vmin, vmax = float(gray.min()), float(gray.max())
    span = vmax - vmin
    if span < _MIN_SPAN:
        return np.zeros(gray.shape, dtype=np.float32)

    # Scale to [0, 255] using float32 (avoids float64 intermediate)
    scaled = (gray - vmin) * np.float32(255.0 / span)
    gray_u8 = scaled.astype(np.uint8)

    ent = _rank_entropy(gray_u8, footprint=_get_footprint(window_size))

    return ent.astype(np.float32)
