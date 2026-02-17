"""Bilinear (Delaunay linear) gap-filling."""

from __future__ import annotations

import logging

import numpy as np

from pdi_pipeline.methods._griddata import griddata_fill
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class BilinearInterpolator(BaseMethod):
    """C0 piecewise-linear fill on Delaunay triangulation (barycentric weights).

    Wraps ``scipy.interpolate.griddata(method='linear')``.
    See: Amidror (2002), J. Electronic Imaging 11(2).
    """

    name = "bilinear"

    def __init__(self) -> None:
        """Initialize the bilinear interpolator."""

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply bilinear (Delaunay linear) interpolation to fill gaps.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
                Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata (CRS, transform, band names, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        logger.debug("Running bilinear (Delaunay linear) gap-filling.")
        result = griddata_fill(
            degraded, mask_2d, "linear", self._apply_channelwise
        )
        if result is None:
            logger.debug(
                "griddata_fill returned None (no valid pixels); "
                "falling back to input copy."
            )
            return self._finalize(degraded.copy())
        return self._finalize(result)
