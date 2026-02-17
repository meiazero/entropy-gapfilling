"""Bicubic (Clough-Tocher C1) gap-filling."""

from __future__ import annotations

import logging

import numpy as np

from pdi_pipeline.methods._griddata import griddata_fill
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class BicubicInterpolator(BaseMethod):
    """C1 Clough-Tocher cubic fill on Delaunay triangulation.

    Wraps ``scipy.interpolate.griddata(method='cubic')``.
    See: Clough & Tocher (1965); Alfeld (1984).
    """

    name = "bicubic"

    def __init__(self) -> None:
        """Initialize the bicubic interpolator."""

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply bicubic (Clough--Tocher) interpolation to fill gaps.

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

        logger.debug("Running bicubic (Clough-Tocher C1) gap-filling.")
        result = griddata_fill(
            degraded, mask_2d, "cubic", self._apply_channelwise
        )
        if result is None:
            logger.debug(
                "griddata_fill returned None (no valid pixels); "
                "falling back to input copy."
            )
            return self._finalize(degraded.copy())
        return self._finalize(result)
