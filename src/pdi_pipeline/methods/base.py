"""Base interface for interpolation methods.

Every interpolation method in the pipeline inherits from
:class:`BaseMethod` and implements :meth:`apply`. The base class
provides shared validation, channel-wise dispatch, and output
finalization so that subclasses can focus on the mathematical kernel.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from pdi_pipeline.exceptions import DimensionError

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """Abstract base for all spatial interpolation methods.

    Implementations should be stateless or carry only lightweight
    configuration parameters set in ``__init__``.  The ``apply`` method
    must be purely functional with respect to the input data.

    Attributes:
        name: Human-readable method identifier used in logs and results.
    """

    name: str

    @abstractmethod
    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply interpolation to recover missing pixels.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to
                fill.  Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata dict (CRS, transform, band names,
                etc.).  Implementations that do not need metadata should
                ignore this parameter.

        Returns:
            Reconstructed ``float32`` array with same shape as
            *degraded*, values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """

    def fit(self, *args: Any, **kwargs: Any) -> BaseMethod:
        """Optional fitting step (no-op by default).

        Subclasses that need a training phase (e.g. kriging variogram
        estimation) should override this method.

        Returns:
            ``self`` for method chaining.
        """
        return self

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        degraded: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize inputs before interpolation.

        Args:
            degraded: Input image array.
            mask: Gap mask array.

        Returns:
            Tuple of ``(degraded_f32, mask_2d_bool)`` ready for use.

        Raises:
            DimensionError: If shapes are incompatible or unsupported.
            ValidationError: If dtypes cannot be safely converted.
        """
        degraded = np.asarray(degraded, dtype=np.float32)
        if degraded.ndim not in (2, 3):
            msg = (
                f"degraded must be 2D (H, W) or 3D (H, W, C), "
                f"got ndim={degraded.ndim}, shape={degraded.shape}"
            )
            raise DimensionError(msg)

        mask_2d = BaseMethod._normalize_mask(mask)

        if degraded.shape[:2] != mask_2d.shape:
            msg = (
                f"Spatial dimensions mismatch: degraded {degraded.shape[:2]} "
                f"vs mask {mask_2d.shape}"
            )
            raise DimensionError(msg)

        return degraded, mask_2d

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        """Convert any mask format to a 2D boolean array.

        Accepts float (threshold > 0.5), integer, or boolean masks.
        For 3D masks, collapses channels via logical OR.

        Args:
            mask: Mask array of any numeric dtype.

        Returns:
            2D boolean array where ``True`` marks gap pixels.

        Raises:
            DimensionError: If mask is not 2D or 3D.
        """
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 2:
            return mask_bool
        if mask_bool.ndim == 3:
            return np.any(mask_bool, axis=2)
        msg = (
            "Mask must be 2D or 3D, got "
            f"ndim={mask_bool.ndim}, shape={mask_bool.shape}"
        )
        raise DimensionError(msg)

    @staticmethod
    def _finalize(
        reconstructed: np.ndarray,
        *,
        clip_range: tuple[float, float] | None = (0.0, 1.0),
    ) -> np.ndarray:
        """Post-process the reconstructed array.

        Converts to ``float32``, clips to the valid range, and replaces
        any remaining ``NaN`` / ``Inf`` values.

        Args:
            reconstructed: Raw output from the interpolation kernel.
            clip_range: ``(min, max)`` for clipping, or ``None`` to skip.

        Returns:
            Clean ``float32`` array with no ``NaN`` or ``Inf``.
        """
        out = np.asarray(reconstructed, dtype=np.float32)
        if clip_range is not None:
            out = np.clip(out, clip_range[0], clip_range[1])
        out = np.nan_to_num(
            out,
            nan=0.0,
            posinf=clip_range[1] if clip_range else 0.0,
            neginf=clip_range[0] if clip_range else 0.0,
        )
        return out

    @staticmethod
    def _apply_channelwise(
        degraded: np.ndarray,
        mask_2d: np.ndarray,
        channel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Apply a per-channel function to a possibly multichannel image.

        Args:
            degraded: Input array ``(H, W)`` or ``(H, W, C)``.
            mask_2d: 2D boolean gap mask.
            channel_fn: ``fn(channel_2d, mask_2d) -> result_2d``.

        Returns:
            Reconstructed array with same shape as *degraded*.
        """
        if degraded.ndim == 2:
            return channel_fn(degraded, mask_2d)

        result = np.zeros_like(degraded)
        for c in range(degraded.shape[2]):
            result[:, :, c] = channel_fn(degraded[:, :, c], mask_2d)
        return result

    @staticmethod
    def _early_exit_if_no_gaps(
        degraded: np.ndarray,
        mask_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Return a copy of *degraded* if the mask contains no gaps.

        Args:
            degraded: Input image.
            mask_2d: 2D boolean gap mask.

        Returns:
            Finalized copy of *degraded* if no gaps, else ``None``.
        """
        if not np.any(mask_2d):
            logger.debug("No gap pixels found; returning input unchanged.")
            return BaseMethod._finalize(degraded.copy())
        return None
