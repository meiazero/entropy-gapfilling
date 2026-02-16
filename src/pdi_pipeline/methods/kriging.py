"""Ordinary Kriging interpolation with variogram modeling.

Kriging is a geostatistical method that provides Best Linear Unbiased Prediction (BLUP)
by modeling spatial correlation through a variogram. It minimizes prediction variance
while ensuring unbiasedness through the constraint that weights sum to unity.
"""

from __future__ import annotations

import logging

import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata

from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_MIN_TRAINING_POINTS = 10
"""Minimum number of unique training points required for variogram fitting."""

_DEFAULT_RANDOM_SEED = 13
"""Default random seed used when ``random_seed`` is ``None``."""

_MIN_KERNEL_RADIUS = 1
"""Minimum kernel radius in pixels."""


class KrigingInterpolator(BaseMethod):
    r"""Ordinary Kriging interpolation with variogram modeling.

    Kriging is a geostatistical method that provides Best Linear Unbiased Prediction (BLUP)
    by modeling spatial correlation through a variogram. It minimizes prediction variance
    while ensuring unbiasedness through the constraint that weights sum to unity.

    Mathematical Formulation:
        The kriging estimator is:

        $$\hat{Z}(x_0) = \sum_{i=1}^N w_i(x_0) Z(x_i)$$

        where the weights $w_i$ are found by solving the kriging system:

        $$\begin{bmatrix}
            \gamma(x_1, x_1) & \cdots & \gamma(x_1, x_n) & 1 \\
            \vdots & \ddots & \vdots & \vdots \\
            \gamma(x_n, x_1) & \cdots & \gamma(x_n, x_n) & 1 \\
            1 & \cdots & 1 & 0
        \end{bmatrix}
        \begin{bmatrix} w_1 \\ \vdots \\ w_n \\ \mu \end{bmatrix} =
        \begin{bmatrix} \gamma(x_1, x_0) \\ \vdots \\ \gamma(x_n, x_0) \\ 1 \end{bmatrix}$$

        Here $\gamma$ is the semivariogram, and $\mu$ is a Lagrange multiplier ensuring
        unbiasedness (weights sum to 1).

    Citation: Wikipedia contributors. "Kriging." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Kriging
    """

    name = "kriging"

    def __init__(
        self,
        variogram_model: str = "spherical",
        nlags: int = 6,
        max_points: int = 2500,
        random_seed: int | None = 13,
        kernel_size: int | None = None,
    ) -> None:
        """Initialize Kriging interpolator.

        Args:
            variogram_model: Type of variogram model. Options: 'spherical',
                'exponential', 'gaussian', 'linear', 'power'.
            nlags: Number of averaging bins for the semivariogram.
            max_points: Maximum number of training points to use.
            random_seed: Random seed for reproducible downsampling.
            kernel_size: Search window size. If None, uses entire image.
        """
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.max_points = max_points
        self.random_seed = random_seed
        self.kernel_size = kernel_size

    def _kriging_single_channel(
        self,
        channel_data: np.ndarray,
        train_y: np.ndarray,
        train_x: np.ndarray,
        gap_y: np.ndarray,
        gap_x: np.ndarray,
    ) -> np.ndarray | None:
        """Interpolate one 2D channel via ordinary kriging.

        Args:
            channel_data: 2D array of pixel values for one channel.
            train_y: Row indices of training pixels.
            train_x: Column indices of training pixels.
            gap_y: Row indices of gap pixels.
            gap_x: Column indices of gap pixels.

        Returns:
            Filled channel array, or ``None`` if insufficient data.
        """
        train_values = channel_data[train_y, train_x]
        finite_mask = np.isfinite(train_values)
        if int(np.sum(finite_mask)) < _MIN_TRAINING_POINTS:
            logger.warning(
                "Insufficient finite training values (%d < %d); skipping channel",
                int(np.sum(finite_mask)),
                _MIN_TRAINING_POINTS,
            )
            return None

        x_points = train_x[finite_mask].astype(np.float64)
        y_points = train_y[finite_mask].astype(np.float64)
        z_points = train_values[finite_mask].astype(np.float64)

        unique_xy, unique_index = np.unique(
            np.column_stack([x_points, y_points]), axis=0, return_index=True
        )
        x_points = unique_xy[:, 0]
        y_points = unique_xy[:, 1]
        z_points = z_points[unique_index]

        if x_points.size < _MIN_TRAINING_POINTS:
            logger.warning(
                "Insufficient unique training points (%d < %d); skipping channel",
                x_points.size,
                _MIN_TRAINING_POINTS,
            )
            return None

        result = channel_data.copy()
        try:
            kriging_model = OrdinaryKriging(
                x_points,
                y_points,
                z_points,
                variogram_model=self.variogram_model,
                nlags=self.nlags,
                enable_plotting=False,
                verbose=False,
            )
            interpolated_values, _ = kriging_model.execute(
                "points",
                gap_x.astype(np.float64),
                gap_y.astype(np.float64),
            )
            result[gap_y, gap_x] = interpolated_values
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            logger.warning(
                "Kriging failed (variogram=%s): %s; "
                "falling back to nearest-neighbor griddata",
                self.variogram_model,
                exc,
            )
            coords_valid = np.column_stack([x_points, y_points])
            coords_gap = np.column_stack([
                gap_x.astype(np.float64),
                gap_y.astype(np.float64),
            ])
            filled = griddata(
                coords_valid, z_points, coords_gap, method="nearest"
            )
            result[gap_y, gap_x] = filled.astype(result.dtype, copy=False)

        return result

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Ordinary Kriging to recover missing pixels.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        h, w = mask_2d.shape

        valid_y, valid_x = np.where(~mask_2d)
        gap_y, gap_x = np.where(mask_2d)

        if len(valid_y) < _MIN_TRAINING_POINTS:
            logger.warning(
                "Insufficient valid pixels (%d < %d); returning input unchanged",
                len(valid_y),
                _MIN_TRAINING_POINTS,
            )
            return self._finalize(degraded.copy())

        logger.debug(
            "Kriging interpolation: %d gap pixels, %d valid pixels, variogram=%s",
            len(gap_y),
            len(valid_y),
            self.variogram_model,
        )

        selected_indices = self._select_training_points(
            valid_y, valid_x, gap_y, gap_x, (h, w)
        )
        if selected_indices.size < _MIN_TRAINING_POINTS:
            logger.warning(
                "Insufficient selected training points (%d < %d); "
                "returning input unchanged",
                selected_indices.size,
                _MIN_TRAINING_POINTS,
            )
            return self._finalize(degraded.copy())

        train_y = valid_y[selected_indices]
        train_x = valid_x[selected_indices]

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            filled = self._kriging_single_channel(
                ch, train_y, train_x, gap_y, gap_x
            )
            return filled if filled is not None else ch

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)

    def _select_training_points(
        self,
        valid_y: np.ndarray,
        valid_x: np.ndarray,
        gap_y: np.ndarray,
        gap_x: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Select a spatially limited subset of valid points.

        The goal is to reduce numerical issues and cubic cost in kriging
        by restricting the number of training points.

        Args:
            valid_y: Y coordinates of valid pixels.
            valid_x: X coordinates of valid pixels.
            gap_y: Y coordinates of gap pixels.
            gap_x: X coordinates of gap pixels.
            image_shape: ``(height, width)`` of the image.

        Returns:
            Indices of selected training points.
        """
        if valid_y.size == 0:
            return np.array([], dtype=np.int64)

        h, w = image_shape
        radius = self.kernel_size if self.kernel_size is not None else max(h, w)
        radius = max(_MIN_KERNEL_RADIUS, radius)

        if gap_y.size == 0 or radius <= 0:
            indices = np.arange(valid_y.size, dtype=np.int64)
        else:
            y_min = int(np.clip(gap_y.min() - radius, 0, h - 1))
            y_max = int(np.clip(gap_y.max() + radius, 0, h - 1))
            x_min = int(np.clip(gap_x.min() - radius, 0, w - 1))
            x_max = int(np.clip(gap_x.max() + radius, 0, w - 1))

            in_box = (
                (valid_y >= y_min)
                & (valid_y <= y_max)
                & (valid_x >= x_min)
                & (valid_x <= x_max)
            )
            indices = np.where(in_box)[0].astype(np.int64)
            if indices.size == 0:
                indices = np.arange(valid_y.size, dtype=np.int64)

        if indices.size > self.max_points:
            rng_seed = (
                self.random_seed
                if self.random_seed is not None
                else _DEFAULT_RANDOM_SEED
            )
            rng = np.random.default_rng(rng_seed)
            indices = rng.choice(
                indices, size=self.max_points, replace=False
            ).astype(np.int64)

        return indices
