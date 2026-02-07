"""Multi-temporal interpolators: temporal spline, temporal Fourier, space-time kriging.

These methods exploit temporal correlation in time-series data to reconstruct
missing values across multiple acquisition dates.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy.interpolate import UnivariateSpline

from pdi_pipeline.methods.base import BaseMethod


class TemporalSplineInterpolator(BaseMethod):
    r"""Temporal Spline interpolation for time-series reconstruction.

    This method fits cubic splines along the temporal dimension for each spatial
    location, allowing smooth interpolation of missing values across time.

    Mathematical Formulation:
        For each pixel location $(x, y)$, the time series $f(t)$ is approximated by a
        piecewise cubic polynomial $S(t)$ with continuous first and second derivatives:

        $$S(t) = \begin{cases}
            S_0(t) & t \in [t_0, t_1] \\
            \vdots \\
            S_{n-1}(t) & t \in [t_{n-1}, t_n]
        \end{cases}$$

        where each piece $S_i(t)$ is a cubic polynomial:
        $$S_i(t) = a_i + b_i(t - t_i) + c_i(t - t_i)^2 + d_i(t - t_i)^3$$

        satisfying continuity conditions $S(t) \in C^2[t_0, t_n]$:
        - $S_i(t_{i+1}) = S_{i+1}(t_{i+1})$ (value continuity)
        - $S_i'(t_{i+1}) = S_{i+1}'(t_{i+1})$ (first derivative continuity)
        - $S_i''(t_{i+1}) = S_{i+1}''(t_{i+1})$ (second derivative continuity)

    Note:
        This interpolator requires temporal data with shape (T, H, W) or (T, H, W, C).

    Citation: Wikipedia contributors. "Spline interpolation." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Spline_interpolation
    """

    name = "temporal_spline"

    def __init__(self, smoothing: float = 0.0, degree: int = 3) -> None:
        """Initialize temporal spline interpolator.

        Args:
            smoothing: Smoothing factor for spline fitting (0 = exact interpolation)
            degree: Degree of spline (typically 3 for cubic splines)
        """
        self.smoothing = smoothing
        self.degree = degree

    def _fit_pixel_timeseries(
        self,
        time_series: np.ndarray,
        time_mask: np.ndarray,
        time_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Fit a spline to one pixel's time series and return interpolated missing values.

        Returns None if fitting fails or there are too few valid points.
        """
        valid_times = time_indices[~time_mask]
        valid_values = time_series[~time_mask]

        if len(valid_times) < 2:
            return None

        try:
            spline_degree = int(min(self.degree, len(valid_times) - 1, 5))
            spline_degree = max(1, spline_degree)
            k = cast(Literal[1, 2, 3, 4, 5], spline_degree)
            spline = UnivariateSpline(
                valid_times,
                valid_values,
                k=k,
                s=self.smoothing,
            )

            missing_times = time_indices[time_mask]
            if len(missing_times) > 0:
                return np.asarray(spline(missing_times), dtype=np.float32)
        except Exception:  # noqa: S110
            pass

        return None

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply temporal spline interpolation to recover missing pixels.

        Args:
            degraded: Input time series with shape (T, H, W) or (T, H, W, C)
            mask: Boolean mask where True indicates missing pixels, shape (T, H, W)
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed time series with gaps filled
        """
        if degraded.ndim < 3:
            raise ValueError(
                "Temporal interpolation requires at least 3D data (T, H, W)"
            )

        mask_arr = np.asarray(mask)
        if mask_arr.shape[:3] != degraded.shape[:3]:
            raise ValueError("Image and mask must match on (T,H,W)")
        if mask_arr.ndim == 4:
            mask_3d = np.any(mask_arr, axis=3)
        else:
            mask_3d = mask_arr.astype(bool)

        n_timesteps = degraded.shape[0]
        time_indices = np.arange(n_timesteps, dtype=np.float32)
        height, width = degraded.shape[1], degraded.shape[2]
        n_channels = degraded.shape[3] if degraded.ndim == 4 else 1

        reconstructed = degraded.copy().astype(np.float32)

        for h in range(height):
            for w in range(width):
                time_mask = mask_3d[:, h, w]
                for c in range(n_channels):
                    if degraded.ndim == 4:
                        ts = degraded[:, h, w, c].astype(np.float32)
                    else:
                        ts = degraded[:, h, w].astype(np.float32)

                    interpolated = self._fit_pixel_timeseries(
                        ts, time_mask, time_indices
                    )
                    if interpolated is not None:
                        missing_times = time_indices[time_mask]
                        if degraded.ndim == 4:
                            reconstructed[
                                missing_times.astype(int), h, w, c
                            ] = interpolated
                        else:
                            reconstructed[missing_times.astype(int), h, w] = (
                                interpolated
                            )

        return self._finalize(reconstructed)


class TemporalFourierInterpolator(BaseMethod):
    r"""Fourier Temporal interpolation using harmonic analysis.

    This method decomposes time series into harmonic components (Fourier series)
    and reconstructs missing values. It is particularly effective for periodic
    phenomena like seasonal vegetation cycles.

    Mathematical Formulation:
        The time series $f(t)$ is modeled as a truncated Fourier series:

        $$f(t) = a_0 + \sum_{k=1}^{M} \left[ a_k \cos\left(\frac{2\pi k t}{T}\right) + b_k \sin\left(\frac{2\pi k t}{T}\right) \right]$$

        where:
        - $a_0$ is the mean (DC component).
        - $a_k, b_k$ are Fourier coefficients determined by least-squares fitting to observed data.
        - $T$ is the period of the series.
        - $M$ is the number of harmonics (user-specified).

        The coefficients are found by solving the linear system:
        $$\mathbf{X}^T \mathbf{X} \mathbf{c} = \mathbf{X}^T \mathbf{y}$$
        where $\mathbf{X}$ is the design matrix containing $[1, \cos(\omega_k t), \sin(\omega_k t)]$
        for each harmonic $k$, and $\mathbf{y}$ are the observed values.

    Note:
        This interpolator requires temporal data with shape (T, H, W) or (T, H, W, C).

    Citation: Wikipedia contributors. "Fourier series." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Fourier_series
    """

    name = "temporal_fourier"

    def __init__(self, n_harmonics: int = 3, period: int | None = None) -> None:
        """Initialize Fourier temporal interpolator.

        Args:
            n_harmonics: Number of harmonic components to use
            period: Period length in time steps (if None, uses full time series length)
        """
        self.n_harmonics = n_harmonics
        self.period = period

    def _fit_pixel_fourier(
        self,
        time_series: np.ndarray,
        time_mask: np.ndarray,
        design_matrix_full: np.ndarray,
    ) -> np.ndarray | None:
        """Fit Fourier coefficients for one pixel's time series and return interpolated missing values.

        Returns None if fitting fails or there are too few valid points.
        """
        valid_mask = ~time_mask
        if valid_mask.sum() < (1 + 2 * self.n_harmonics):
            return None

        x_valid = design_matrix_full[valid_mask]
        y_valid = time_series[valid_mask]

        try:
            coeffs, *_ = np.linalg.lstsq(x_valid, y_valid, rcond=None)

            if time_mask.sum() > 0:
                x_missing = design_matrix_full[time_mask]
                return np.asarray(x_missing @ coeffs, dtype=np.float32)
        except Exception:  # noqa: S110
            pass

        return None

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Fourier harmonic analysis to recover missing pixels.

        Args:
            degraded: Input time series with shape (T, H, W) or (T, H, W, C)
            mask: Boolean mask where True indicates missing pixels, shape (T, H, W)
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed time series with gaps filled
        """
        if degraded.ndim not in (3, 4):
            raise ValueError(
                "Temporal Fourier requires input shape (T,H,W[,C])"
            )
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:3] != degraded.shape[:3]:
            raise ValueError("Image and mask must match on (T,H,W)")
        if mask_arr.ndim == 4:
            mask_3d = np.any(mask_arr, axis=3)
        else:
            mask_3d = mask_arr.astype(bool)

        n_timesteps = degraded.shape[0]
        time_indices = np.arange(n_timesteps, dtype=np.float32)
        height, width = degraded.shape[1], degraded.shape[2]
        n_channels = degraded.shape[3] if degraded.ndim == 4 else 1

        period = self.period if self.period is not None else n_timesteps

        reconstructed = degraded.copy().astype(np.float32)

        design_matrix_full = np.ones((n_timesteps, 1 + 2 * self.n_harmonics))
        for k in range(1, self.n_harmonics + 1):
            omega_k = 2 * np.pi * k / period
            design_matrix_full[:, 2 * k - 1] = np.cos(omega_k * time_indices)
            design_matrix_full[:, 2 * k] = np.sin(omega_k * time_indices)

        for h in range(height):
            for w in range(width):
                time_mask = mask_3d[:, h, w]
                for c in range(n_channels):
                    if degraded.ndim == 4:
                        ts = degraded[:, h, w, c].astype(np.float32)
                    else:
                        ts = degraded[:, h, w].astype(np.float32)

                    interpolated = self._fit_pixel_fourier(
                        ts, time_mask, design_matrix_full
                    )
                    if interpolated is not None:
                        if degraded.ndim == 4:
                            reconstructed[time_mask, h, w, c] = interpolated
                        else:
                            reconstructed[time_mask, h, w] = interpolated

        return self._finalize(reconstructed)


class SpaceTimeKriging(BaseMethod):
    r"""Space-Time Kriging for spatiotemporal interpolation.

    Extends ordinary kriging to the spatiotemporal domain by modeling correlation
    structures in both space and time through a space-time variogram.

    Mathematical Formulation:
        The estimator at a space-time point $(s_0, t_0)$ is:

        $$\hat{Z}(s_0, t_0) = \sum_{i=1}^N \lambda_i Z(s_i, t_i)$$

        where the weights $\lambda_i$ are determined by solving the space-time kriging system:

        $$\begin{bmatrix}
            C(h_{11}, \tau_{11}) & \cdots & C(h_{1N}, \tau_{1N}) & 1 \\
            \vdots & \ddots & \vdots & \vdots \\
            C(h_{N1}, \tau_{N1}) & \cdots & C(h_{NN}, \tau_{NN}) & 1 \\
            1 & \cdots & 1 & 0
        \end{bmatrix}
        \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_N \\ \mu \end{bmatrix} =
        \begin{bmatrix} C(h_{10}, \tau_{10}) \\ \vdots \\ C(h_{N0}, \tau_{N0}) \\ 1 \end{bmatrix}$$

        where:
        - $C(h, \tau)$ is the space-time covariance function depending on spatial lag $h = \|s_i - s_j\|$
          and temporal lag $\tau = |t_i - t_j|$.
        - A common separable model is: $C(h, \tau) = C_s(h) \cdot C_t(\tau)$
        - Example: $C(h, \tau) = \sigma^2 \exp(-h/r_s) \exp(-\tau/r_t)$ (exponential model).

    Note:
        This interpolator requires spatiotemporal data with shape (T, H, W) or (T, H, W, C).
        Space-time kriging exploits temporal correlation to improve predictions,
        ideal for filling cloud-cover gaps across multiple acquisition dates.

    Citation: Wikipedia contributors. "Kriging." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Kriging
    """

    name = "kriging_spacetime"

    def __init__(
        self,
        range_space: float = 5.0,
        range_time: float = 2.0,
        sill: float = 1.0,
        nugget: float = 1e-3,
        max_points: int = 256,
        random_seed: int = 13,
        kernel_size: int | None = None,
    ) -> None:
        """Initialize space-time kriging.

        Args:
            range_space: Spatial range parameter for exponential covariance.
            range_time: Temporal range parameter for exponential covariance.
            sill: Sill (overall variance scale) for the covariance.
            nugget: Nugget (diagonal regularization) for numerical stability.
            max_points: Maximum number of observations used per prediction.
            random_seed: Seed used for deterministic subsampling.
            kernel_size: Spatial search radius. If None, uses entire image.
        """
        self.range_space = float(range_space)
        self.range_time = float(range_time)
        self.sill = float(sill)
        self.nugget = float(nugget)
        self.max_points = int(max_points)
        self.random_seed = int(random_seed)
        self.kernel_size = kernel_size

    def _normalize_time_mask(self, mask: np.ndarray) -> np.ndarray:
        """Normalize mask to 3D (T, H, W) format."""
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 3:
            return mask_bool
        if mask_bool.ndim == 4:
            return np.any(mask_bool, axis=3).astype(bool, copy=False)
        raise ValueError(
            "For space-time kriging, mask must have shape (T, H, W) or (T, H, W, C)."
        )

    def _covariance(
        self,
        dt: np.ndarray,
        dy: np.ndarray,
        dx: np.ndarray,
    ) -> np.ndarray:
        """Compute space-time covariance using separable exponential model."""
        spatial_distance = np.sqrt(dy * dy + dx * dx)
        return np.asarray(
            self.sill
            * np.exp(-spatial_distance / max(self.range_space, 1e-6))
            * np.exp(-np.abs(dt) / max(self.range_time, 1e-6)),
            dtype=np.float32,
        )

    def _select_local_observations(
        self,
        obs_t: np.ndarray,
        obs_y: np.ndarray,
        obs_x: np.ndarray,
        target_t: int,
        target_y: int,
        target_x: int,
        spatial_radius: int,
        time_radius: int,
    ) -> np.ndarray:
        """Select observations within spatial and temporal radius."""
        dt = np.abs(obs_t - target_t)
        dy = obs_y - target_y
        dx = obs_x - target_x
        within_time = dt <= time_radius
        within_space = (dy * dy + dx * dx) <= spatial_radius * spatial_radius
        local = np.where(within_time & within_space)[0].astype(np.int64)

        if local.size == 0:
            return local

        if local.size <= self.max_points:
            return local

        dy_local = dy[local].astype(np.float32)
        dx_local = dx[local].astype(np.float32)
        dt_local = dt[local].astype(np.float32)
        scaled_distance = (
            (dy_local / max(self.range_space, 1e-6)) ** 2
            + (dx_local / max(self.range_space, 1e-6)) ** 2
            + (dt_local / max(self.range_time, 1e-6)) ** 2
        )

        order = np.argsort(scaled_distance)
        selected = local[order[: self.max_points]]
        return selected.astype(np.int64, copy=False)

    def _ordinary_kriging_predict(
        self,
        obs_t: np.ndarray,
        obs_y: np.ndarray,
        obs_x: np.ndarray,
        obs_values: np.ndarray,
        target_t: int,
        target_y: int,
        target_x: int,
    ) -> float:
        """Solve space-time kriging system and predict value."""
        n_points = obs_values.size
        if n_points == 0:
            return float("nan")
        if n_points == 1:
            return float(obs_values[0])

        obs_t_f = obs_t.astype(np.float32)
        obs_y_f = obs_y.astype(np.float32)
        obs_x_f = obs_x.astype(np.float32)

        dt_matrix = obs_t_f[:, None] - obs_t_f[None, :]
        dy_matrix = obs_y_f[:, None] - obs_y_f[None, :]
        dx_matrix = obs_x_f[:, None] - obs_x_f[None, :]

        covariance = self._covariance(dt_matrix, dy_matrix, dx_matrix)
        covariance = (
            covariance + np.eye(n_points, dtype=np.float32) * self.nugget
        )

        dt0 = obs_t_f - np.float32(target_t)
        dy0 = obs_y_f - np.float32(target_y)
        dx0 = obs_x_f - np.float32(target_x)
        covariance_0 = self._covariance(dt0, dy0, dx0)

        system = np.zeros((n_points + 1, n_points + 1), dtype=np.float32)
        system[:n_points, :n_points] = covariance
        system[:n_points, n_points] = 1.0
        system[n_points, :n_points] = 1.0

        rhs = np.zeros(n_points + 1, dtype=np.float32)
        rhs[:n_points] = covariance_0
        rhs[n_points] = 1.0

        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return float(np.mean(obs_values))

        weights = solution[:n_points]
        estimate = float(np.sum(weights * obs_values))
        return estimate

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply space-time kriging to recover missing pixels.

        Args:
            degraded: Input time series with shape (T, H, W) or (T, H, W, C)
            mask: Boolean mask where True indicates missing pixels
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed time series
        """
        image_float = np.asarray(degraded, dtype=np.float32)
        if image_float.ndim < 3:
            raise ValueError(
                "Space-time kriging requires a time series with shape (T, H, W) or (T, H, W, C)."
            )

        mask_3d = self._normalize_time_mask(mask)
        if image_float.shape[:3] != mask_3d.shape:
            raise ValueError(
                f"Image and mask must match on (T, H, W). Image: {image_float.shape[:3]}, Mask: {mask_3d.shape}."
            )

        _n_timesteps, height, width = image_float.shape[:3]
        spatial_radius = (
            self.kernel_size
            if self.kernel_size is not None
            else max(height, width)
        )
        spatial_radius = max(1, spatial_radius)
        time_radius = max(1, int(np.ceil(3.0 * self.range_time)))

        is_multichannel = image_float.ndim == 4
        reconstructed = image_float.copy()

        obs_t, obs_y, obs_x = np.where(~mask_3d)
        gap_t, gap_y, gap_x = np.where(mask_3d)

        if obs_t.size < 5 or gap_t.size == 0:
            return self._finalize(reconstructed)

        rng = np.random.default_rng(self.random_seed)

        def fill_channel(channel_values: np.ndarray) -> np.ndarray:
            """Fill missing values for a single channel."""
            filled_channel = channel_values.copy()
            for target_t, target_y, target_x in zip(gap_t, gap_y, gap_x):
                local_indices = self._select_local_observations(
                    obs_t.astype(np.int64),
                    obs_y.astype(np.int64),
                    obs_x.astype(np.int64),
                    int(target_t),
                    int(target_y),
                    int(target_x),
                    spatial_radius=spatial_radius,
                    time_radius=time_radius,
                )

                if local_indices.size < 5:
                    local_indices = np.arange(obs_t.size, dtype=np.int64)
                    if local_indices.size > self.max_points:
                        local_indices = rng.choice(
                            local_indices,
                            size=self.max_points,
                            replace=False,
                        ).astype(np.int64)

                local_t = obs_t[local_indices].astype(np.int64)
                local_y = obs_y[local_indices].astype(np.int64)
                local_x = obs_x[local_indices].astype(np.int64)
                local_values = channel_values[local_t, local_y, local_x]

                finite = np.isfinite(local_values)
                if int(np.sum(finite)) < 2:
                    continue

                estimate = self._ordinary_kriging_predict(
                    local_t[finite],
                    local_y[finite],
                    local_x[finite],
                    local_values[finite].astype(np.float32),
                    int(target_t),
                    int(target_y),
                    int(target_x),
                )
                if np.isfinite(estimate):
                    filled_channel[
                        int(target_t), int(target_y), int(target_x)
                    ] = estimate

            return filled_channel

        if is_multichannel:
            channels = image_float.shape[3]
            for channel_idx in range(channels):
                reconstructed[:, :, :, channel_idx] = fill_channel(
                    image_float[:, :, :, channel_idx]
                )
        else:
            reconstructed = fill_channel(image_float)

        return self._finalize(reconstructed)
