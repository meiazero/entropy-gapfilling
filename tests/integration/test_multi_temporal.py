"""Integration tests for temporal interpolators on synthetic time series.

Tests cover TemporalSplineInterpolator, TemporalFourierInterpolator, and
SpaceTimeKriging.  All require (T, H, W) or (T, H, W, C) data so we build
synthetic stacks from real satellite patches.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.multi_temporal import (
    SpaceTimeKriging,
    TemporalFourierInterpolator,
    TemporalSplineInterpolator,
)
from tests.conftest import PatchSample

# ---------------------------------------------------------------------------
# Shared fixtures for temporal data
# ---------------------------------------------------------------------------


@pytest.fixture()
def temporal_3d(
    sentinel2_patch: PatchSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic (T, H, W) time series from a single-channel patch.

    Returns (clean_series, degraded_series, mask_3d).
    """
    n_timesteps = 8
    clean_ch = sentinel2_patch.clean[:, :, 0]
    mask_2d = sentinel2_patch.mask.astype(bool)
    h, w = clean_ch.shape

    rng = np.random.default_rng(77)
    clean_series = np.zeros((n_timesteps, h, w), dtype=np.float32)
    for t in range(n_timesteps):
        phase = 2.0 * np.pi * t / n_timesteps
        scale = 1.0 + 0.1 * np.sin(phase)
        noise = rng.normal(0, 0.01, size=(h, w)).astype(np.float32)
        clean_series[t] = np.clip(clean_ch * scale + noise, 0, 1)

    mask_3d = np.zeros((n_timesteps, h, w), dtype=bool)
    for t in range(n_timesteps):
        mask_3d[t] = np.roll(mask_2d, shift=t * 4, axis=0)

    degraded = clean_series.copy()
    degraded[mask_3d] = 0.0

    return clean_series, degraded, mask_3d


@pytest.fixture()
def temporal_4d(
    sentinel2_patch: PatchSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic (T, H, W, C) time series from a multi-channel patch.

    Returns (clean_series, degraded_series, mask_3d).
    """
    n_timesteps = 6
    clean = sentinel2_patch.clean
    mask_2d = sentinel2_patch.mask.astype(bool)
    h, w, c = clean.shape

    rng = np.random.default_rng(55)
    clean_series = np.zeros((n_timesteps, h, w, c), dtype=np.float32)
    for t in range(n_timesteps):
        phase = 2.0 * np.pi * t / n_timesteps
        scale = 1.0 + 0.08 * np.sin(phase)
        noise = rng.normal(0, 0.005, size=(h, w, c)).astype(np.float32)
        clean_series[t] = np.clip(clean * scale + noise, 0, 1)

    mask_3d = np.zeros((n_timesteps, h, w), dtype=bool)
    for t in range(n_timesteps):
        mask_3d[t] = np.roll(mask_2d, shift=t * 3, axis=1)

    degraded = clean_series.copy()
    for t in range(n_timesteps):
        degraded[t][mask_3d[t]] = 0.0

    return clean_series, degraded, mask_3d


@pytest.fixture()
def small_temporal_3d(
    sentinel2_patch: PatchSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small (T, h, w) stack for SpaceTimeKriging (which is O(n^3)).

    Takes a 16x16 crop to keep runtime manageable.
    """
    n_timesteps = 5
    clean_ch = sentinel2_patch.clean[:16, :16, 0]
    mask_2d = sentinel2_patch.mask[:16, :16].astype(bool)
    h, w = clean_ch.shape

    rng = np.random.default_rng(33)
    clean_series = np.zeros((n_timesteps, h, w), dtype=np.float32)
    for t in range(n_timesteps):
        scale = 1.0 + 0.05 * (t - n_timesteps / 2)
        noise = rng.normal(0, 0.01, size=(h, w)).astype(np.float32)
        clean_series[t] = np.clip(clean_ch * scale + noise, 0, 1)

    mask_3d = np.zeros((n_timesteps, h, w), dtype=bool)
    for t in range(n_timesteps):
        mask_3d[t] = np.roll(mask_2d, shift=t * 2, axis=0)

    degraded = clean_series.copy()
    degraded[mask_3d] = 0.0

    return clean_series, degraded, mask_3d


# ---------------------------------------------------------------------------
# TemporalSplineInterpolator
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTemporalSplineOnSyntheticData:
    def test_output_contract(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalSplineInterpolator(smoothing=0.0, degree=3)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalSplineInterpolator(smoothing=0.0, degree=3)
        result = method.apply(degraded, mask_3d)
        valid = ~mask_3d
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-3)

    def test_reconstruction_bounded(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        clean, degraded, mask_3d = temporal_3d
        method = TemporalSplineInterpolator(smoothing=0.0)
        result = method.apply(degraded, mask_3d)
        mse = float(np.mean((clean - result) ** 2))
        assert mse < 1.0

    def test_multichannel_support(
        self,
        temporal_4d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_4d
        assert degraded.ndim == 4
        method = TemporalSplineInterpolator(smoothing=0.0)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_spline_degrees(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalSplineInterpolator(smoothing=0.0, degree=degree)
        result = method.apply(degraded, mask_3d)
        assert np.all(np.isfinite(result))

    def test_rejects_2d_input(self) -> None:
        method = TemporalSplineInterpolator()
        arr = np.zeros((16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        with pytest.raises(ValueError, match="3D"):
            method.apply(arr, mask)


# ---------------------------------------------------------------------------
# TemporalFourierInterpolator
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTemporalFourierOnSyntheticData:
    def test_output_contract(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalFourierInterpolator(n_harmonics=2)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalFourierInterpolator(n_harmonics=2)
        result = method.apply(degraded, mask_3d)
        valid = ~mask_3d
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-3)

    def test_reconstruction_bounded(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        clean, degraded, mask_3d = temporal_3d
        method = TemporalFourierInterpolator(n_harmonics=3)
        result = method.apply(degraded, mask_3d)
        mse = float(np.mean((clean - result) ** 2))
        assert mse < 1.0

    def test_multichannel_support(
        self,
        temporal_4d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_4d
        assert degraded.ndim == 4
        method = TemporalFourierInterpolator(n_harmonics=2)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("n_harmonics", [1, 2, 3])
    def test_harmonic_counts(
        self,
        temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_harmonics: int,
    ) -> None:
        _clean, degraded, mask_3d = temporal_3d
        method = TemporalFourierInterpolator(n_harmonics=n_harmonics)
        result = method.apply(degraded, mask_3d)
        assert np.all(np.isfinite(result))

    def test_rejects_2d_input(self) -> None:
        method = TemporalFourierInterpolator()
        arr = np.zeros((16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        with pytest.raises(ValueError, match="Temporal Fourier"):
            method.apply(arr, mask)


# ---------------------------------------------------------------------------
# SpaceTimeKriging
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSpaceTimeKrigingOnSyntheticData:
    def test_output_contract(
        self,
        small_temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = small_temporal_3d
        method = SpaceTimeKriging(
            range_space=5.0,
            range_time=2.0,
            max_points=64,
            kernel_size=8,
        )
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(
        self,
        small_temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = small_temporal_3d
        method = SpaceTimeKriging(max_points=64, kernel_size=8)
        result = method.apply(degraded, mask_3d)
        valid = ~mask_3d
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-3)

    def test_reconstruction_finite(
        self,
        small_temporal_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        clean, degraded, mask_3d = small_temporal_3d
        method = SpaceTimeKriging(max_points=64, kernel_size=8)
        result = method.apply(degraded, mask_3d)
        mse = float(np.mean((clean - result) ** 2))
        assert mse < 1.0

    def test_rejects_2d_input(self) -> None:
        method = SpaceTimeKriging()
        arr = np.zeros((16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        with pytest.raises(ValueError, match="time series"):
            method.apply(arr, mask)
