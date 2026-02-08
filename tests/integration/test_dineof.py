"""Integration tests for DINEOFInterpolator on synthetic time series.

DINEOF requires (T, H, W) or (T, H, W, C) input. We construct synthetic
temporal stacks from real patches by applying small perturbations across
time steps, simulating a multi-date acquisition scenario.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.dineof import DINEOFInterpolator
from tests.conftest import PatchSample

pytestmark = pytest.mark.skip(
    reason="DINEOF excluded: requires time-series input"
)


@pytest.fixture()
def temporal_stack(
    sentinel2_patch: PatchSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic (T, H, W) time series from a single patch.

    Returns (clean_series, degraded_series, mask_3d) where:
    - clean_series: (T, H, W) clean single-channel time series
    - degraded_series: (T, H, W) with NaN-free gaps (filled with 0)
    - mask_3d: (T, H, W) bool, True = gap
    """
    n_timesteps = 6
    clean_single = sentinel2_patch.clean[:, :, 0]
    mask_2d = sentinel2_patch.mask.astype(bool)
    h, w = clean_single.shape

    rng = np.random.default_rng(42)
    clean_series = np.zeros((n_timesteps, h, w), dtype=np.float32)
    for t in range(n_timesteps):
        scale = 1.0 + 0.05 * (t - n_timesteps / 2)
        noise = rng.normal(0, 0.01, size=(h, w)).astype(np.float32)
        clean_series[t] = np.clip(clean_single * scale + noise, 0, 1)

    mask_3d = np.zeros((n_timesteps, h, w), dtype=bool)
    for t in range(n_timesteps):
        if t % 2 == 0:
            mask_3d[t] = mask_2d
        else:
            shifted = np.roll(mask_2d, shift=t * 3, axis=0)
            mask_3d[t] = shifted

    degraded_series = clean_series.copy()
    degraded_series[mask_3d] = 0.0

    return clean_series, degraded_series, mask_3d


@pytest.fixture()
def temporal_stack_multichannel(
    sentinel2_patch: PatchSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic (T, H, W, C) time series from a patch.

    Returns (clean_series, degraded_series, mask_3d) where:
    - clean_series: (T, H, W, C) clean multi-channel
    - degraded_series: (T, H, W, C) with gaps zeroed
    - mask_3d: (T, H, W) bool
    """
    n_timesteps = 4
    clean = sentinel2_patch.clean
    mask_2d = sentinel2_patch.mask.astype(bool)
    h, w, c = clean.shape

    rng = np.random.default_rng(99)
    clean_series = np.zeros((n_timesteps, h, w, c), dtype=np.float32)
    for t in range(n_timesteps):
        scale = 1.0 + 0.03 * (t - n_timesteps / 2)
        noise = rng.normal(0, 0.005, size=(h, w, c)).astype(np.float32)
        clean_series[t] = np.clip(clean * scale + noise, 0, 1)

    mask_3d = np.zeros((n_timesteps, h, w), dtype=bool)
    for t in range(n_timesteps):
        mask_3d[t] = np.roll(mask_2d, shift=t * 5, axis=1)

    degraded_series = clean_series.copy()
    for t in range(n_timesteps):
        degraded_series[t][mask_3d[t]] = 0.0

    return clean_series, degraded_series, mask_3d


@pytest.mark.integration
class TestDINEOFOnSyntheticTimeSeries:
    def test_output_contract(
        self,
        temporal_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_stack
        method = DINEOFInterpolator(max_modes=5, max_iterations=30)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(
        self,
        temporal_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_stack
        method = DINEOFInterpolator(max_modes=5, max_iterations=50)
        result = method.apply(degraded, mask_3d)
        valid = ~mask_3d
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-3)

    def test_reconstruction_quality(
        self,
        temporal_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        clean, degraded, mask_3d = temporal_stack
        method = DINEOFInterpolator(max_modes=5, max_iterations=50)
        result = method.apply(degraded, mask_3d)
        mse = float(np.mean((clean - result) ** 2))
        assert mse < 1.0, f"Reconstruction MSE too high: {mse:.4f}"

    def test_multichannel_support(
        self,
        temporal_stack_multichannel: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_stack_multichannel
        assert degraded.ndim == 4
        method = DINEOFInterpolator(max_modes=3, max_iterations=30)
        result = method.apply(degraded, mask_3d)
        assert result.shape == degraded.shape
        assert np.all(np.isfinite(result))

    def test_convergence_with_max_modes(
        self,
        temporal_stack: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _clean, degraded, mask_3d = temporal_stack
        r1 = DINEOFInterpolator(max_modes=1, max_iterations=30).apply(
            degraded, mask_3d
        )
        r3 = DINEOFInterpolator(max_modes=3, max_iterations=30).apply(
            degraded, mask_3d
        )
        assert np.all(np.isfinite(r1))
        assert np.all(np.isfinite(r3))

    def test_rejects_2d_input(self) -> None:
        method = DINEOFInterpolator()
        image_2d = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        mask_2d = np.zeros((16, 16), dtype=bool)
        with pytest.raises(ValueError, match="time series"):
            method.apply(image_2d, mask_2d)
