"""Integration tests for SplineInterpolator using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.registry import get_interpolator
from tests.conftest import make_degraded, make_gradient, make_random_mask

# Use 16x16 images for speed -- spline fits a dense system.
SIZE = 16
CHANNELS = 4


def _make_method(**kwargs):
    defaults = {"max_training_points": 500}
    defaults.update(kwargs)
    return get_interpolator("spline", **defaults)


def _small_clean_3d() -> np.ndarray:
    return make_gradient(SIZE, SIZE, CHANNELS)


def _small_clean_2d() -> np.ndarray:
    return make_gradient(SIZE, SIZE, 0)


def _small_mask() -> np.ndarray:
    return make_random_mask(SIZE, SIZE, gap_fraction=0.3, seed=42)


def _small_mask_no_gap() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=bool)


def _small_mask_single_pixel() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[SIZE // 2, SIZE // 2] = True
    return mask


@pytest.mark.slow
@pytest.mark.integration
class TestSplineInterpolator:
    """Standard integration tests for thin plate spline interpolation."""

    def test_output_contract(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        method = _make_method()
        result = method.apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        method = _make_method()
        result = method.apply(degraded, mask)
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-4)

    def test_no_gap_passthrough(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask_no_gap()
        method = _make_method()
        result = method.apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask_single_pixel()
        degraded = make_degraded(clean, mask)
        method = _make_method()
        result = method.apply(degraded, mask)

        assert np.all(np.isfinite(result))
        cx = SIZE // 2
        filled = result[cx, cx]
        assert np.all(np.isfinite(filled))

    def test_multichannel_support(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        method = _make_method()
        result = method.apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _small_clean_2d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        method = _make_method()
        result = method.apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2

    def test_smoothing_parameter(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        exact = _make_method(smoothing=0.0)
        smooth = _make_method(smoothing=0.5)
        r_exact = exact.apply(degraded, mask)
        r_smooth = smooth.apply(degraded, mask)
        assert np.all(np.isfinite(r_exact))
        assert np.all(np.isfinite(r_smooth))
