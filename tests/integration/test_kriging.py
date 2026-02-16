"""Integration tests for KrigingInterpolator using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.registry import get_interpolator
from tests.conftest import make_degraded, make_gradient, make_random_mask

# Use 16x16 images for speed -- kriging fits a variogram and solves
# a dense linear system per prediction point.
SIZE = 16
CHANNELS = 4


def _make_method(**kwargs):
    defaults = {"variogram_model": "spherical", "max_points": 200}
    defaults.update(kwargs)
    return get_interpolator("kriging", **defaults)


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
class TestKrigingInterpolator:
    """Standard integration tests for ordinary kriging interpolation."""

    def test_output_contract(self) -> None:
        clean = _small_clean_3d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        method = _make_method()
        result = method.apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
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

    @pytest.mark.parametrize(
        "variogram",
        ["spherical", "exponential", "gaussian", "linear"],
    )
    def test_variogram_models(self, variogram: str) -> None:
        clean = _small_clean_2d()
        mask = _small_mask()
        degraded = make_degraded(clean, mask)
        method = _make_method(variogram_model=variogram, max_points=100)
        result = method.apply(degraded, mask)
        assert np.all(np.isfinite(result))
