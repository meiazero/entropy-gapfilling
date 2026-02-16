"""Integration tests for DCT, Wavelet, and TV inpainting using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.registry import get_interpolator
from tests.conftest import make_degraded, make_gradient, make_random_mask

# DCT and Wavelet are iterative but fast on small images.
# TV is iterative too. Use 32x32 for DCT, 16x16 for Wavelet/TV.
SIZE_FAST = 32
SIZE_SLOW = 16
CHANNELS = 4


# -- helpers ----------------------------------------------------------------


def _clean_3d(size: int) -> np.ndarray:
    return make_gradient(size, size, CHANNELS)


def _clean_2d(size: int) -> np.ndarray:
    return make_gradient(size, size, 0)


def _mask(size: int) -> np.ndarray:
    return make_random_mask(size, size, gap_fraction=0.3, seed=42)


def _mask_no_gap(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=bool)


def _mask_single_pixel(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 2, size // 2] = True
    return mask


# ---- DCT -----------------------------------------------------------------


@pytest.mark.integration
class TestDCTInpainting:
    """Standard integration tests for DCT inpainting."""

    SIZE = SIZE_FAST

    def _method(self, **kwargs):
        defaults = {"max_iterations": 30}
        defaults.update(kwargs)
        return get_interpolator("dct", **defaults)

    def test_output_contract(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-4)

    def test_no_gap_passthrough(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_no_gap(self.SIZE)
        result = self._method().apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_single_pixel(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        assert np.all(np.isfinite(result))

    def test_multichannel_support(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _clean_2d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2


# ---- Wavelet --------------------------------------------------------------


@pytest.mark.integration
class TestWaveletInpainting:
    """Standard integration tests for wavelet inpainting."""

    SIZE = SIZE_SLOW

    def _method(self, **kwargs):
        defaults = {"wavelet": "db4", "level": 2, "max_iterations": 20}
        defaults.update(kwargs)
        return get_interpolator("wavelet", **defaults)

    def test_output_contract(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-4)

    def test_no_gap_passthrough(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_no_gap(self.SIZE)
        result = self._method().apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_single_pixel(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        assert np.all(np.isfinite(result))

    def test_multichannel_support(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _clean_2d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2


# ---- TV -------------------------------------------------------------------


@pytest.mark.integration
class TestTVInpainting:
    """Standard integration tests for total variation inpainting."""

    SIZE = SIZE_SLOW

    def _method(self, **kwargs):
        defaults = {"lambda_param": 0.1, "max_iterations": 30}
        defaults.update(kwargs)
        return get_interpolator("tv", **defaults)

    def test_output_contract(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        # TV with high lambda enforces stronger data fidelity.
        result = self._method(lambda_param=10.0, max_iterations=100).apply(
            degraded, mask
        )
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=0.1)

    def test_no_gap_passthrough(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_no_gap(self.SIZE)
        result = self._method().apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask_single_pixel(self.SIZE)
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        assert np.all(np.isfinite(result))

    def test_multichannel_support(self) -> None:
        clean = _clean_3d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _clean_2d(self.SIZE)
        mask = _mask(self.SIZE)
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2
