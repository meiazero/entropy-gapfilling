"""Integration tests for NonLocalMeans and ExemplarBased.

Uses synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.registry import get_interpolator
from tests.conftest import make_degraded, make_gradient, make_random_mask

# Use 16x16 images for speed -- patch-based methods scan neighborhoods.
SIZE = 16
CHANNELS = 4


def _clean_3d() -> np.ndarray:
    return make_gradient(SIZE, SIZE, CHANNELS)


def _clean_2d() -> np.ndarray:
    return make_gradient(SIZE, SIZE, 0)


def _mask() -> np.ndarray:
    return make_random_mask(SIZE, SIZE, gap_fraction=0.3, seed=42)


def _mask_no_gap() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=bool)


def _mask_single_pixel() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[SIZE // 2, SIZE // 2] = True
    return mask


# ---- NonLocalMeans --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestNonLocalMeansInterpolator:
    """Standard integration tests for non-local means interpolation."""

    def _method(self, **kwargs):
        defaults = {"patch_size": 3, "patch_distance": 4, "h_rel": 0.8}
        defaults.update(kwargs)
        return get_interpolator("non_local", **defaults)

    def test_output_contract(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-4)

    def test_no_gap_passthrough(self) -> None:
        clean = _clean_3d()
        mask = _mask_no_gap()
        result = self._method().apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _clean_3d()
        mask = _mask_single_pixel()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        assert np.all(np.isfinite(result))

    def test_multichannel_support(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _clean_2d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2


# ---- ExemplarBased --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestExemplarBasedInterpolator:
    """Standard integration tests for exemplar-based (biharmonic) inpainting."""

    def _method(self, **kwargs):
        return get_interpolator("exemplar", **kwargs)

    def test_output_contract(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)

        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        valid = ~mask
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-4)

    def test_no_gap_passthrough(self) -> None:
        clean = _clean_3d()
        mask = _mask_no_gap()
        result = self._method().apply(clean, mask)
        np.testing.assert_allclose(result, np.clip(clean, 0, 1), atol=1e-6)

    def test_single_pixel_gap(self) -> None:
        clean = _clean_3d()
        mask = _mask_single_pixel()
        degraded = make_degraded(clean, mask)
        result = self._method().apply(degraded, mask)
        assert np.all(np.isfinite(result))

    def test_multichannel_support(self) -> None:
        clean = _clean_3d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 3
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 3

    def test_single_channel_support(self) -> None:
        clean = _clean_2d()
        mask = _mask()
        degraded = make_degraded(clean, mask)
        assert degraded.ndim == 2
        result = self._method().apply(degraded, mask)
        assert result.shape == degraded.shape
        assert result.ndim == 2
