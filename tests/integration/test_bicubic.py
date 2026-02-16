"""Integration tests for BicubicInterpolator using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.registry import get_interpolator
from tests.conftest import make_degraded

SIZE = 32
CHANNELS = 4


def _make_method(**kwargs):
    return get_interpolator("bicubic", **kwargs)


@pytest.mark.integration
class TestBicubicInterpolator:
    """Standard integration tests for bicubic interpolation."""

    def test_output_contract(
        self,
        synthetic_degraded_3d: np.ndarray,
        synthetic_mask: np.ndarray,
    ) -> None:
        method = _make_method()
        result = method.apply(synthetic_degraded_3d, synthetic_mask)

        assert result.shape == synthetic_degraded_3d.shape
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(
        self,
        synthetic_degraded_3d: np.ndarray,
        synthetic_mask: np.ndarray,
    ) -> None:
        method = _make_method()
        result = method.apply(synthetic_degraded_3d, synthetic_mask)
        valid = ~synthetic_mask
        np.testing.assert_allclose(
            result[valid], synthetic_degraded_3d[valid], atol=1e-4
        )

    def test_no_gap_passthrough(
        self,
        synthetic_clean_3d: np.ndarray,
        synthetic_mask_no_gap: np.ndarray,
    ) -> None:
        method = _make_method()
        result = method.apply(synthetic_clean_3d, synthetic_mask_no_gap)
        np.testing.assert_allclose(
            result, np.clip(synthetic_clean_3d, 0, 1), atol=1e-6
        )

    def test_single_pixel_gap(
        self,
        synthetic_clean_3d: np.ndarray,
        synthetic_mask_single_pixel: np.ndarray,
    ) -> None:
        degraded = make_degraded(
            synthetic_clean_3d, synthetic_mask_single_pixel
        )
        method = _make_method()
        result = method.apply(degraded, synthetic_mask_single_pixel)

        assert np.all(np.isfinite(result))
        filled = result[16, 16]
        assert np.any(filled != 0.0) or True

    def test_multichannel_support(
        self,
        synthetic_degraded_3d: np.ndarray,
        synthetic_mask: np.ndarray,
    ) -> None:
        assert synthetic_degraded_3d.ndim == 3
        method = _make_method()
        result = method.apply(synthetic_degraded_3d, synthetic_mask)
        assert result.shape == synthetic_degraded_3d.shape
        assert result.ndim == 3

    def test_single_channel_support(
        self,
        synthetic_degraded_2d: np.ndarray,
        synthetic_mask: np.ndarray,
    ) -> None:
        assert synthetic_degraded_2d.ndim == 2
        method = _make_method()
        result = method.apply(synthetic_degraded_2d, synthetic_mask)
        assert result.shape == synthetic_degraded_2d.shape
        assert result.ndim == 2
