"""Unit tests for the entropy module."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.entropy import shannon_entropy
from pdi_pipeline.exceptions import DimensionError, ValidationError


class TestShannonEntropy:
    """Tests for shannon_entropy()."""

    def test_constant_image_returns_zero(self) -> None:
        img = np.full((32, 32), 0.5, dtype=np.float32)
        result = shannon_entropy(img, window_size=7)
        assert result.shape == (32, 32)
        assert np.allclose(result, 0.0)

    def test_output_shape_matches_input_2d(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64), dtype=np.float64)
        for ws in (7, 15, 31):
            result = shannon_entropy(img, window_size=ws)
            assert result.shape == (64, 64)
            assert result.dtype == np.float32

    def test_output_shape_matches_input_3d(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 4)).astype(np.float32)
        result = shannon_entropy(img, window_size=7)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_random_image_has_positive_entropy(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64), dtype=np.float64)
        result = shannon_entropy(img, window_size=7)
        interior = result[4:-4, 4:-4]
        assert np.all(interior > 0.0)

    def test_larger_window_smoother_output(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64), dtype=np.float64)
        e7 = shannon_entropy(img, window_size=7)
        e31 = shannon_entropy(img, window_size=31)
        # Larger window should produce a smoother (lower variance) map
        assert float(np.std(e31)) < float(np.std(e7))

    def test_multichannel_uses_mean(self) -> None:
        rng = np.random.default_rng(42)
        img_3d = rng.random((32, 32, 4)).astype(np.float32)
        mean_band = np.mean(img_3d, axis=2)
        e_3d = shannon_entropy(img_3d, window_size=7)
        e_2d = shannon_entropy(mean_band, window_size=7)
        np.testing.assert_array_equal(e_3d, e_2d)

    def test_invalid_even_window_raises_validation_error(self) -> None:
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValidationError, match="positive odd integer"):
            shannon_entropy(img, window_size=8)

    def test_invalid_zero_window_raises_validation_error(self) -> None:
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValidationError, match="positive odd integer"):
            shannon_entropy(img, window_size=0)

    def test_negative_window_raises_validation_error(self) -> None:
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValidationError, match="positive odd integer"):
            shannon_entropy(img, window_size=-3)

    def test_invalid_ndim_raises_dimension_error(self) -> None:
        img = np.zeros((4,), dtype=np.float32)
        with pytest.raises(DimensionError, match="2D or 3D"):
            shannon_entropy(img, window_size=7)

    def test_4d_raises_dimension_error(self) -> None:
        img = np.zeros((4, 4, 3, 2), dtype=np.float32)
        with pytest.raises(DimensionError, match="2D or 3D"):
            shannon_entropy(img, window_size=7)

    def test_entropy_values_non_negative(self) -> None:
        rng = np.random.default_rng(123)
        img = rng.random((32, 32), dtype=np.float64)
        result = shannon_entropy(img, window_size=7)
        assert np.all(result >= 0.0)

    def test_validation_error_is_also_value_error(self) -> None:
        """ValidationError inherits ValueError, so callers can catch either."""
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="positive odd integer"):
            shannon_entropy(img, window_size=4)

    def test_dimension_error_is_also_value_error(self) -> None:
        """DimensionError inherits ValueError, so callers can catch either."""
        img = np.zeros((4,), dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            shannon_entropy(img, window_size=7)
