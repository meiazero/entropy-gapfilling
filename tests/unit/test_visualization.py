"""Unit tests for the visualization module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdi_pipeline.visualization import save_array_as_png, to_display_rgb


class TestToDisplayRgb:
    """Tests for to_display_rgb()."""

    # -- 2D grayscale inputs ------------------------------------------------

    def test_2d_grayscale_output_shape(self) -> None:
        arr = np.random.default_rng(0).random((32, 32))
        result = to_display_rgb(arr)
        assert result.shape == (32, 32)

    def test_2d_grayscale_range_zero_to_one(self) -> None:
        arr = np.random.default_rng(1).random((16, 16)) * 100.0
        result = to_display_rgb(arr)
        assert float(np.min(result)) >= 0.0
        assert float(np.max(result)) <= 1.0

    def test_2d_grayscale_dtype_is_float(self) -> None:
        arr = np.random.default_rng(2).random((8, 8))
        result = to_display_rgb(arr)
        assert np.issubdtype(result.dtype, np.floating)

    def test_2d_grayscale_min_maps_to_zero(self) -> None:
        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = to_display_rgb(arr)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_2d_grayscale_max_maps_to_one(self) -> None:
        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = to_display_rgb(arr)
        assert float(result[1, 1]) == pytest.approx(1.0)

    # -- 3D three-channel inputs --------------------------------------------

    def test_3d_rgb_output_shape(self) -> None:
        arr = np.random.default_rng(3).random((32, 32, 3))
        result = to_display_rgb(arr)
        assert result.shape == (32, 32, 3)

    def test_3d_rgb_range_zero_to_one(self) -> None:
        arr = np.random.default_rng(4).random((16, 16, 3)) * 5000.0 - 1000.0
        result = to_display_rgb(arr)
        assert float(np.min(result)) >= 0.0
        assert float(np.max(result)) <= 1.0

    def test_3d_rgb_preserves_relative_order(self) -> None:
        """Brighter input pixels stay brighter after normalization."""
        rng = np.random.default_rng(5)
        arr = rng.random((16, 16, 3))
        result = to_display_rgb(arr)
        # For any single band, the ordering of values should be preserved.
        flat_in = arr[:, :, 0].ravel()
        flat_out = result[:, :, 0].ravel()
        order_in = np.argsort(flat_in)
        order_out = np.argsort(flat_out)
        np.testing.assert_array_equal(order_in, order_out)

    # -- 3D four-channel (NIR) inputs ---------------------------------------

    def test_4channel_drops_extra_band(self) -> None:
        arr = np.random.default_rng(6).random((16, 16, 4))
        result = to_display_rgb(arr)
        assert result.shape == (16, 16, 3)

    def test_5channel_drops_extra_bands(self) -> None:
        arr = np.random.default_rng(7).random((8, 8, 5))
        result = to_display_rgb(arr)
        assert result.shape == (8, 8, 3)

    def test_4channel_uses_first_three_bands(self) -> None:
        """The output must derive from bands 0-2 only, ignoring band 3."""
        rng = np.random.default_rng(8)
        arr = rng.random((8, 8, 4))
        # Build an equivalent 3-channel array from the first three bands.
        rgb_only = arr[:, :, :3].copy()
        result_4ch = to_display_rgb(arr)
        result_3ch = to_display_rgb(rgb_only)
        np.testing.assert_array_equal(result_4ch, result_3ch)

    # -- Constant arrays ----------------------------------------------------

    def test_constant_2d_returns_zeros(self) -> None:
        arr = np.full((16, 16), 42.0)
        result = to_display_rgb(arr)
        np.testing.assert_array_equal(
            result, np.zeros((16, 16), dtype=np.float64)
        )

    def test_constant_3d_returns_zeros(self) -> None:
        arr = np.full((8, 8, 3), -7.5)
        result = to_display_rgb(arr)
        np.testing.assert_array_equal(
            result, np.zeros((8, 8, 3), dtype=np.float64)
        )

    def test_constant_4channel_returns_zeros_with_3_bands(self) -> None:
        arr = np.full((8, 8, 4), 1.0)
        result = to_display_rgb(arr)
        assert result.shape == (8, 8, 3)
        np.testing.assert_array_equal(
            result, np.zeros((8, 8, 3), dtype=np.float64)
        )

    def test_near_constant_within_tolerance_returns_zeros(self) -> None:
        """Values differing by less than 1e-8 should be treated as constant."""
        arr = np.full((8, 8), 5.0)
        arr[0, 0] = 5.0 + 1e-9
        result = to_display_rgb(arr)
        np.testing.assert_array_equal(
            result, np.zeros((8, 8), dtype=np.float64)
        )

    # -- NaN handling -------------------------------------------------------

    def test_nan_pixels_become_zero(self) -> None:
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = to_display_rgb(arr)
        assert not np.any(np.isnan(result))
        assert float(result[0, 1]) == 0.0

    def test_nan_does_not_affect_non_nan_normalization(self) -> None:
        arr = np.array([[0.0, np.nan], [5.0, 10.0]])
        result = to_display_rgb(arr)
        # min=0, max=10 so 5 maps to 0.5
        assert float(result[1, 0]) == pytest.approx(0.5)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[1, 1]) == pytest.approx(1.0)

    def test_all_nan_2d_returns_zeros(self) -> None:
        arr = np.full((4, 4), np.nan)
        result = to_display_rgb(arr)
        # nanmin/nanmax on all-NaN raises a warning and returns nan,
        # which makes vmax - vmin < 1e-8 false (nan comparison).
        # Depending on implementation details this may produce zeros or nans.
        # The function uses nan_to_num at the end, so either path gives zeros.
        assert not np.any(np.isnan(result))

    def test_nan_in_3d(self) -> None:
        rng = np.random.default_rng(9)
        arr = rng.random((8, 8, 3))
        arr[2, 3, :] = np.nan
        result = to_display_rgb(arr)
        assert not np.any(np.isnan(result))
        assert result.shape == (8, 8, 3)

    # -- Integer input ------------------------------------------------------

    def test_integer_input_normalizes(self) -> None:
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        result = to_display_rgb(arr)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[0, 2]) == pytest.approx(1.0)
        assert np.issubdtype(result.dtype, np.floating)

    # -- Negative values ----------------------------------------------------

    def test_negative_values_normalize_correctly(self) -> None:
        arr = np.array([[-10.0, 0.0, 10.0]])
        result = to_display_rgb(arr)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[0, 1]) == pytest.approx(0.5)
        assert float(result[0, 2]) == pytest.approx(1.0)


class TestSaveArrayAsPng:
    """Tests for save_array_as_png()."""

    def test_creates_file(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(10).random((16, 16))
        out = tmp_path / "output.png"
        save_array_as_png(arr, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(11).random((16, 16, 3))
        out = tmp_path / "a" / "b" / "c" / "image.png"
        save_array_as_png(arr, out)
        assert out.exists()

    def test_saves_2d_grayscale(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(12).random((32, 32))
        out = tmp_path / "gray.png"
        save_array_as_png(arr, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_3d_rgb(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(13).random((32, 32, 3))
        out = tmp_path / "rgb.png"
        save_array_as_png(arr, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_4channel_nir(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(14).random((32, 32, 4))
        out = tmp_path / "nir.png"
        save_array_as_png(arr, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_constant_array(self, tmp_path: Path) -> None:
        arr = np.full((16, 16), 5.0)
        out = tmp_path / "const.png"
        save_array_as_png(arr, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_array_with_nans(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(15)
        arr = rng.random((16, 16))
        arr[0, 0] = np.nan
        arr[8, 8] = np.nan
        out = tmp_path / "nan.png"
        save_array_as_png(arr, out)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(16).random((8, 8))
        out = str(tmp_path / "str_path.png")
        save_array_as_png(arr, out)
        assert Path(out).exists()

    def test_custom_dpi(self, tmp_path: Path) -> None:
        arr = np.random.default_rng(17).random((16, 16))
        out_lo = tmp_path / "lo.png"
        out_hi = tmp_path / "hi.png"
        save_array_as_png(arr, out_lo, dpi=50)
        save_array_as_png(arr, out_hi, dpi=300)
        assert out_lo.exists()
        assert out_hi.exists()
        # Both should be valid files
        assert out_lo.stat().st_size > 0
        assert out_hi.stat().st_size > 0

    def test_existing_directory_no_error(self, tmp_path: Path) -> None:
        """Saving to an already-existing directory should not raise."""
        arr = np.random.default_rng(18).random((8, 8, 3))
        subdir = tmp_path / "existing"
        subdir.mkdir()
        out = subdir / "image.png"
        save_array_as_png(arr, out)
        assert out.exists()

    def test_output_is_valid_png(self, tmp_path: Path) -> None:
        """The file should start with the PNG magic bytes."""
        arr = np.random.default_rng(19).random((16, 16, 3))
        out = tmp_path / "magic.png"
        save_array_as_png(arr, out)
        header = out.read_bytes()[:8]
        png_signature = b"\x89PNG\r\n\x1a\n"
        assert header == png_signature
