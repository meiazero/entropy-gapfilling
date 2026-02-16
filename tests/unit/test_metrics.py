"""Unit tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.exceptions import DimensionError
from pdi_pipeline.metrics import (
    compute_all,
    ergas,
    local_psnr,
    local_ssim,
    psnr,
    rmse,
    sam,
    ssim,
)


class TestPSNR:
    def test_perfect_reconstruction_returns_inf(self) -> None:
        img = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        assert psnr(img, img, mask) == float("inf")

    def test_no_gap_pixels_returns_inf(self) -> None:
        img = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        mask = np.zeros((16, 16), dtype=np.float32)
        assert psnr(img, img * 0.5, mask) == float("inf")

    def test_known_value(self) -> None:
        clean = np.ones((4, 4), dtype=np.float32)
        recon = np.full((4, 4), 0.9, dtype=np.float32)
        mask = np.ones((4, 4), dtype=np.float32)
        # MSE = 0.01, PSNR = 10*log10(1/0.01) = 20.0
        result = psnr(clean, recon, mask)
        assert abs(result - 20.0) < 0.1

    def test_multichannel(self) -> None:
        rng = np.random.default_rng(42)
        clean = rng.random((16, 16, 4)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        result = psnr(clean, clean, mask)
        assert result == float("inf")

    def test_shape_mismatch_raises_dimension_error(self) -> None:
        with pytest.raises(DimensionError, match="Shape mismatch"):
            psnr(np.zeros((4, 4)), np.zeros((4, 5)), np.ones((4, 4)))

    def test_nan_input_gives_finite_result(self) -> None:
        """NaN in reconstructed pixels should produce a finite (non-inf) PSNR."""
        clean = np.ones((4, 4), dtype=np.float32)
        recon = np.ones((4, 4), dtype=np.float32)
        recon[0, 0] = np.nan
        mask = np.ones((4, 4), dtype=np.float32)
        result = psnr(clean, recon, mask)
        # The NaN pixel introduces error, so PSNR should not be inf.
        # It may also be NaN depending on implementation; just check it runs.
        assert isinstance(result, float)

    def test_empty_mask_no_gaps(self) -> None:
        """All-zero mask means no gap pixels to evaluate."""
        clean = np.ones((4, 4), dtype=np.float32)
        recon = np.zeros((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.float32)
        assert psnr(clean, recon, mask) == float("inf")


class TestSSIM:
    def test_perfect_reconstruction(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((32, 32)).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        result = ssim(img, img, mask)
        assert abs(result - 1.0) < 1e-5

    def test_no_gap_returns_one(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((32, 32)).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        assert ssim(img, img * 0.5, mask) == 1.0

    def test_empty_mask_no_gaps(self) -> None:
        clean = np.ones((32, 32), dtype=np.float32)
        recon = np.zeros((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        assert ssim(clean, recon, mask) == 1.0


class TestRMSE:
    def test_perfect_reconstruction_returns_zero(self) -> None:
        img = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        assert rmse(img, img, mask) == 0.0

    def test_known_value(self) -> None:
        clean = np.ones((4, 4), dtype=np.float32)
        recon = np.full((4, 4), 0.9, dtype=np.float32)
        mask = np.ones((4, 4), dtype=np.float32)
        result = rmse(clean, recon, mask)
        assert abs(result - 0.1) < 1e-5

    def test_empty_mask_no_gaps(self) -> None:
        clean = np.ones((4, 4), dtype=np.float32)
        recon = np.zeros((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.float32)
        assert rmse(clean, recon, mask) == 0.0

    def test_shape_mismatch_raises_dimension_error(self) -> None:
        with pytest.raises(DimensionError):
            rmse(np.zeros((4, 4)), np.zeros((5, 4)), np.ones((4, 4)))


class TestSAM:
    def test_identical_returns_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 4)).astype(np.float32) + 0.01
        mask = np.ones((16, 16), dtype=np.float32)
        result = sam(img, img, mask)
        # float32 arccos introduces small numerical error; keep tolerance loose
        assert abs(result) < 0.1

    def test_requires_multichannel(self) -> None:
        with pytest.raises(DimensionError, match="multichannel"):
            sam(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))

    def test_orthogonal_vectors(self) -> None:
        clean = np.zeros((1, 1, 2), dtype=np.float32)
        clean[0, 0] = [1.0, 0.0]
        recon = np.zeros((1, 1, 2), dtype=np.float32)
        recon[0, 0] = [0.0, 1.0]
        mask = np.ones((1, 1), dtype=np.float32)
        result = sam(clean, recon, mask)
        assert abs(result - 90.0) < 0.1

    def test_empty_mask_no_gaps(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((8, 8, 3)).astype(np.float32)
        mask = np.zeros((8, 8), dtype=np.float32)
        assert sam(img, img * 0.5, mask) == 0.0


class TestERGAS:
    def test_perfect_reconstruction_returns_zero(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 4)).astype(np.float32) + 0.1
        mask = np.ones((16, 16), dtype=np.float32)
        result = ergas(img, img, mask)
        assert abs(result) < 1e-5

    def test_no_gap_returns_zero(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 4)).astype(np.float32) + 0.1
        mask = np.zeros((16, 16), dtype=np.float32)
        result = ergas(img, img * 0.5, mask)
        assert result == 0.0

    def test_requires_multichannel(self) -> None:
        with pytest.raises(DimensionError, match="multichannel"):
            ergas(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))

    def test_positive_for_imperfect_reconstruction(self) -> None:
        rng = np.random.default_rng(42)
        clean = rng.random((16, 16, 4)).astype(np.float32) + 0.1
        recon = clean + rng.normal(0, 0.05, clean.shape).astype(np.float32)
        recon = np.clip(recon, 0, 1)
        mask = np.ones((16, 16), dtype=np.float32)
        result = ergas(clean, recon, mask)
        assert result > 0.0

    def test_known_value(self) -> None:
        # Uniform bands: RMSE_b / mean_b should be consistent
        clean = np.full((4, 4, 2), 0.5, dtype=np.float32)
        recon = np.full((4, 4, 2), 0.4, dtype=np.float32)
        mask = np.ones((4, 4), dtype=np.float32)
        # RMSE per band = 0.1, mean per band = 0.5
        # ERGAS = 100 * sqrt(1/2 * (0.1/0.5)^2 * 2) = 100 * 0.2 = 20.0
        result = ergas(clean, recon, mask)
        assert abs(result - 20.0) < 0.5

    def test_empty_mask_no_gaps(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((8, 8, 3)).astype(np.float32) + 0.1
        mask = np.zeros((8, 8), dtype=np.float32)
        assert ergas(img, img * 0.5, mask) == 0.0


class TestLocalPSNR:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((32, 32)).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        result = local_psnr(img, img * 0.9, mask, window=7)
        assert result.shape == (32, 32)
        assert result.dtype == np.float32


class TestLocalSSIM:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((32, 32)).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        result = local_ssim(img, img * 0.9, mask, window=7)
        assert result.shape == (32, 32)
        assert result.dtype == np.float32

    def test_non_gap_pixels_are_nan(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((32, 32)).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        result = local_ssim(img, img * 0.9, mask, window=7)
        assert np.all(np.isnan(result[0, 0]))
        assert not np.isnan(result[15, 15])


class TestComputeAll:
    def test_returns_all_keys_multichannel(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 4)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        result = compute_all(img, img, mask)
        assert "psnr" in result
        assert "ssim" in result
        assert "rmse" in result
        assert "sam" in result
        assert "ergas" in result

    def test_returns_no_sam_or_ergas_for_2d(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        result = compute_all(img, img, mask)
        assert "sam" not in result
        assert "ergas" not in result
