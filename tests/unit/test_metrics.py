"""Unit tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.metrics import (
    compute_all,
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

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            psnr(np.zeros((4, 4)), np.zeros((4, 5)), np.ones((4, 4)))


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


class TestSAM:
    def test_identical_returns_zero(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 4)).astype(np.float32) + 0.01
        mask = np.ones((16, 16), dtype=np.float32)
        result = sam(img, img, mask)
        assert abs(result) < 1e-5

    def test_requires_multichannel(self) -> None:
        with pytest.raises(ValueError, match="multichannel"):
            sam(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))

    def test_orthogonal_vectors(self) -> None:
        clean = np.zeros((1, 1, 2), dtype=np.float32)
        clean[0, 0] = [1.0, 0.0]
        recon = np.zeros((1, 1, 2), dtype=np.float32)
        recon[0, 0] = [0.0, 1.0]
        mask = np.ones((1, 1), dtype=np.float32)
        result = sam(clean, recon, mask)
        assert abs(result - 90.0) < 0.1


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

    def test_returns_no_sam_for_2d(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        result = compute_all(img, img, mask)
        assert "sam" not in result
