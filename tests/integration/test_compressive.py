"""Integration tests for L1 Wavelet and L1 DCT CS on real patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.compressive import (
    L1DCTInpainting,
    L1WaveletInpainting,
)
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)

# ── L1 Wavelet ───────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestL1WaveletOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = L1WaveletInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = L1WaveletInpainting(max_iterations=30)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-4
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = L1WaveletInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_multichannel_support(self, sentinel2_patch: PatchSample) -> None:
        """Verify that multichannel (H, W, C) input is accepted."""
        method = L1WaveletInpainting(max_iterations=20)
        degraded = sentinel2_patch.degraded_inf
        assert degraded.ndim == 3
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = L1WaveletInpainting(max_iterations=20)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape


# ── L1 DCT ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestL1DCTOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = L1DCTInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = L1DCTInpainting(max_iterations=30)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-4
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = L1DCTInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_multichannel_support(self, sentinel2_patch: PatchSample) -> None:
        method = L1DCTInpainting(max_iterations=20)
        degraded = sentinel2_patch.degraded_inf
        assert degraded.ndim == 3
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = L1DCTInpainting(max_iterations=20)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
