"""Integration tests for DCT, Wavelet, and TV inpainting on real patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.transforms import (
    DCTInpainting,
    TVInpainting,
    WaveletInpainting,
)
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)

# ── DCT ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestDCTOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = DCTInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = DCTInpainting(max_iterations=30)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-5
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = DCTInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = DCTInpainting(max_iterations=20)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape


# ── Wavelet ──────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestWaveletOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = WaveletInpainting(wavelet="db4", level=3, max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = WaveletInpainting(max_iterations=30)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-5
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = WaveletInpainting(max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    @pytest.mark.parametrize("wavelet", ["db4", "haar", "sym4"])
    def test_wavelet_families(
        self, sentinel2_patch: PatchSample, wavelet: str
    ) -> None:
        method = WaveletInpainting(wavelet=wavelet, level=2, max_iterations=20)
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        assert np.all(np.isfinite(result))


# ── TV ───────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestTVOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = TVInpainting(lambda_param=0.1, max_iterations=50)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = TVInpainting(lambda_param=10.0, max_iterations=100)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        # TV with high lambda should enforce strong fidelity
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=0.05
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = TVInpainting(max_iterations=50)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = TVInpainting(max_iterations=30)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
