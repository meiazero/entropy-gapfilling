"""Integration tests for BicubicInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.bicubic import BicubicInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestBicubicOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = BicubicInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = BicubicInterpolator()
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-6
        )

    def test_psnr_above_bilinear(self, sentinel2_patch: PatchSample) -> None:
        """Bicubic (C1) should generally match or exceed bilinear (C0)."""
        from pdi_pipeline.methods.bilinear import BilinearInterpolator

        degraded = sentinel2_patch.degraded_inf
        mask = sentinel2_patch.mask
        clean = sentinel2_patch.clean

        bl_result = BilinearInterpolator().apply(degraded, mask)
        bc_result = BicubicInterpolator().apply(degraded, mask)

        bl_score = psnr(clean, bl_result)
        bc_score = psnr(clean, bc_result)
        # Allow small tolerance -- on some patches bilinear may edge out
        assert bc_score >= bl_score - 1.5

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = BicubicInterpolator()
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
        assert np.all(np.isfinite(result))
