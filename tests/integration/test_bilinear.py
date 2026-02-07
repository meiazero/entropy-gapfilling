"""Integration tests for BilinearInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.bilinear import BilinearInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestBilinearOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = BilinearInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = BilinearInterpolator()
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-6
        )

    def test_gaps_are_filled(self, sentinel2_patch: PatchSample) -> None:
        method = BilinearInterpolator()
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        assert np.all(np.isfinite(result[sentinel2_patch.mask.astype(bool)]))

    def test_psnr_above_nearest(self, sentinel2_patch: PatchSample) -> None:
        """Bilinear should generally match or exceed nearest-neighbour."""
        from pdi_pipeline.methods.nearest import NearestInterpolator

        degraded = sentinel2_patch.degraded_inf
        mask = sentinel2_patch.mask
        clean = sentinel2_patch.clean

        nn_result = NearestInterpolator().apply(degraded, mask)
        bl_result = BilinearInterpolator().apply(degraded, mask)

        nn_psnr = psnr(clean, nn_result)
        bl_psnr = psnr(clean, bl_result)
        # Bilinear should be at least as good; allow small tolerance
        assert bl_psnr >= nn_psnr - 1.0

    def test_no_gaps_passthrough(self, sentinel2_patch: PatchSample) -> None:
        method = BilinearInterpolator()
        empty_mask = np.zeros_like(sentinel2_patch.mask)
        result = method.apply(sentinel2_patch.clean, empty_mask)
        np.testing.assert_allclose(
            result, np.clip(sentinel2_patch.clean, 0, 1), atol=1e-6
        )

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = BilinearInterpolator()
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
        assert result.dtype == np.float32
