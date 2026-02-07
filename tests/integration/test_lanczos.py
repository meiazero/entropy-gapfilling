"""Integration tests for LanczosInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.lanczos import LanczosInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestLanczosOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = LanczosInterpolator(a=3, max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = LanczosInterpolator(a=3, max_iterations=30)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-5
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = LanczosInterpolator(a=3, max_iterations=30)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_parameter_a_validation(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            LanczosInterpolator(a=0)

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = LanczosInterpolator(a=2, max_iterations=20)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
        assert np.all(np.isfinite(result))

    def test_a2_vs_a3_both_valid(self, sentinel2_patch: PatchSample) -> None:
        """Both a=2 and a=3 should produce valid reconstructions."""
        degraded = sentinel2_patch.degraded_inf
        mask = sentinel2_patch.mask
        for a in (2, 3):
            method = LanczosInterpolator(a=a, max_iterations=20)
            result = method.apply(degraded, mask)
            assert np.all(np.isfinite(result))
