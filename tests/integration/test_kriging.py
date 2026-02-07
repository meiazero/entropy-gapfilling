"""Integration tests for KrigingInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.kriging import KrigingInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestKrigingOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = KrigingInterpolator(
            variogram_model="spherical",
            max_points=500,
        )
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = KrigingInterpolator(max_points=500)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-5
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = KrigingInterpolator(max_points=500)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    @pytest.mark.parametrize(
        "variogram",
        ["spherical", "exponential", "gaussian", "linear"],
    )
    def test_variogram_models(
        self, sentinel2_patch: PatchSample, variogram: str
    ) -> None:
        method = KrigingInterpolator(variogram_model=variogram, max_points=300)
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        assert np.all(np.isfinite(result))

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = KrigingInterpolator(max_points=300)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
        assert np.all(np.isfinite(result))
