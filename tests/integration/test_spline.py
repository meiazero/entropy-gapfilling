"""Integration tests for SplineInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.spline import SplineInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestSplineOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = SplineInterpolator(max_training_points=2000)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = SplineInterpolator(max_training_points=2000)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-4
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = SplineInterpolator(max_training_points=2000)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_smoothing_parameter(self, sentinel2_patch: PatchSample) -> None:
        exact = SplineInterpolator(smoothing=0.0, max_training_points=1000)
        smooth = SplineInterpolator(smoothing=0.5, max_training_points=1000)
        degraded = sentinel2_patch.degraded_inf
        mask = sentinel2_patch.mask
        r_exact = exact.apply(degraded, mask)
        r_smooth = smooth.apply(degraded, mask)
        assert np.all(np.isfinite(r_exact))
        assert np.all(np.isfinite(r_smooth))

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = SplineInterpolator(max_training_points=1000)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
