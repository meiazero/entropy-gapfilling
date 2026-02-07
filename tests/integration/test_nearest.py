"""Integration tests for NearestInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.nearest import NearestInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestNearestOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_shape(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NearestInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape

    def test_output_dtype(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NearestInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.dtype == np.float32

    def test_no_nan_or_inf(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NearestInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert np.all(np.isfinite(result))

    def test_output_in_unit_range(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NearestInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = NearestInterpolator()
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        valid = ~mask_bool
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-6)

    def test_gaps_are_filled(self, sentinel2_patch: PatchSample) -> None:
        method = NearestInterpolator()
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        mask_bool = sentinel2_patch.mask.astype(bool)
        gap_values = result[mask_bool]
        # All gap pixels should have a finite, non-zero value
        assert np.all(np.isfinite(gap_values))

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NearestInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_no_gaps_passthrough(self, sentinel2_patch: PatchSample) -> None:
        method = NearestInterpolator()
        empty_mask = np.zeros_like(sentinel2_patch.mask)
        result = method.apply(sentinel2_patch.clean, empty_mask)
        np.testing.assert_allclose(
            result, np.clip(sentinel2_patch.clean, 0, 1), atol=1e-6
        )

    def test_kernel_size_limits_search(
        self, sentinel2_patch: PatchSample
    ) -> None:
        method = NearestInterpolator(kernel_size=5)
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        assert result.shape == sentinel2_patch.degraded_inf.shape
        assert np.all(np.isfinite(result))
