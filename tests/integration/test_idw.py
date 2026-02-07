"""Integration tests for IDWInterpolator on real satellite patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.idw import IDWInterpolator
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)


@pytest.mark.integration
class TestIDWOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = IDWInterpolator(power=2.0, kernel_size=16)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = IDWInterpolator(power=2.0)
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-6
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = IDWInterpolator(power=2.0, kernel_size=16)
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    @pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
    def test_different_power_params(
        self, sentinel2_patch: PatchSample, power: float
    ) -> None:
        method = IDWInterpolator(power=power, kernel_size=16)
        result = method.apply(
            sentinel2_patch.degraded_inf, sentinel2_patch.mask
        )
        assert np.all(np.isfinite(result))

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = IDWInterpolator(power=2.0, kernel_size=16)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
        assert np.all(np.isfinite(result))
