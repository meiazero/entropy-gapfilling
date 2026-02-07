"""Integration tests for NonLocalMeans and ExemplarBased on real patches."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.patch_based import (
    ExemplarBasedInterpolator,
    NonLocalMeansInterpolator,
)
from tests.conftest import (
    NOISE_VARIANTS,
    PatchSample,
    degraded_for_variant,
    psnr,
)

# -- NonLocalMeans ------------------------------------------------------------


@pytest.mark.integration
class TestNonLocalMeansOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NonLocalMeansInterpolator(
            patch_size=5, patch_distance=6, h_rel=0.8
        )
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = NonLocalMeansInterpolator()
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-4
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = NonLocalMeansInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_multichannel_support(self, sentinel2_patch: PatchSample) -> None:
        method = NonLocalMeansInterpolator(patch_size=3, patch_distance=4)
        degraded = sentinel2_patch.degraded_inf
        assert degraded.ndim == 3
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = NonLocalMeansInterpolator(patch_size=3, patch_distance=4)
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape


# -- ExemplarBased ------------------------------------------------------------


@pytest.mark.integration
class TestExemplarBasedOnRealData:
    @pytest.fixture(params=NOISE_VARIANTS)
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_output_contract(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = ExemplarBasedInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_known_pixels_preserved(self, sentinel2_patch: PatchSample) -> None:
        method = ExemplarBasedInterpolator()
        degraded = sentinel2_patch.degraded_inf
        mask_bool = sentinel2_patch.mask.astype(bool)
        result = method.apply(degraded, sentinel2_patch.mask)
        np.testing.assert_allclose(
            result[~mask_bool], degraded[~mask_bool], atol=1e-4
        )

    def test_psnr_positive(
        self, sentinel2_patch: PatchSample, variant: str
    ) -> None:
        method = ExemplarBasedInterpolator()
        degraded = degraded_for_variant(sentinel2_patch, variant)
        result = method.apply(degraded, sentinel2_patch.mask)
        score = psnr(sentinel2_patch.clean, result)
        assert score > 0.0

    def test_multichannel_support(self, sentinel2_patch: PatchSample) -> None:
        method = ExemplarBasedInterpolator()
        degraded = sentinel2_patch.degraded_inf
        assert degraded.ndim == 3
        result = method.apply(degraded, sentinel2_patch.mask)
        assert result.shape == degraded.shape

    def test_single_channel(self, sentinel2_patch: PatchSample) -> None:
        method = ExemplarBasedInterpolator()
        single = sentinel2_patch.degraded_inf[:, :, 0]
        result = method.apply(single, sentinel2_patch.mask)
        assert result.shape == single.shape
