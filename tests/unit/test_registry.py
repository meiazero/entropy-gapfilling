"""Unit tests for the method registry and factory."""

from __future__ import annotations

import pytest

from pdi_pipeline.exceptions import ConfigError
from pdi_pipeline.methods.base import BaseMethod
from pdi_pipeline.methods.compressive import (
    L1DCTInpainting,
    L1WaveletInpainting,
)
from pdi_pipeline.methods.nearest import NearestInterpolator
from pdi_pipeline.methods.registry import (
    get_interpolator,
    list_categories,
    list_methods,
)


class TestGetInterpolator:
    def test_nearest_returns_correct_type(self) -> None:
        interp = get_interpolator("nearest")
        assert isinstance(interp, BaseMethod)
        assert interp.name == "nearest"

    def test_nearest_class_name(self) -> None:
        interp = get_interpolator("nearest")
        assert isinstance(interp, NearestInterpolator)

    def test_alias_l1_dct_resolves(self) -> None:
        interp = get_interpolator("l1_dct")
        assert isinstance(interp, BaseMethod)
        assert isinstance(interp, L1DCTInpainting)

    def test_alias_l1_wavelet_resolves(self) -> None:
        interp = get_interpolator("l1_wavelet")
        assert isinstance(interp, BaseMethod)
        assert isinstance(interp, L1WaveletInpainting)

    def test_unknown_name_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown method"):
            get_interpolator("unknown_method_xyz")

    def test_unknown_name_is_also_value_error(self) -> None:
        """ConfigError inherits ValueError, so callers can catch either."""
        with pytest.raises(ValueError, match="Unknown method"):
            get_interpolator("unknown_method_xyz")

    def test_kwargs_forwarded(self) -> None:
        interp = get_interpolator("nearest", kernel_size=5)
        assert interp.kernel_size == 5  # type: ignore[attr-defined]


class TestListMethods:
    def test_returns_sorted_list(self) -> None:
        methods = list_methods()
        assert isinstance(methods, list)
        assert methods == sorted(methods)

    def test_contains_known_methods(self) -> None:
        methods = list_methods()
        for name in ("nearest", "bilinear", "bicubic", "kriging", "cs_dct"):
            assert name in methods

    def test_no_aliases_in_list(self) -> None:
        """list_methods returns canonical names only, not aliases."""
        methods = list_methods()
        assert "l1_dct" not in methods
        assert "l1_wavelet" not in methods


class TestListCategories:
    def test_returns_dict(self) -> None:
        cats = list_categories()
        assert isinstance(cats, dict)

    def test_expected_category_keys(self) -> None:
        cats = list_categories()
        expected = {
            "spatial",
            "kernel",
            "geostatistical",
            "transform",
            "compressive",
            "patch_based",
        }
        assert set(cats.keys()) == expected

    def test_spatial_contains_nearest(self) -> None:
        cats = list_categories()
        assert "nearest" in cats["spatial"]

    def test_values_are_lists_of_strings(self) -> None:
        cats = list_categories()
        for key, methods in cats.items():
            assert isinstance(methods, list), f"{key} value is not a list"
            for m in methods:
                assert isinstance(m, str), f"{m} in {key} is not a string"
