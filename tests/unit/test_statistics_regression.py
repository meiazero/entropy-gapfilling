"""Unit tests for robust regression and aggregation additions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdi_pipeline.aggregation import summary_by_satellite
from pdi_pipeline.statistics import RegressionResult, robust_regression


def _make_fake_results(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic experiment results for testing."""
    rng = np.random.default_rng(seed)
    methods = ["bilinear", "bicubic", "kriging", "idw"]
    satellites = ["sentinel2", "landsat8"]
    noise_levels = ["inf", "40", "30"]

    rows = []
    for i in range(n):
        method = methods[i % len(methods)]
        sat = satellites[i % len(satellites)]
        noise = noise_levels[i % len(noise_levels)]
        entropy_7 = rng.uniform(1.0, 5.0)
        entropy_15 = rng.uniform(1.0, 5.0)
        psnr_val = 30.0 - 2.0 * entropy_7 + rng.normal(0, 1.0)
        rows.append({
            "method": method,
            "satellite": sat,
            "noise_level": noise,
            "entropy_7": entropy_7,
            "entropy_15": entropy_15,
            "psnr": psnr_val,
            "ssim": rng.uniform(0.7, 1.0),
            "rmse": rng.uniform(0.01, 0.2),
        })
    return pd.DataFrame(rows)


class TestRobustRegression:
    def test_returns_regression_result(self) -> None:
        df = _make_fake_results()
        result = robust_regression(df, metric_col="psnr")
        assert isinstance(result, RegressionResult)
        assert result.model_type == "rlm"
        assert result.n > 0

    def test_coefficient_table_has_expected_columns(self) -> None:
        df = _make_fake_results()
        result = robust_regression(df, metric_col="psnr")
        expected_cols = {
            "variable",
            "beta",
            "std_err",
            "z_value",
            "p_value",
            "ci_lo",
            "ci_hi",
        }
        assert expected_cols.issubset(set(result.coefficients.columns))

    def test_entropy_coefficients_present(self) -> None:
        df = _make_fake_results()
        result = robust_regression(
            df, metric_col="psnr", entropy_cols=["entropy_7"]
        )
        variables = result.coefficients["variable"].tolist()
        assert "entropy_7" in variables

    def test_r_squared_adj_is_reasonable(self) -> None:
        df = _make_fake_results()
        result = robust_regression(df, metric_col="psnr")
        # With synthetic data driven by entropy, R2 should be > 0
        assert result.r_squared_adj > 0.0
        assert result.r_squared_adj <= 1.0

    def test_vif_computed_for_entropy_cols(self) -> None:
        df = _make_fake_results()
        result = robust_regression(
            df,
            metric_col="psnr",
            entropy_cols=["entropy_7", "entropy_15"],
        )
        assert not result.vif.empty
        assert "vif" in result.vif.columns

    def test_too_few_rows_returns_empty(self) -> None:
        df = _make_fake_results(n=3)
        result = robust_regression(df, metric_col="psnr")
        assert result.coefficients.empty
        assert np.isnan(result.r_squared_adj)

    def test_single_entropy_col(self) -> None:
        df = _make_fake_results()
        result = robust_regression(
            df, metric_col="psnr", entropy_cols=["entropy_7"]
        )
        assert not result.coefficients.empty


class TestSummaryBySatellite:
    def test_returns_dataframe(self) -> None:
        df = _make_fake_results()
        result = summary_by_satellite(df, metric="psnr")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_has_expected_columns(self) -> None:
        df = _make_fake_results()
        result = summary_by_satellite(df, metric="psnr")
        expected = {"method", "satellite", "n", "mean", "ci95_lo", "ci95_hi"}
        assert expected.issubset(set(result.columns))

    def test_all_satellites_present(self) -> None:
        df = _make_fake_results()
        result = summary_by_satellite(df, metric="psnr")
        assert set(result["satellite"].unique()) == {"sentinel2", "landsat8"}

    def test_all_methods_present(self) -> None:
        df = _make_fake_results()
        result = summary_by_satellite(df, metric="psnr")
        assert set(result["method"].unique()) == {
            "bilinear",
            "bicubic",
            "kriging",
            "idw",
        }

    def test_ci_bounds_order(self) -> None:
        df = _make_fake_results()
        result = summary_by_satellite(df, metric="psnr")
        for _, row in result.iterrows():
            if not np.isnan(row["ci95_lo"]):
                assert row["ci95_lo"] <= row["ci95_hi"]
