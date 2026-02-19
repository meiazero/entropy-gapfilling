"""Unit tests for the statistics module.

Covers correlation analysis, correlation matrix with FDR correction,
non-parametric method comparison (Kruskal-Wallis + Dunn post-hoc),
robust regression (RLM HuberT with VIF), and spatial autocorrelation
(Moran's I global + LISA local).

All tests use synthetic data built from numpy random generators.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pdi_pipeline.statistics import (
    ComparisonResult,
    CorrelationResult,
    RegressionResult,
    SpatialResult,
    correlation_analysis,
    correlation_matrix,
    method_comparison,
    robust_regression,
    spatial_autocorrelation,
)

# ---------------------------------------------------------------------------
# Helpers -- synthetic data generators
# ---------------------------------------------------------------------------


def _make_corr_df(
    n: int = 100,
    seed: int = 42,
    *,
    inject_nans: bool = False,
) -> pd.DataFrame:
    """Create a DataFrame with correlated entropy and metric columns."""
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(1.0, 5.0, size=n)
    # Metric negatively correlated with entropy, plus noise.
    metric = 30.0 - 2.0 * entropy + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"entropy": entropy, "metric": metric})
    if inject_nans:
        df.loc[0, "entropy"] = np.nan
        df.loc[1, "metric"] = np.nan
    return df


def _make_method_df(
    n_per_method: int = 50,
    methods: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a DataFrame with a 'method' column and metric values."""
    rng = np.random.default_rng(seed)
    if methods is None:
        methods = ["bilinear", "bicubic", "kriging"]
    rows = []
    for i, m in enumerate(methods):
        # Shift means so Kruskal-Wallis can detect a difference.
        vals = rng.normal(loc=20.0 + 3.0 * i, scale=1.0, size=n_per_method)
        for v in vals:
            rows.append({"method": m, "psnr": v})
    return pd.DataFrame(rows)


def _make_regression_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame suitable for robust_regression."""
    rng = np.random.default_rng(seed)
    methods = ["bilinear", "bicubic", "kriging", "idw"]
    noise_levels = ["inf", "40", "30"]
    rows = []
    for i in range(n):
        entropy_7 = rng.uniform(1.0, 5.0)
        entropy_15 = rng.uniform(1.0, 5.0)
        psnr_val = (
            30.0 - 2.0 * entropy_7 + 0.5 * entropy_15 + rng.normal(0, 1.0)
        )
        rows.append({
            "method": methods[i % len(methods)],
            "noise_level": noise_levels[i % len(noise_levels)],
            "entropy_7": entropy_7,
            "entropy_15": entropy_15,
            "psnr": psnr_val,
        })
    return pd.DataFrame(rows)


def _make_matrix_df(
    n_per_method: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a DataFrame with method, entropy, and metric columns for
    correlation_matrix tests."""
    rng = np.random.default_rng(seed)
    methods = ["bilinear", "bicubic"]
    rows = []
    for m in methods:
        for _ in range(n_per_method):
            e7 = rng.uniform(1.0, 5.0)
            e15 = rng.uniform(1.0, 5.0)
            psnr_val = 30.0 - 2.0 * e7 + rng.normal(0, 0.5)
            ssim_val = 0.95 - 0.02 * e15 + rng.normal(0, 0.005)
            rows.append({
                "method": m,
                "entropy_7": e7,
                "entropy_15": e15,
                "psnr": psnr_val,
                "ssim": ssim_val,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CorrelationResult dataclass
# ---------------------------------------------------------------------------


class TestCorrelationResult:
    def test_fields_are_accessible(self) -> None:
        cr = CorrelationResult(
            pearson_r=0.9,
            pearson_p=0.001,
            spearman_rho=0.85,
            spearman_p=0.002,
            n=50,
            significant_fdr=True,
        )
        assert cr.pearson_r == 0.9
        assert cr.pearson_p == 0.001
        assert cr.spearman_rho == 0.85
        assert cr.spearman_p == 0.002
        assert cr.n == 50
        assert cr.significant_fdr is True

    def test_default_significant_fdr_is_false(self) -> None:
        cr = CorrelationResult(
            pearson_r=0.5,
            pearson_p=0.1,
            spearman_rho=0.4,
            spearman_p=0.15,
            n=20,
        )
        assert cr.significant_fdr is False

    def test_frozen_raises_on_assignment(self) -> None:
        cr = CorrelationResult(
            pearson_r=0.5,
            pearson_p=0.1,
            spearman_rho=0.4,
            spearman_p=0.15,
            n=20,
        )
        with pytest.raises(AttributeError):
            cr.pearson_r = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ComparisonResult dataclass
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_fields_are_accessible(self) -> None:
        posthoc = pd.DataFrame({"a": [1], "b": [2]})
        cr = ComparisonResult(
            statistic=15.3,
            p_value=0.001,
            n_groups=3,
            epsilon_squared=0.12,
            posthoc=posthoc,
        )
        assert cr.statistic == 15.3
        assert cr.p_value == 0.001
        assert cr.n_groups == 3
        assert cr.epsilon_squared == 0.12
        assert len(cr.posthoc) == 1

    def test_default_posthoc_is_empty_dataframe(self) -> None:
        cr = ComparisonResult(statistic=1.0, p_value=0.5, n_groups=2)
        assert isinstance(cr.posthoc, pd.DataFrame)
        assert cr.posthoc.empty


# ---------------------------------------------------------------------------
# SpatialResult dataclass
# ---------------------------------------------------------------------------


class TestSpatialResult:
    def test_fields_are_accessible(self) -> None:
        labels = np.zeros((4, 4), dtype=int)
        pvals = np.ones((4, 4))
        sr = SpatialResult(
            morans_i=0.3,
            morans_p=0.01,
            expected_i=-0.05,
            lisa_labels=labels,
            lisa_p_values=pvals,
        )
        assert sr.morans_i == 0.3
        assert sr.morans_p == 0.01
        assert sr.expected_i == -0.05
        assert sr.lisa_labels.shape == (4, 4)
        assert sr.lisa_p_values.shape == (4, 4)


# ---------------------------------------------------------------------------
# RegressionResult dataclass
# ---------------------------------------------------------------------------


class TestRegressionResult:
    def test_fields_are_accessible(self) -> None:
        coef = pd.DataFrame({"variable": ["x"], "beta": [1.0]})
        vif = pd.DataFrame({"variable": ["x"], "vif": [1.1]})
        rr = RegressionResult(
            coefficients=coef,
            r_squared_adj=0.85,
            n=100,
            model_type="rlm",
            vif=vif,
        )
        assert rr.r_squared_adj == 0.85
        assert rr.model_type == "rlm"
        assert rr.n == 100
        assert not rr.coefficients.empty
        assert not rr.vif.empty

    def test_default_vif_is_empty_dataframe(self) -> None:
        rr = RegressionResult(
            coefficients=pd.DataFrame(),
            r_squared_adj=0.0,
            n=0,
            model_type="rlm",
        )
        assert isinstance(rr.vif, pd.DataFrame)
        assert rr.vif.empty


# ---------------------------------------------------------------------------
# correlation_analysis
# ---------------------------------------------------------------------------


class TestCorrelationAnalysis:
    def test_returns_correlation_result(self) -> None:
        df = _make_corr_df(n=50)
        result = correlation_analysis(df, "entropy", "metric")
        assert isinstance(result, CorrelationResult)

    def test_negative_correlation_detected(self) -> None:
        """Synthetic data has metric = 30 - 2*entropy + noise."""
        df = _make_corr_df(n=200)
        result = correlation_analysis(df, "entropy", "metric")
        assert result.pearson_r < -0.5
        assert result.spearman_rho < -0.5

    def test_n_matches_valid_rows(self) -> None:
        df = _make_corr_df(n=80)
        result = correlation_analysis(df, "entropy", "metric")
        assert result.n == 80

    def test_n_excludes_nans(self) -> None:
        df = _make_corr_df(n=80, inject_nans=True)
        result = correlation_analysis(df, "entropy", "metric")
        assert result.n == 78  # 2 rows dropped

    def test_p_values_are_small_for_strong_correlation(self) -> None:
        df = _make_corr_df(n=200)
        result = correlation_analysis(df, "entropy", "metric")
        assert result.pearson_p < 0.05
        assert result.spearman_p < 0.05

    def test_fewer_than_3_returns_nan(self) -> None:
        df = _make_corr_df(n=2)
        result = correlation_analysis(df, "entropy", "metric")
        assert math.isnan(result.pearson_r)
        assert math.isnan(result.pearson_p)
        assert math.isnan(result.spearman_rho)
        assert math.isnan(result.spearman_p)
        assert result.n == 2

    def test_exactly_3_observations_works(self) -> None:
        df = _make_corr_df(n=3)
        result = correlation_analysis(df, "entropy", "metric")
        assert not math.isnan(result.pearson_r)
        assert result.n == 3

    def test_zero_valid_after_nan_drop(self) -> None:
        df = pd.DataFrame({"entropy": [np.nan, np.nan], "metric": [1.0, 2.0]})
        result = correlation_analysis(df, "entropy", "metric")
        assert math.isnan(result.pearson_r)
        assert result.n == 0

    def test_perfect_positive_correlation(self) -> None:
        df = pd.DataFrame({
            "entropy": [1.0, 2.0, 3.0, 4.0],
            "metric": [10.0, 20.0, 30.0, 40.0],
        })
        result = correlation_analysis(df, "entropy", "metric")
        assert abs(result.pearson_r - 1.0) < 1e-10
        assert abs(result.spearman_rho - 1.0) < 1e-10

    def test_no_correlation_gives_low_r(self) -> None:
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            "entropy": rng.uniform(0, 10, 500),
            "metric": rng.uniform(0, 10, 500),
        })
        result = correlation_analysis(df, "entropy", "metric")
        assert abs(result.pearson_r) < 0.15
        assert abs(result.spearman_rho) < 0.15


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_returns_dataframe(self) -> None:
        df = _make_matrix_df()
        result = correlation_matrix(df, ["entropy_7"], ["psnr"])
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_expected_columns_present(self) -> None:
        df = _make_matrix_df()
        result = correlation_matrix(
            df,
            ["entropy_7", "entropy_15"],
            ["psnr", "ssim"],
        )
        expected = {
            "method",
            "entropy_col",
            "metric_col",
            "spearman_rho",
            "spearman_p",
            "pearson_r",
            "pearson_p",
            "n",
            "spearman_significant_fdr",
            "pearson_significant_fdr",
        }
        assert expected.issubset(set(result.columns))

    def test_row_count_matches_combinations(self) -> None:
        df = _make_matrix_df()
        entropy_cols = ["entropy_7", "entropy_15"]
        metric_cols = ["psnr", "ssim"]
        n_methods = df["method"].nunique()
        result = correlation_matrix(df, entropy_cols, metric_cols)
        expected_rows = n_methods * len(entropy_cols) * len(metric_cols)
        assert len(result) == expected_rows

    def test_methods_filter(self) -> None:
        df = _make_matrix_df()
        result = correlation_matrix(
            df,
            ["entropy_7"],
            ["psnr"],
            methods=["bilinear"],
        )
        assert all(result["method"] == "bilinear")
        assert len(result) == 1

    def test_fdr_correction_applied(self) -> None:
        """With strongly correlated data and few tests, FDR flags should
        match the raw significance at alpha=0.05."""
        df = _make_matrix_df(n_per_method=100)
        result = correlation_matrix(df, ["entropy_7"], ["psnr"])
        # entropy_7 is strongly correlated with psnr in synthetic data.
        for _, row in result.iterrows():
            if row["spearman_p"] < 0.01:
                # With very small p-values, FDR should still flag significant.
                assert row["spearman_significant_fdr"]

    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame(columns=["method", "entropy_7", "psnr"])
        result = correlation_matrix(df, ["entropy_7"], ["psnr"])
        assert result.empty

    def test_nan_p_values_handled(self) -> None:
        """When a method subgroup has < 3 rows, p-values are NaN. FDR
        correction should not crash."""
        rng = np.random.default_rng(99)
        # One method has enough data, the other has only 2 rows.
        rows = []
        for _ in range(50):
            rows.append({
                "method": "bilinear",
                "entropy_7": rng.uniform(1, 5),
                "psnr": rng.normal(25, 1),
            })
        for _ in range(2):
            rows.append({
                "method": "kriging",
                "entropy_7": rng.uniform(1, 5),
                "psnr": rng.normal(25, 1),
            })
        df = pd.DataFrame(rows)
        result = correlation_matrix(df, ["entropy_7"], ["psnr"])
        assert len(result) == 2
        # kriging row should have NaN p-values.
        kriging_row = result[result["method"] == "kriging"].iloc[0]
        assert math.isnan(kriging_row["spearman_p"])


# ---------------------------------------------------------------------------
# method_comparison
# ---------------------------------------------------------------------------


class TestMethodComparison:
    def test_returns_comparison_result(self) -> None:
        df = _make_method_df()
        result = method_comparison(df, metric_col="psnr")
        assert isinstance(result, ComparisonResult)

    def test_detects_significant_difference(self) -> None:
        """Methods have shifted means, so Kruskal-Wallis should detect it."""
        df = _make_method_df(n_per_method=100)
        result = method_comparison(df, metric_col="psnr")
        assert result.p_value < 0.05
        assert result.n_groups == 3

    def test_epsilon_squared_positive(self) -> None:
        df = _make_method_df(n_per_method=100)
        result = method_comparison(df, metric_col="psnr")
        assert result.epsilon_squared > 0.0

    def test_posthoc_has_expected_columns(self) -> None:
        df = _make_method_df()
        result = method_comparison(df, metric_col="psnr")
        expected = {
            "method_a",
            "method_b",
            "u_statistic",
            "p_value",
            "p_corrected",
            "significant",
            "cliffs_delta",
        }
        assert expected.issubset(set(result.posthoc.columns))

    def test_posthoc_row_count(self) -> None:
        """n_groups choose 2 pairwise comparisons."""
        df = _make_method_df(methods=["a", "b", "c", "d"])
        result = method_comparison(df, metric_col="psnr")
        # 4 choose 2 = 6
        assert len(result.posthoc) == 6

    def test_bonferroni_corrected_p_at_most_one(self) -> None:
        df = _make_method_df()
        result = method_comparison(df, metric_col="psnr")
        for _, row in result.posthoc.iterrows():
            assert row["p_corrected"] <= 1.0

    def test_cliffs_delta_in_range(self) -> None:
        df = _make_method_df()
        result = method_comparison(df, metric_col="psnr")
        for _, row in result.posthoc.iterrows():
            assert -1.0 <= row["cliffs_delta"] <= 1.0

    def test_fewer_than_2_groups_returns_nan(self) -> None:
        df = _make_method_df(methods=["bilinear"])
        result = method_comparison(df, metric_col="psnr")
        assert math.isnan(result.statistic)
        assert math.isnan(result.p_value)
        assert result.n_groups == 1
        assert result.posthoc.empty

    def test_zero_groups_returns_nan(self) -> None:
        df = pd.DataFrame(columns=["method", "psnr"])
        result = method_comparison(df, metric_col="psnr")
        assert math.isnan(result.statistic)
        assert result.n_groups == 0

    def test_exactly_2_groups(self) -> None:
        df = _make_method_df(methods=["a", "b"])
        result = method_comparison(df, metric_col="psnr")
        assert result.n_groups == 2
        assert len(result.posthoc) == 1

    def test_nan_metric_values_dropped(self) -> None:
        df = _make_method_df(n_per_method=30)
        df.loc[0, "psnr"] = np.nan
        df.loc[1, "psnr"] = np.nan
        result = method_comparison(df, metric_col="psnr")
        # Should still produce a valid result since plenty of non-NaN rows.
        assert result.n_groups == 3
        assert not math.isnan(result.statistic)

    def test_identical_groups_not_significant(self) -> None:
        """When all groups are drawn from the same distribution, the test
        should typically not reject H0 (p > 0.05). With seed-fixed data
        this is deterministic."""
        rng = np.random.default_rng(77)
        rows = []
        for m in ["a", "b", "c"]:
            for v in rng.normal(10.0, 1.0, 50):
                rows.append({"method": m, "psnr": v})
        df = pd.DataFrame(rows)
        result = method_comparison(df, metric_col="psnr")
        assert result.p_value > 0.05


# ---------------------------------------------------------------------------
# robust_regression
# ---------------------------------------------------------------------------


class TestRobustRegression:
    def test_returns_regression_result(self) -> None:
        df = _make_regression_df()
        result = robust_regression(df, metric_col="psnr")
        assert isinstance(result, RegressionResult)
        assert result.model_type == "rlm"

    def test_n_matches_valid_rows(self) -> None:
        df = _make_regression_df(n=200)
        result = robust_regression(df, metric_col="psnr")
        assert result.n == 200

    def test_coefficient_table_has_expected_columns(self) -> None:
        df = _make_regression_df()
        result = robust_regression(df, metric_col="psnr")
        expected = {
            "variable",
            "beta",
            "std_err",
            "z_value",
            "p_value",
            "ci_lo",
            "ci_hi",
        }
        assert expected.issubset(set(result.coefficients.columns))

    def test_entropy_coefficient_is_negative(self) -> None:
        """psnr = 30 - 2*entropy_7 + noise.

        Therefore, entropy_7 beta should be negative.
        """
        df = _make_regression_df(n=500)
        result = robust_regression(
            df,
            metric_col="psnr",
            entropy_cols=["entropy_7"],
        )
        entropy_row = result.coefficients[
            result.coefficients["variable"] == "entropy_7"
        ]
        assert not entropy_row.empty
        assert entropy_row.iloc[0]["beta"] < 0.0

    def test_r_squared_adj_positive(self) -> None:
        df = _make_regression_df()
        result = robust_regression(df, metric_col="psnr")
        assert result.r_squared_adj > 0.0
        assert result.r_squared_adj <= 1.0

    def test_vif_computed(self) -> None:
        df = _make_regression_df()
        result = robust_regression(
            df,
            metric_col="psnr",
            entropy_cols=["entropy_7", "entropy_15"],
        )
        assert not result.vif.empty
        assert "vif" in result.vif.columns
        assert "variable" in result.vif.columns

    def test_vif_values_near_one_for_independent_predictors(self) -> None:
        """entropy_7 and entropy_15 are independent uniform draws, so VIF
        should be close to 1."""
        df = _make_regression_df(n=500)
        result = robust_regression(
            df,
            metric_col="psnr",
            entropy_cols=["entropy_7", "entropy_15"],
        )
        for _, row in result.vif.iterrows():
            assert row["vif"] < 5.0

    def test_fewer_than_10_rows_returns_empty(self) -> None:
        df = _make_regression_df(n=5)
        result = robust_regression(df, metric_col="psnr")
        assert result.coefficients.empty
        assert math.isnan(result.r_squared_adj)
        assert result.n < 10

    def test_exactly_10_rows_works(self) -> None:
        df = _make_regression_df(n=10)
        result = robust_regression(df, metric_col="psnr")
        assert result.n == 10
        # With 10 rows the model may or may not fit depending on degrees
        # of freedom, but it should not return the empty fallback.
        assert not result.coefficients.empty

    def test_auto_detects_entropy_cols(self) -> None:
        """When entropy_cols is None, function should auto-detect columns
        starting with 'entropy_'."""
        df = _make_regression_df()
        result = robust_regression(df, metric_col="psnr", entropy_cols=None)
        variables = result.coefficients["variable"].tolist()
        assert "entropy_7" in variables
        assert "entropy_15" in variables

    def test_single_entropy_col(self) -> None:
        df = _make_regression_df()
        result = robust_regression(
            df,
            metric_col="psnr",
            entropy_cols=["entropy_7"],
        )
        assert not result.coefficients.empty
        vif_vars = (
            result.vif["variable"].tolist() if not result.vif.empty else []
        )
        assert "entropy_15" not in vif_vars

    def test_confidence_intervals_bracket_beta(self) -> None:
        df = _make_regression_df(n=500)
        result = robust_regression(df, metric_col="psnr")
        for _, row in result.coefficients.iterrows():
            assert row["ci_lo"] <= row["beta"] <= row["ci_hi"]


# ---------------------------------------------------------------------------
# spatial_autocorrelation
# ---------------------------------------------------------------------------


esda = pytest.importorskip("esda", reason="esda not installed")
libpysal = pytest.importorskip("libpysal", reason="libpysal not installed")


@pytest.mark.slow
class TestSpatialAutocorrelation:
    """Tests for spatial_autocorrelation using esda and libpysal.

    Requires esda and libpysal. The entire class is skipped when
    either library is absent. Marked slow because lattice weight
    construction is non-trivial even for small grids.
    """

    @staticmethod
    def _make_spatially_correlated_map(
        size: int = 8,
        seed: int = 42,
    ) -> np.ndarray:
        """Create an 8x8 error map with positive spatial autocorrelation.

        Uses a simple block pattern: top-left quadrant has high values,
        bottom-right has low values. This guarantees positive Moran's I.
        """
        rng = np.random.default_rng(seed)
        error_map = np.zeros((size, size), dtype=np.float64)
        half = size // 2
        error_map[:half, :half] = 0.8 + rng.uniform(0, 0.1, (half, half))
        error_map[half:, half:] = 0.1 + rng.uniform(
            0, 0.05, (size - half, size - half)
        )
        error_map[:half, half:] = 0.4 + rng.uniform(
            0, 0.05, (half, size - half)
        )
        error_map[half:, :half] = 0.4 + rng.uniform(
            0, 0.05, (size - half, half)
        )
        return error_map

    def test_returns_spatial_result(self) -> None:
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map)
        assert isinstance(result, SpatialResult)

    def test_morans_i_positive_for_clustered_map(self) -> None:
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map)
        assert result.morans_i > 0.0

    def test_expected_i_negative(self) -> None:
        """Expected Moran's I under spatial randomness is -1/(N-1), which
        is negative for any N > 1."""
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map)
        assert result.expected_i < 0.0

    def test_morans_p_small_for_clustered_map(self) -> None:
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map)
        assert result.morans_p < 0.10

    def test_lisa_labels_shape(self) -> None:
        error_map = self._make_spatially_correlated_map(size=8)
        result = spatial_autocorrelation(error_map)
        assert result.lisa_labels.shape == (8, 8)

    def test_lisa_p_values_shape(self) -> None:
        error_map = self._make_spatially_correlated_map(size=8)
        result = spatial_autocorrelation(error_map)
        assert result.lisa_p_values.shape == (8, 8)

    def test_lisa_p_values_in_valid_range(self) -> None:
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map)
        assert np.all(result.lisa_p_values >= 0.0)
        assert np.all(result.lisa_p_values <= 1.0)

    def test_with_mask(self) -> None:
        error_map = self._make_spatially_correlated_map()
        mask = np.ones_like(error_map, dtype=bool)
        mask[0, 0] = False
        mask[0, 1] = False
        result = spatial_autocorrelation(error_map, mask=mask)
        assert isinstance(result, SpatialResult)
        assert result.lisa_labels.shape == error_map.shape

    def test_no_mask_means_all_included(self) -> None:
        error_map = self._make_spatially_correlated_map()
        result = spatial_autocorrelation(error_map, mask=None)
        assert isinstance(result, SpatialResult)

    def test_random_map_has_low_morans_i(self) -> None:
        """A purely random map should have Moran's I close to the expected
        value under randomness."""
        rng = np.random.default_rng(99)
        error_map = rng.uniform(0, 1, (8, 8))
        result = spatial_autocorrelation(error_map)
        assert abs(result.morans_i - result.expected_i) < 0.5
