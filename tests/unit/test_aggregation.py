"""Unit tests for the aggregation module."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdi_pipeline.aggregation import (
    _bootstrap_ci,
    load_results,
    summary_by_entropy_bin,
    summary_by_gap_fraction,
    summary_by_method,
    summary_by_noise,
    summary_by_satellite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results_df(
    methods: list[str] | None = None,
    n_per_method: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic raw-results DataFrame with typical columns."""
    rng = np.random.default_rng(seed)
    if methods is None:
        methods = ["bilinear", "bicubic", "idw"]

    rows: list[dict] = []
    for method in methods:
        for _i in range(n_per_method):
            rows.append({
                "method": method,
                "satellite": rng.choice(["sentinel2", "landsat8"]),
                "noise_level": rng.choice(["inf", "40", "30", "20"]),
                "gap_fraction": rng.uniform(0.05, 0.6),
                "entropy_7": rng.uniform(0.0, 5.0),
                "psnr": rng.uniform(20.0, 45.0),
                "ssim": rng.uniform(0.6, 1.0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_load_from_csv_file(self, tmp_path: Path) -> None:
        # Direct path to a CSV file.
        df = _make_results_df(n_per_method=5)
        csv_path = tmp_path / "raw_results.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_results(csv_path)
        assert loaded.shape == df.shape
        assert list(loaded.columns) == list(df.columns)

    def test_load_from_directory(self, tmp_path: Path) -> None:
        # Pass a directory; function should append raw_results.csv.
        df = _make_results_df(n_per_method=5)
        csv_path = tmp_path / "raw_results.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_results(tmp_path)
        assert loaded.shape == df.shape

    def test_load_accepts_string_path(self, tmp_path: Path) -> None:
        # Should work with a plain str, not only pathlib.Path.
        df = _make_results_df(n_per_method=3)
        csv_path = tmp_path / "raw_results.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_results(str(tmp_path))
        assert loaded.shape == df.shape

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        # Missing file should raise.
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent.csv")

    def test_roundtrip_numeric_columns(self, tmp_path: Path) -> None:
        # Pure numeric columns survive the CSV write/read roundtrip.
        df = _make_results_df(n_per_method=5, seed=99)
        csv_path = tmp_path / "raw_results.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_results(csv_path)
        # Only check numeric columns since CSV may coerce string-ish
        # values like noise_level ("inf", "40") into floats.
        numeric_cols = ["gap_fraction", "entropy_7", "psnr", "ssim"]
        pd.testing.assert_frame_equal(
            loaded[numeric_cols], df[numeric_cols], check_dtype=False
        )


# ---------------------------------------------------------------------------
# _bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_tuple_of_two_floats(self) -> None:
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = _bootstrap_ci(vals, n_boot=500, seed=0)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_sample_mean(self) -> None:
        # The true mean of the sample should lie within the 95% CI.
        rng = np.random.default_rng(7)
        vals = rng.normal(loc=10.0, scale=1.0, size=200)
        lo, hi = _bootstrap_ci(vals, n_boot=5000, ci=0.95, seed=42)
        sample_mean = float(np.mean(vals))
        assert lo <= sample_mean <= hi

    def test_wider_ci_with_higher_confidence(self) -> None:
        vals = np.random.default_rng(1).normal(size=50)
        lo_90, hi_90 = _bootstrap_ci(vals, n_boot=2000, ci=0.90, seed=0)
        lo_99, hi_99 = _bootstrap_ci(vals, n_boot=2000, ci=0.99, seed=0)
        assert (hi_99 - lo_99) >= (hi_90 - lo_90)

    def test_single_value(self) -> None:
        # With a single value, CI collapses to that value.
        vals = np.array([42.0])
        lo, hi = _bootstrap_ci(vals, n_boot=100, seed=0)
        assert lo == pytest.approx(42.0)
        assert hi == pytest.approx(42.0)

    def test_empty_array_returns_nan(self) -> None:
        vals = np.array([], dtype=np.float64)
        lo, hi = _bootstrap_ci(vals)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_all_nan_values_returns_nan(self) -> None:
        vals = np.array([np.nan, np.nan, np.nan])
        lo, hi = _bootstrap_ci(vals)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_nan_values_are_stripped(self) -> None:
        # NaNs should be dropped before bootstrapping.
        clean = np.array([1.0, 2.0, 3.0])
        with_nan = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        lo_clean, hi_clean = _bootstrap_ci(clean, n_boot=500, seed=0)
        lo_nan, hi_nan = _bootstrap_ci(with_nan, n_boot=500, seed=0)
        assert lo_clean == pytest.approx(lo_nan)
        assert hi_clean == pytest.approx(hi_nan)

    def test_deterministic_with_same_seed(self) -> None:
        vals = np.random.default_rng(3).normal(size=100)
        result1 = _bootstrap_ci(vals, n_boot=1000, seed=42)
        result2 = _bootstrap_ci(vals, n_boot=1000, seed=42)
        assert result1 == result2

    def test_lo_leq_hi(self) -> None:
        vals = np.random.default_rng(5).uniform(0, 10, size=60)
        lo, hi = _bootstrap_ci(vals, n_boot=2000, seed=0)
        assert lo <= hi


# ---------------------------------------------------------------------------
# summary_by_method
# ---------------------------------------------------------------------------


class TestSummaryByMethod:
    def test_one_row_per_method(self) -> None:
        df = _make_results_df(methods=["A", "B", "C"], n_per_method=20)
        result = summary_by_method(df, metric="psnr")
        assert set(result["method"]) == {"A", "B", "C"}
        assert len(result) == 3

    def test_output_columns(self) -> None:
        df = _make_results_df(n_per_method=10)
        result = summary_by_method(df, metric="psnr")
        expected_cols = {
            "method",
            "n",
            "mean",
            "median",
            "std",
            "ci95_lo",
            "ci95_hi",
        }
        assert expected_cols == set(result.columns)

    def test_sorted_by_mean_descending(self) -> None:
        # Construct methods with known ordering.
        rng = np.random.default_rng(0)
        rows = []
        for method, base in [("low", 10.0), ("mid", 25.0), ("high", 40.0)]:
            for _ in range(30):
                rows.append({
                    "method": method,
                    "psnr": base + rng.normal(0, 0.5),
                })
        df = pd.DataFrame(rows)
        result = summary_by_method(df, metric="psnr")
        means = result["mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_n_counts_match(self) -> None:
        df = _make_results_df(methods=["X"], n_per_method=17)
        result = summary_by_method(df, metric="psnr")
        assert result.iloc[0]["n"] == 17

    def test_ci_bounds_contain_mean(self) -> None:
        df = _make_results_df(n_per_method=50)
        result = summary_by_method(df, metric="psnr")
        for _, row in result.iterrows():
            assert row["ci95_lo"] <= row["mean"] <= row["ci95_hi"]

    def test_single_method(self) -> None:
        df = _make_results_df(methods=["only_one"], n_per_method=10)
        result = summary_by_method(df, metric="psnr")
        assert len(result) == 1
        assert result.iloc[0]["method"] == "only_one"

    def test_different_metric_column(self) -> None:
        # Should work with any numeric metric column.
        df = _make_results_df(n_per_method=10)
        result = summary_by_method(df, metric="ssim")
        assert "mean" in result.columns
        # SSIM values are in [0.6, 1.0] by construction.
        for _, row in result.iterrows():
            assert 0.5 <= row["mean"] <= 1.05

    def test_nan_metric_values_excluded_from_count(self) -> None:
        # Rows with NaN in the metric column should be excluded.
        df = _make_results_df(methods=["A"], n_per_method=10, seed=5)
        df.loc[0:2, "psnr"] = np.nan  # inject 3 NaNs
        result = summary_by_method(df, metric="psnr")
        assert result.iloc[0]["n"] == 7

    def test_empty_dataframe_raises(self) -> None:
        # An empty input produces an empty rows list, and sort_values
        # on a column-less DataFrame raises KeyError.
        df = pd.DataFrame(columns=["method", "psnr"])
        with pytest.raises(KeyError):
            summary_by_method(df, metric="psnr")

    def test_mean_and_std_correctness(self) -> None:
        # Use known values to verify mean/std.
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        df = pd.DataFrame({"method": ["A"] * 5, "psnr": vals})
        result = summary_by_method(df, metric="psnr")
        row = result.iloc[0]
        assert row["mean"] == pytest.approx(np.mean(vals))
        assert row["median"] == pytest.approx(np.median(vals))
        assert row["std"] == pytest.approx(float(np.std(vals)))


# ---------------------------------------------------------------------------
# summary_by_entropy_bin
# ---------------------------------------------------------------------------


class TestSummaryByEntropyBin:
    def test_produces_three_bins(self) -> None:
        df = _make_results_df(n_per_method=60)
        result = summary_by_entropy_bin(df, metric="psnr")
        assert set(result["entropy_bin"].unique()) == {"low", "medium", "high"}

    def test_output_columns(self) -> None:
        df = _make_results_df(n_per_method=30)
        result = summary_by_entropy_bin(df, metric="psnr")
        expected = {"method", "entropy_bin", "n", "mean", "ci95_lo", "ci95_hi"}
        assert expected == set(result.columns)

    def test_sorted_by_method_then_bin(self) -> None:
        df = _make_results_df(methods=["A", "B"], n_per_method=60)
        result = summary_by_entropy_bin(df, metric="psnr")
        # Within each method, bins should be ordered low < medium < high.
        for method in result["method"].unique():
            subset = result[result["method"] == method]
            bins = subset["entropy_bin"].tolist()
            assert bins == ["low", "medium", "high"]

    def test_entropy_bin_is_categorical_ordered(self) -> None:
        df = _make_results_df(n_per_method=30)
        result = summary_by_entropy_bin(df, metric="psnr")
        assert result["entropy_bin"].dtype.name == "category"
        assert result["entropy_bin"].cat.ordered

    def test_custom_entropy_column(self) -> None:
        df = _make_results_df(n_per_method=30)
        df["entropy_custom"] = np.random.default_rng(1).uniform(0, 3, len(df))
        result = summary_by_entropy_bin(
            df, entropy_col="entropy_custom", metric="psnr"
        )
        assert not result.empty

    def test_empty_after_dropna(self) -> None:
        # All NaN in entropy or metric yields empty result.
        df = _make_results_df(n_per_method=5)
        df["entropy_7"] = np.nan
        result = summary_by_entropy_bin(df, metric="psnr")
        assert result.empty

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["method", "entropy_7", "psnr"])
        result = summary_by_entropy_bin(df, metric="psnr")
        assert result.empty

    def test_single_method(self) -> None:
        df = _make_results_df(methods=["solo"], n_per_method=60)
        result = summary_by_entropy_bin(df, metric="psnr")
        assert set(result["method"].unique()) == {"solo"}

    def test_total_n_matches_valid_rows(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=45)
        result = summary_by_entropy_bin(df, metric="psnr")
        total_n = result["n"].sum()
        valid_n = df.dropna(subset=["entropy_7", "psnr"]).shape[0]
        assert total_n == valid_n


# ---------------------------------------------------------------------------
# summary_by_gap_fraction
# ---------------------------------------------------------------------------


class TestSummaryByGapFraction:
    def test_produces_three_bins(self) -> None:
        df = _make_results_df(n_per_method=60)
        result = summary_by_gap_fraction(df, metric="psnr")
        assert set(result["gap_bin"].unique()) == {"small", "medium", "large"}

    def test_output_columns(self) -> None:
        df = _make_results_df(n_per_method=30)
        result = summary_by_gap_fraction(df, metric="psnr")
        expected = {"method", "gap_bin", "n", "mean", "ci95_lo", "ci95_hi"}
        assert expected == set(result.columns)

    def test_gap_bin_is_categorical_ordered(self) -> None:
        df = _make_results_df(n_per_method=30)
        result = summary_by_gap_fraction(df, metric="psnr")
        assert result["gap_bin"].dtype.name == "category"
        assert result["gap_bin"].cat.ordered

    def test_sorted_by_method_then_bin(self) -> None:
        df = _make_results_df(methods=["X", "Y"], n_per_method=60)
        result = summary_by_gap_fraction(df, metric="psnr")
        for method in result["method"].unique():
            subset = result[result["method"] == method]
            bins = subset["gap_bin"].tolist()
            assert bins == ["small", "medium", "large"]

    def test_empty_after_dropna(self) -> None:
        df = _make_results_df(n_per_method=5)
        df["gap_fraction"] = np.nan
        result = summary_by_gap_fraction(df, metric="psnr")
        assert result.empty

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["method", "gap_fraction", "psnr"])
        result = summary_by_gap_fraction(df, metric="psnr")
        assert result.empty

    def test_total_n_matches_valid_rows(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=45)
        result = summary_by_gap_fraction(df, metric="psnr")
        total_n = result["n"].sum()
        valid_n = df.dropna(subset=["gap_fraction", "psnr"]).shape[0]
        assert total_n == valid_n

    def test_ci_bounds_contain_mean(self) -> None:
        df = _make_results_df(n_per_method=60)
        result = summary_by_gap_fraction(df, metric="psnr")
        for _, row in result.iterrows():
            assert row["ci95_lo"] <= row["mean"] <= row["ci95_hi"]


# ---------------------------------------------------------------------------
# summary_by_satellite
# ---------------------------------------------------------------------------


class TestSummaryBySatellite:
    def test_output_columns(self) -> None:
        df = _make_results_df(n_per_method=20)
        result = summary_by_satellite(df, metric="psnr")
        expected = {
            "method",
            "satellite",
            "n",
            "mean",
            "median",
            "std",
            "ci95_lo",
            "ci95_hi",
        }
        assert expected == set(result.columns)

    def test_groups_by_method_and_satellite(self) -> None:
        df = _make_results_df(methods=["A", "B"], n_per_method=40)
        result = summary_by_satellite(df, metric="psnr")
        # Each (method, satellite) pair should appear at most once.
        pairs = list(zip(result["method"], result["satellite"]))
        assert len(pairs) == len(set(pairs))

    def test_sorted_by_method_then_satellite(self) -> None:
        df = _make_results_df(methods=["B", "A"], n_per_method=30)
        result = summary_by_satellite(df, metric="psnr")
        keys = list(zip(result["method"], result["satellite"]))
        assert keys == sorted(keys)

    def test_nan_metric_excluded(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=10, seed=3)
        total_valid = df["psnr"].notna().sum()
        df.loc[0:3, "psnr"] = np.nan
        result = summary_by_satellite(df, metric="psnr")
        assert result["n"].sum() == total_valid - 4

    def test_empty_dataframe_raises(self) -> None:
        # Empty input leads to sort_values on a column-less DataFrame.
        df = pd.DataFrame(columns=["method", "satellite", "psnr"])
        with pytest.raises(KeyError):
            summary_by_satellite(df, metric="psnr")

    def test_single_satellite(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=10)
        df["satellite"] = "sentinel2"
        result = summary_by_satellite(df, metric="psnr")
        assert len(result) == 1
        assert result.iloc[0]["satellite"] == "sentinel2"

    def test_ci_bounds_contain_mean(self) -> None:
        df = _make_results_df(n_per_method=50)
        result = summary_by_satellite(df, metric="psnr")
        for _, row in result.iterrows():
            assert row["ci95_lo"] <= row["mean"] <= row["ci95_hi"]


# ---------------------------------------------------------------------------
# summary_by_noise
# ---------------------------------------------------------------------------


class TestSummaryByNoise:
    def test_output_columns(self) -> None:
        df = _make_results_df(n_per_method=20)
        result = summary_by_noise(df, metric="psnr")
        expected = {"method", "noise_level", "n", "mean", "ci95_lo", "ci95_hi"}
        assert expected == set(result.columns)

    def test_groups_by_method_and_noise_level(self) -> None:
        df = _make_results_df(methods=["A", "B"], n_per_method=40)
        result = summary_by_noise(df, metric="psnr")
        pairs = list(zip(result["method"], result["noise_level"]))
        assert len(pairs) == len(set(pairs))

    def test_sorted_by_method_then_noise(self) -> None:
        df = _make_results_df(methods=["B", "A"], n_per_method=30)
        result = summary_by_noise(df, metric="psnr")
        keys = list(zip(result["method"], result["noise_level"]))
        assert keys == sorted(keys)

    def test_nan_metric_excluded(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=20, seed=11)
        valid_before = df["psnr"].notna().sum()
        df.loc[0:4, "psnr"] = np.nan
        result = summary_by_noise(df, metric="psnr")
        assert result["n"].sum() == valid_before - 5

    def test_empty_dataframe_raises(self) -> None:
        # Empty input leads to sort_values on a column-less DataFrame.
        df = pd.DataFrame(columns=["method", "noise_level", "psnr"])
        with pytest.raises(KeyError):
            summary_by_noise(df, metric="psnr")

    def test_single_noise_level(self) -> None:
        df = _make_results_df(methods=["A"], n_per_method=10)
        df["noise_level"] = "inf"
        result = summary_by_noise(df, metric="psnr")
        assert len(result) == 1
        assert result.iloc[0]["noise_level"] == "inf"

    def test_ci_bounds_contain_mean(self) -> None:
        df = _make_results_df(n_per_method=50)
        result = summary_by_noise(df, metric="psnr")
        for _, row in result.iterrows():
            assert row["ci95_lo"] <= row["mean"] <= row["ci95_hi"]

    def test_mean_correctness_single_group(self) -> None:
        # Single method, single noise level, known values.
        vals = [10.0, 20.0, 30.0]
        df = pd.DataFrame({
            "method": ["A"] * 3,
            "noise_level": ["inf"] * 3,
            "psnr": vals,
        })
        result = summary_by_noise(df, metric="psnr")
        assert result.iloc[0]["mean"] == pytest.approx(20.0)
