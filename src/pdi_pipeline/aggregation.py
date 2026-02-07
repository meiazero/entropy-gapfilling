"""Result aggregation for the gap-filling experiment.

Loads raw Parquet results and produces summary DataFrames stratified
by method, entropy bin, gap fraction bin, etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    """Load raw experiment results from Parquet.

    Args:
        path: Path to the raw_results.parquet file, or the directory
            containing it.

    Returns:
        DataFrame with all experiment rows.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "raw_results.parquet"
    return pd.read_parquet(path)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = np.array([
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


def summary_by_method(
    df: pd.DataFrame,
    metric: str = "psnr",
) -> pd.DataFrame:
    """Per-method summary with mean, median, std, and 95% CI.

    Args:
        df: Raw results DataFrame.
        metric: Metric column to summarize.

    Returns:
        DataFrame indexed by method with aggregation columns.
    """
    rows = []
    for method, group in df.groupby("method"):
        vals = group[metric].dropna().values
        ci_lo, ci_hi = _bootstrap_ci(vals)
        rows.append({
            "method": method,
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "median": float(np.median(vals)) if len(vals) > 0 else float("nan"),
            "std": float(np.std(vals)) if len(vals) > 0 else float("nan"),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def summary_by_entropy_bin(
    df: pd.DataFrame,
    entropy_col: str = "entropy_7",
    metric: str = "psnr",
) -> pd.DataFrame:
    """Stratify results by entropy terciles (low/medium/high).

    Tercile thresholds are computed from the data distribution, not
    arbitrary cutoffs.

    Args:
        df: Raw results DataFrame.
        entropy_col: Entropy column name.
        metric: Metric column to summarize.

    Returns:
        DataFrame with method x entropy_bin summary.
    """
    valid = df.dropna(subset=[entropy_col, metric])
    if valid.empty:
        return pd.DataFrame()

    t1 = float(valid[entropy_col].quantile(1 / 3))
    t2 = float(valid[entropy_col].quantile(2 / 3))

    def _bin(v: float) -> str:
        if v <= t1:
            return "low"
        if v <= t2:
            return "medium"
        return "high"

    valid = valid.copy()
    valid["entropy_bin"] = valid[entropy_col].apply(_bin)

    rows = []
    for (method, ebin), group in valid.groupby(["method", "entropy_bin"]):
        vals = group[metric].values
        ci_lo, ci_hi = _bootstrap_ci(vals)
        rows.append({
            "method": method,
            "entropy_bin": ebin,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })

    result = pd.DataFrame(rows)
    bin_order = pd.CategoricalDtype(
        categories=["low", "medium", "high"], ordered=True
    )
    result["entropy_bin"] = result["entropy_bin"].astype(bin_order)
    return result.sort_values(["method", "entropy_bin"])


def summary_by_gap_fraction(
    df: pd.DataFrame,
    metric: str = "psnr",
) -> pd.DataFrame:
    """Stratify results by gap fraction size (small/medium/large).

    Tercile thresholds computed from the data distribution.

    Args:
        df: Raw results DataFrame.
        metric: Metric column to summarize.

    Returns:
        DataFrame with method x gap_bin summary.
    """
    valid = df.dropna(subset=["gap_fraction", metric])
    if valid.empty:
        return pd.DataFrame()

    t1 = float(valid["gap_fraction"].quantile(1 / 3))
    t2 = float(valid["gap_fraction"].quantile(2 / 3))

    def _bin(v: float) -> str:
        if v <= t1:
            return "small"
        if v <= t2:
            return "medium"
        return "large"

    valid = valid.copy()
    valid["gap_bin"] = valid["gap_fraction"].apply(_bin)

    rows = []
    for (method, gbin), group in valid.groupby(["method", "gap_bin"]):
        vals = group[metric].values
        ci_lo, ci_hi = _bootstrap_ci(vals)
        rows.append({
            "method": method,
            "gap_bin": gbin,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })

    result = pd.DataFrame(rows)
    bin_order = pd.CategoricalDtype(
        categories=["small", "medium", "large"], ordered=True
    )
    result["gap_bin"] = result["gap_bin"].astype(bin_order)
    return result.sort_values(["method", "gap_bin"])


def summary_by_noise(
    df: pd.DataFrame,
    metric: str = "psnr",
) -> pd.DataFrame:
    """Per-method x noise level summary.

    Args:
        df: Raw results DataFrame.
        metric: Metric column to summarize.

    Returns:
        DataFrame with method x noise_level summary.
    """
    rows = []
    for (method, noise), group in df.groupby(["method", "noise_level"]):
        vals = group[metric].dropna().values
        ci_lo, ci_hi = _bootstrap_ci(vals)
        rows.append({
            "method": method,
            "noise_level": noise,
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })
    return pd.DataFrame(rows).sort_values(["method", "noise_level"])
