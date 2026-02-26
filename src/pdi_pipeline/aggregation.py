"""Result aggregation for the gap-filling experiment.

Loads raw Parquet results and produces summary DataFrames stratified
by method, entropy bin, gap fraction bin, etc.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

_METRIC_COLS: frozenset[str] = frozenset({
    "psnr",
    "ssim",
    "rmse",
    "sam",
    "ergas",
    "gap_fraction",
    "entropy_7",
    "entropy_15",
    "entropy_31",
})


def _read_mixed_schema_csv(path: Path) -> pd.DataFrame:
    """Read a CSV where some rows have more fields than the header.

    Handles schema evolution where extra entropy columns were added after
    some rows were already written. The extra columns are identified from
    _METRIC_COLS and inserted after the last entropy column present in the
    existing header.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the full reconstructed schema.

    Raises:
        ValueError: When the number of extra columns cannot be reconciled
            with the candidates found in _METRIC_COLS.
    """
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        return pd.DataFrame(columns=header)

    max_fields = max(len(row) for row in rows)
    if max_fields == len(header):
        records = [dict(zip(header, row, strict=False)) for row in rows]
        return pd.DataFrame(records)

    n_extra = max_fields - len(header)
    missing_entropy = sorted(
        col
        for col in _METRIC_COLS
        if col.startswith("entropy_") and col not in header
    )
    if len(missing_entropy) != n_extra:
        n_missing = len(missing_entropy)
        raise ValueError(  # noqa: TRY003
            f"{path}: found rows with {n_extra} extra field(s) but identified "
            f"{n_missing} missing entropy candidate(s):"
            f"{missing_entropy}. "
            "Cannot auto-repair mixed-schema CSV."
        )

    entropy_in_header = [c for c in header if c.startswith("entropy_")]
    insert_after = (
        max(header.index(c) for c in entropy_in_header)
        if entropy_in_header
        else -1
    )
    full_header = (
        header[: insert_after + 1]
        + missing_entropy
        + header[insert_after + 1 :]
    )

    records: list[dict] = []
    for row in rows:
        if len(row) == len(full_header):
            records.append(dict(zip(full_header, row, strict=False)))
        elif len(row) == len(header):
            padded = (
                row[: insert_after + 1]
                + [""] * n_extra
                + row[insert_after + 1 :]
            )
            records.append(dict(zip(full_header, padded, strict=False)))
    return pd.DataFrame(records)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load raw experiment results from CSV.

    Args:
        path: Path to the raw_results.csv file, or the directory
            containing it.

    Returns:
        DataFrame with all experiment rows.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "raw_results.csv"
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = _read_mixed_schema_csv(path)
    for col in (
        "method",
        "method_category",
        "satellite",
        "noise_level",
        "status",
    ):
        if col in df.columns:
            df[col] = df[col].astype("category")
    float32_cols = [c for c in df.columns if c in _METRIC_COLS]
    if float32_cols:
        df[float32_cols] = df[float32_cols].astype(np.float32)
    return df


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean (vectorized)."""
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _summary_with_ci(
    df: pd.DataFrame,
    groupby_cols: list[str],
    metric: str,
    *,
    include_spread: bool = False,
    sort_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-group mean with bootstrap 95% CI.

    Args:
        df: DataFrame to aggregate.
        groupby_cols: Columns to group by.
        metric: Metric column to summarize.
        include_spread: If True, also compute median and std.
        sort_cols: Columns to sort result by. Defaults to
            sorting by ``groupby_cols``.

    Returns:
        DataFrame with group keys, n, mean, ci95_lo, ci95_hi
        (and optionally median, std).
    """
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(groupby_cols, observed=True):
        vals = group[metric].dropna().to_numpy()
        ci_lo, ci_hi = _bootstrap_ci(vals)
        n = len(vals)
        has_vals = n > 0

        row: dict[str, object] = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(groupby_cols, keys, strict=True):
            row[col] = val

        row["n"] = n
        row["mean"] = float(np.mean(vals)) if has_vals else float("nan")
        if include_spread:
            row["median"] = float(np.median(vals)) if has_vals else float("nan")
            row["std"] = float(np.std(vals)) if has_vals else float("nan")
        row["ci95_lo"] = ci_lo
        row["ci95_hi"] = ci_hi
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_by = sort_cols if sort_cols is not None else groupby_cols
    if not result.empty:
        result = result.sort_values(sort_by)
    return result


def _tercile_bin(
    df: pd.DataFrame,
    source_col: str,
    bin_col: str,
    labels: list[str],
) -> pd.DataFrame:
    """Add a tercile-based bin column to a copy of *df*.

    Args:
        df: Input DataFrame (not modified).
        source_col: Numeric column to bin.
        bin_col: Name of the new categorical column.
        labels: Three labels for the bins.

    Returns:
        Copy of *df* with the new ``bin_col`` column.
    """
    t1 = float(df[source_col].quantile(1 / 3))
    t2 = float(df[source_col].quantile(2 / 3))
    out = df.copy()
    out[bin_col] = pd.cut(
        out[source_col],
        bins=[-np.inf, t1, t2, np.inf],
        labels=labels,
        right=True,
    )
    return out


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
    return _summary_with_ci(
        df,
        ["method"],
        metric,
        include_spread=True,
        sort_cols=["mean"],
    ).sort_values("mean", ascending=False)


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

    valid = _tercile_bin(
        valid, entropy_col, "entropy_bin", ["low", "medium", "high"]
    )

    result = _summary_with_ci(valid, ["method", "entropy_bin"], metric)
    if not result.empty:
        bin_order = pd.CategoricalDtype(
            categories=["low", "medium", "high"], ordered=True
        )
        result["entropy_bin"] = result["entropy_bin"].astype(bin_order)
    return result


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

    valid = _tercile_bin(
        valid, "gap_fraction", "gap_bin", ["small", "medium", "large"]
    )

    result = _summary_with_ci(valid, ["method", "gap_bin"], metric)
    if not result.empty:
        bin_order = pd.CategoricalDtype(
            categories=["small", "medium", "large"], ordered=True
        )
        result["gap_bin"] = result["gap_bin"].astype(bin_order)
    return result


def summary_by_satellite(
    df: pd.DataFrame,
    metric: str = "psnr",
) -> pd.DataFrame:
    """Per-method x satellite summary with mean and 95% CI.

    Args:
        df: Raw results DataFrame.
        metric: Metric column to summarize.

    Returns:
        DataFrame with method x satellite summary.
    """
    return _summary_with_ci(
        df,
        ["method", "satellite"],
        metric,
        include_spread=True,
    )


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
    return _summary_with_ci(df, ["method", "noise_level"], metric)
