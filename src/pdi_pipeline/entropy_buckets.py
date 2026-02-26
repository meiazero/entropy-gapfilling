"""Entropy bucket utilities for dataset filtering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VALID_BUCKETS = {"low", "medium", "high"}


@dataclass(frozen=True)
class EntropyCutoffs:
    """Quantile cutoffs for entropy buckets."""

    low: float
    high: float


def _validate_buckets(buckets: Iterable[str]) -> list[str]:
    unique = sorted({b.strip().lower() for b in buckets if b is not None})
    invalid = [b for b in unique if b not in VALID_BUCKETS]
    if invalid:
        msg = (
            "Invalid entropy buckets: "
            f"{invalid}. Valid: {sorted(VALID_BUCKETS)}"
        )
        raise ValueError(msg)
    return unique


def compute_entropy_cutoffs(
    manifest_path: str | Path,
    *,
    window: int,
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    split: str = "train",
    satellite: str | None = None,
) -> EntropyCutoffs:
    """Compute global entropy cutoffs from the train split.

    Args:
        manifest_path: Path to manifest CSV.
        window: Entropy window size (e.g. 7, 15, 31).
        quantiles: Low/high quantiles for bucket split.
        split: Dataset split to compute cutoffs from.
        satellite: Optional satellite filter.

    Returns:
        EntropyCutoffs with low/high thresholds.
    """
    manifest_path = Path(manifest_path)
    entropy_col = f"mean_entropy_{window}"

    usecols = ["split", "satellite", entropy_col]
    df = pd.read_csv(manifest_path, usecols=usecols)
    df = df[df["split"] == split]
    if satellite is not None:
        df = df[df["satellite"] == satellite]

    series = df[entropy_col].dropna()
    if series.empty:
        msg = (
            f"No entropy values for {entropy_col} in split='{split}'. "
            "Run precompute_entropy first."
        )
        raise ValueError(msg)

    low_q, high_q = quantiles
    low = float(series.quantile(low_q))
    high = float(series.quantile(high_q))
    return EntropyCutoffs(low=low, high=high)


def bucket_mask(
    series: pd.Series,
    cutoffs: EntropyCutoffs,
    buckets: Iterable[str],
) -> pd.Series:
    """Return a boolean mask for the requested entropy buckets."""
    normalized = _validate_buckets(buckets)
    mask = pd.Series(False, index=series.index)
    if "low" in normalized:
        mask |= series <= cutoffs.low
    if "medium" in normalized:
        mask |= (series > cutoffs.low) & (series <= cutoffs.high)
    if "high" in normalized:
        mask |= series > cutoffs.high
    return mask


def filter_by_entropy_buckets(
    df: pd.DataFrame,
    *,
    window: int,
    cutoffs: EntropyCutoffs,
    buckets: Iterable[str],
) -> pd.DataFrame:
    """Filter a DataFrame by entropy buckets.

    Args:
        df: DataFrame containing mean_entropy_{window}.
        window: Entropy window size.
        cutoffs: Cutoffs computed from train split.
        buckets: Buckets to include (low/medium/high).

    Returns:
        Filtered DataFrame.
    """
    entropy_col = f"mean_entropy_{window}"
    if entropy_col not in df.columns:
        msg = f"Missing entropy column in manifest: {entropy_col}"
        raise ValueError(msg)

    series = df[entropy_col]
    mask = bucket_mask(series, cutoffs, buckets)
    return df[mask & series.notna()]
