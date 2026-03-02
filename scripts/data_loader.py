"""Shared data loading and selection utilities for figure/table generation.

Loads raw CSVs from paper_assets/ and provides helpers for:
- Top-N method/model selection by metric
- Entropy tercile binning
- DL training history loading
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLASSIC_CSV = (
    _PROJECT_ROOT
    / "paper_assets"
    / "classic"
    / "full_results"
    / "raw_results.csv"
)
_DL_EVAL_DIR = _PROJECT_ROOT / "paper_assets" / "dl" / "eval"
_DL_HISTORY_DIR = _PROJECT_ROOT / "paper_assets" / "dl" / "history"

# Noise level display mapping
NOISE_LABELS: dict[str, str] = {
    "inf": "Sem ruído",
    "40": "40 dB",
    "30": "30 dB",
    "20": "20 dB",
}
NOISE_ORDER: list[str] = ["inf", "40", "30", "20"]

# Entropy window sizes available
ENTROPY_WINDOWS: list[int] = [7, 15, 31]

# DL models available
DL_MODELS: tuple[str, ...] = ("ae", "vae", "gan", "unet", "vit")

# DL entropy scenarios
DL_SCENARIOS: tuple[str, ...] = (
    "entropy_all",
    "entropy_high",
    "entropy_medium_high",
)

# Method category display names
CATEGORY_LABELS: dict[str, str] = {
    "spatial": "Espacial",
    "kernel": "Kernel",
    "geostatistical": "Geoestatístico",
    "transform": "Transformada",
    "compressive": "Compressivo",
    "patch_based": "Baseado em Recortes",
}


def load_classic() -> pd.DataFrame:
    """Load classical method results from paper_assets."""
    if not _CLASSIC_CSV.exists():
        log.warning("Classical results not found: %s", _CLASSIC_CSV)
        return pd.DataFrame()
    df = pd.read_csv(_CLASSIC_CSV)
    df["noise_level"] = df["noise_level"].astype(str).replace({"inf": "inf"})
    # Normalize noise_level to string
    df["noise_level"] = df["noise_level"].apply(
        lambda x: "inf" if x in ("inf", "nan") else str(int(float(x)))
    )
    df["type"] = "Clássico"
    log.info("Loaded %d classical rows", len(df))
    return df


def load_dl() -> pd.DataFrame:
    """Load all DL evaluation results from paper_assets."""
    pattern = str(_DL_EVAL_DIR / "**" / "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        log.warning("No DL eval CSVs found in %s", _DL_EVAL_DIR)
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            dfs.append(d)
        except Exception:
            log.warning("Failed to read %s", f)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Normalize column names: model -> method
    if "model" in df.columns and "method" not in df.columns:
        df["method"] = df["model"]
    # Strip _inpainting suffix for cleaner display
    df["method"] = df["method"].str.replace("_inpainting", "", regex=False)
    df["noise_level"] = df["noise_level"].apply(
        lambda x: "inf" if str(x) in ("inf", "nan") else str(int(float(x)))
    )
    df["type"] = "DL"
    if "method_category" not in df.columns:
        df["method_category"] = "deep_learning"
    log.info("Loaded %d DL eval rows", len(df))
    return df


def load_combined() -> pd.DataFrame:
    """Load and concatenate classic + DL data."""
    classic = load_classic()
    dl = load_dl()
    parts = [p for p in [classic, dl] if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_dl_history(scenario: str, model: str) -> dict | None:
    """Load a single DL training history JSON."""
    path = _DL_HISTORY_DIR / f"{scenario}_{model}_history.json"
    if not path.exists():
        log.debug("History not found: %s", path)
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_all_dl_histories() -> dict[str, dict[str, dict]]:
    """Load all DL histories organized as {scenario: {model: history}}.

    Returns:
        Nested dict[scenario][model] = history_dict with keys
        'model_name', 'epochs', etc.
    """
    result: dict[str, dict[str, dict]] = {}
    for scenario in DL_SCENARIOS:
        result[scenario] = {}
        for model in DL_MODELS:
            hist = load_dl_history(scenario, model)
            if hist is not None:
                result[scenario][model] = hist
    total = sum(len(v) for v in result.values())
    log.info("Loaded %d DL training histories", total)
    return result


def select_top_n(
    df: pd.DataFrame,
    *,
    group_col: str = "method",
    metric: str = "psnr",
    n: int = 3,
    ascending: bool = False,
    noise_filter: str | None = "inf",
) -> list[str]:
    """Select top-N methods/models by mean metric value.

    Args:
        df: DataFrame with group_col and metric columns.
        group_col: Column to group by.
        metric: Metric to rank by.
        n: Number of top items to return.
        ascending: If True, lower is better (e.g. RMSE).
        noise_filter: Filter to this noise level before ranking.
            Use None to rank across all noise levels.

    Returns:
        List of top-N group names (methods/models).
    """
    subset = df.copy()
    if noise_filter is not None:
        subset = subset[subset["noise_level"] == noise_filter]
    if subset.empty or metric not in subset.columns:
        return []
    ranking = (
        subset
        .groupby(group_col, observed=True)[metric]
        .mean()
        .sort_values(ascending=ascending)
    )
    return ranking.head(n).index.tolist()


def entropy_terciles(
    df: pd.DataFrame,
    entropy_col: str = "entropy_31",
) -> pd.DataFrame:
    """Add entropy_bin column based on tercile thresholds.

    Returns a copy with a new 'entropy_bin' column with values
    'baixa', 'média', 'alta'.
    """
    df = df.copy()
    if entropy_col not in df.columns:
        log.warning("Column %s not found, skipping terciles", entropy_col)
        df["entropy_bin"] = "N/A"
        return df
    valid = df[entropy_col].dropna()
    if valid.empty:
        df["entropy_bin"] = "N/A"
        return df
    t1 = float(valid.quantile(1 / 3))
    t2 = float(valid.quantile(2 / 3))
    df["entropy_bin"] = pd.cut(
        df[entropy_col],
        bins=[-np.inf, t1, t2, np.inf],
        labels=["baixa", "média", "alta"],
        right=True,
    )
    return df


def noise_label(noise: str) -> str:
    """Human-readable noise level label."""
    return NOISE_LABELS.get(noise, f"{noise} dB")
