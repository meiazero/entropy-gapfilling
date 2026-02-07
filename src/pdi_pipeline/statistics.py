"""Statistical analysis for the gap-filling experiment.

Provides correlation analysis (Pearson, Spearman with FDR correction),
non-parametric method comparison (Kruskal-Wallis + Dunn post-hoc), and
spatial autocorrelation (Moran's I global + LISA local).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class CorrelationResult:
    """Result of a correlation analysis."""

    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n: int
    significant_fdr: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a multi-group comparison."""

    statistic: float
    p_value: float
    n_groups: int
    posthoc: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class SpatialResult:
    """Result of spatial autocorrelation analysis."""

    morans_i: float
    morans_p: float
    expected_i: float
    lisa_labels: np.ndarray  # (H, W) with cluster labels
    lisa_p_values: np.ndarray  # (H, W) with p-values


def correlation_analysis(
    df: pd.DataFrame,
    entropy_col: str,
    metric_col: str,
) -> CorrelationResult:
    """Compute Pearson and Spearman correlations with p-values.

    Args:
        df: DataFrame with at least entropy_col and metric_col.
        entropy_col: Column name for entropy values.
        metric_col: Column name for metric values.

    Returns:
        CorrelationResult with r, rho, p-values, and sample size.
    """
    valid = df[[entropy_col, metric_col]].dropna()
    if len(valid) < 3:
        return CorrelationResult(
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            n=len(valid),
        )

    x = valid[entropy_col].values
    y = valid[metric_col].values

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    return CorrelationResult(
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        n=len(valid),
    )


def correlation_matrix(
    df: pd.DataFrame,
    entropy_cols: list[str],
    metric_cols: list[str],
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Compute correlation matrix across methods, entropy windows, metrics.

    Applies FDR correction (Benjamini-Hochberg) across all tests.

    Args:
        df: Raw results DataFrame.
        entropy_cols: List of entropy column names.
        metric_cols: List of metric column names.
        methods: If given, only include these methods.

    Returns:
        DataFrame with method, entropy_col, metric_col, spearman_rho,
        p_value, significant_fdr columns.
    """
    if methods is None:
        methods = sorted(df["method"].unique())

    rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        for ecol in entropy_cols:
            for mcol in metric_cols:
                result = correlation_analysis(mdf, ecol, mcol)
                rows.append({
                    "method": method,
                    "entropy_col": ecol,
                    "metric_col": mcol,
                    "spearman_rho": result.spearman_rho,
                    "pearson_r": result.pearson_r,
                    "p_value": result.spearman_p,
                    "n": result.n,
                })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # FDR correction
    p_vals = result_df["p_value"].values
    valid_mask = ~np.isnan(p_vals)

    sig = np.full(len(p_vals), False)
    if np.any(valid_mask):
        _, corrected, _, _ = multipletests(p_vals[valid_mask], method="fdr_bh")
        sig[valid_mask] = corrected

    result_df["significant_fdr"] = sig
    return result_df


def method_comparison(
    df: pd.DataFrame,
    metric_col: str = "psnr",
) -> ComparisonResult:
    """Non-parametric comparison of methods using Kruskal-Wallis + Dunn.

    With 77k patches, normality assumptions will be violated, so we use
    Kruskal-Wallis (non-parametric) instead of ANOVA.

    Args:
        df: Raw results DataFrame.
        metric_col: Metric column for comparison.

    Returns:
        ComparisonResult with H statistic, p-value, and Dunn post-hoc
        pairwise comparison table.
    """
    groups = []
    method_names = []
    for method, group in df.groupby("method"):
        vals = group[metric_col].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            method_names.append(method)

    if len(groups) < 2:
        return ComparisonResult(
            statistic=float("nan"),
            p_value=float("nan"),
            n_groups=len(groups),
        )

    h_stat, h_p = stats.kruskal(*groups)

    # Dunn post-hoc with Bonferroni correction
    posthoc_rows = []
    n_comparisons = len(method_names) * (len(method_names) - 1) // 2

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            u_stat, u_p = stats.mannwhitneyu(
                groups[i], groups[j], alternative="two-sided"
            )
            corrected_p = min(u_p * n_comparisons, 1.0)
            posthoc_rows.append({
                "method_a": method_names[i],
                "method_b": method_names[j],
                "u_statistic": float(u_stat),
                "p_value": float(u_p),
                "p_corrected": corrected_p,
                "significant": corrected_p < 0.05,
            })

    return ComparisonResult(
        statistic=float(h_stat),
        p_value=float(h_p),
        n_groups=len(groups),
        posthoc=pd.DataFrame(posthoc_rows),
    )


def spatial_autocorrelation(
    error_map: np.ndarray,
    mask: np.ndarray | None = None,
) -> SpatialResult:
    """Compute Moran's I (global) and LISA (local) spatial autocorrelation.

    Uses Queen contiguity weights on the raster grid.

    Args:
        error_map: 2D array of reconstruction errors (e.g. squared error).
        mask: Optional 2D binary mask (1=gap). If given, only gap pixels
            are included in the analysis.

    Returns:
        SpatialResult with global Moran's I, p-value, and LISA labels/p-values.
    """
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import lat2W

    h, w = error_map.shape[:2]
    error_flat = error_map.ravel().astype(np.float64)

    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
    else:
        mask_flat = np.ones(len(error_flat), dtype=bool)

    # Build lattice weights
    weights = lat2W(h, w, rook=False)

    # Replace non-gap pixels with mean to avoid distortion
    mean_val = float(np.mean(error_flat[mask_flat]))
    filled = error_flat.copy()
    filled[~mask_flat] = mean_val

    # Global Moran's I
    mi = Moran(filled, weights)

    # Local indicators (LISA)
    lisa = Moran_Local(filled, weights)

    lisa_labels = np.array(lisa.q).reshape(h, w)
    lisa_pvals = np.array(lisa.p_sim).reshape(h, w)

    return SpatialResult(
        morans_i=float(mi.I),
        morans_p=float(mi.p_sim),
        expected_i=float(mi.EI),
        lisa_labels=lisa_labels,
        lisa_p_values=lisa_pvals,
    )
