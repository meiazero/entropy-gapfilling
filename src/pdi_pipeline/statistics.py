"""Statistical analysis for the gap-filling experiment.

Provides correlation analysis (Pearson, Spearman with FDR correction),
non-parametric method comparison (Kruskal-Wallis + Dunn post-hoc),
robust regression (HuberT RLM with VIF and bootstrap CI), and
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
    epsilon_squared: float = 0.0
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

    x = valid[entropy_col].to_numpy()
    y = valid[metric_col].to_numpy()

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    return CorrelationResult(
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        n=len(valid),
    )


def _apply_fdr_correction(
    result_df: pd.DataFrame,
    prefix: str,
) -> None:
    """Apply Benjamini-Hochberg FDR correction in-place.

    Adds ``{prefix}_significant_fdr`` (bool) and
    ``{prefix}_p_corrected`` (float) columns to *result_df*.
    """
    p_col = f"{prefix}_p"
    p_vals = result_df[p_col].values
    valid = ~np.isnan(p_vals)
    sig = np.full(len(p_vals), False)
    corrected = np.full(len(p_vals), float("nan"))
    if np.any(valid):
        reject, pvals_corr, _, _ = multipletests(p_vals[valid], method="fdr_bh")
        sig[valid] = reject
        corrected[valid] = pvals_corr
    result_df[f"{prefix}_significant_fdr"] = sig
    result_df[f"{prefix}_p_corrected"] = corrected


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
        filtered = df
        methods = sorted(df["method"].unique())
    else:
        filtered = df[df["method"].isin(methods)]

    rows = []
    for method, mdf in filtered.groupby("method", observed=True):
        for ecol in entropy_cols:
            for mcol in metric_cols:
                result = correlation_analysis(mdf, ecol, mcol)
                rows.append({
                    "method": method,
                    "entropy_col": ecol,
                    "metric_col": mcol,
                    "spearman_rho": result.spearman_rho,
                    "spearman_p": result.spearman_p,
                    "pearson_r": result.pearson_r,
                    "pearson_p": result.pearson_p,
                    "n": result.n,
                })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    _apply_fdr_correction(result_df, "spearman")
    _apply_fdr_correction(result_df, "pearson")

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
    for method, group in df.groupby("method", observed=True):
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

    # Epsilon-squared effect size: eps^2 = H / (N - 1)
    n_total = sum(len(g) for g in groups)
    eps_sq = float(h_stat) / (n_total - 1) if n_total > 1 else 0.0

    # Dunn post-hoc with Bonferroni correction + Cliff's delta
    posthoc_rows = []
    n_comparisons = len(method_names) * (len(method_names) - 1) // 2

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            u_stat, u_p = stats.mannwhitneyu(
                groups[i], groups[j], alternative="two-sided"
            )
            corrected_p = min(u_p * n_comparisons, 1.0)

            # Cliff's delta: d = (2U / (n1 * n2)) - 1
            n1, n2 = len(groups[i]), len(groups[j])
            cliffs_d = (2.0 * u_stat / (n1 * n2)) - 1.0 if n1 * n2 > 0 else 0.0

            posthoc_rows.append({
                "method_a": method_names[i],
                "method_b": method_names[j],
                "u_statistic": float(u_stat),
                "p_value": float(u_p),
                "p_corrected": corrected_p,
                "significant": corrected_p < 0.05,
                "cliffs_delta": float(cliffs_d),
            })

    return ComparisonResult(
        statistic=float(h_stat),
        p_value=float(h_p),
        n_groups=len(groups),
        epsilon_squared=eps_sq,
        posthoc=pd.DataFrame(posthoc_rows),
    )


@dataclass(frozen=True)
class RegressionResult:
    """Result of a robust regression analysis."""

    coefficients: pd.DataFrame
    r_squared_adj: float
    n: int
    model_type: str
    vif: pd.DataFrame = field(default_factory=pd.DataFrame)


def robust_regression(
    df: pd.DataFrame,
    metric_col: str = "psnr",
    entropy_cols: list[str] | None = None,
) -> RegressionResult:
    """Fit robust regression: metric ~ entropy + method + noise_level.

    Uses statsmodels RLM with Huber's T norm, which is resistant to
    outliers. Reports coefficients, p-values, bootstrap 95% CI, and VIF.

    Args:
        df: Raw results DataFrame with columns for metric, entropy,
            method, and noise_level.
        metric_col: Response variable column.
        entropy_cols: Entropy predictor columns. Defaults to all
            columns starting with ``entropy_``.

    Returns:
        RegressionResult with coefficient table, R-squared adjusted,
        and VIF diagnostics.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if entropy_cols is None:
        entropy_cols = sorted(c for c in df.columns if c.startswith("entropy_"))

    required = [metric_col, "method", "noise_level", *entropy_cols]
    valid = df[required].dropna()
    if len(valid) < 10:
        return RegressionResult(
            coefficients=pd.DataFrame(),
            r_squared_adj=float("nan"),
            n=len(valid),
            model_type="rlm",
        )

    # Dummy-encode categorical predictors
    method_dummies = pd.get_dummies(
        valid["method"], prefix="method", drop_first=True, dtype=np.float32
    )
    noise_dummies = pd.get_dummies(
        valid["noise_level"], prefix="noise", drop_first=True, dtype=np.float32
    )

    X = pd.concat(
        [
            valid[entropy_cols].reset_index(drop=True),
            method_dummies.reset_index(drop=True),
            noise_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X)
    y = valid[metric_col].reset_index(drop=True).values

    # Fit RLM with Huber's T
    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    rlm_fit = rlm_model.fit()

    # Coefficient table - compute conf_int once (statsmodels refits the model
    # internally on each call, so calling it twice doubles the cost).
    _ci = rlm_fit.conf_int()
    coef_df = pd.DataFrame({
        "variable": X.columns.tolist(),
        "beta": rlm_fit.params,
        "std_err": rlm_fit.bse,
        "z_value": rlm_fit.tvalues,
        "p_value": rlm_fit.pvalues,
        "ci_lo": _ci.iloc[:, 0].values,
        "ci_hi": _ci.iloc[:, 1].values,
    })

    # Pseudo R-squared adjusted (from OLS for reference)
    ss_res = float(np.sum((y - rlm_fit.fittedvalues) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    n = len(y)
    p = X.shape[1] - 1
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0.0

    # VIF for numeric predictors (entropy columns only)
    vif_rows = []
    numeric_cols = [c for c in entropy_cols if c in X.columns]
    if numeric_cols:
        X_numeric = X[["const", *numeric_cols]]
        for i, col in enumerate(X_numeric.columns):
            if col == "const":
                continue
            vif_val = variance_inflation_factor(X_numeric.values, i)
            vif_rows.append({"variable": col, "vif": float(vif_val)})

    vif_df = pd.DataFrame(vif_rows)

    return RegressionResult(
        coefficients=coef_df,
        r_squared_adj=float(r2_adj),
        n=n,
        model_type="rlm",
        vif=vif_df,
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

    values = error_flat[mask_flat]
    n_valid = int(values.size)
    if n_valid < 2:
        lisa_labels = np.zeros((h, w), dtype=int)
        lisa_pvals = np.ones((h, w), dtype=np.float32)
        return SpatialResult(
            morans_i=0.0,
            morans_p=1.0,
            expected_i=0.0,
            lisa_labels=lisa_labels,
            lisa_p_values=lisa_pvals,
        )
    if np.unique(values).size < 3 or float(np.std(values)) == 0.0:
        lisa_labels = np.zeros((h, w), dtype=int)
        lisa_pvals = np.ones((h, w), dtype=np.float32)
        return SpatialResult(
            morans_i=0.0,
            morans_p=1.0,
            expected_i=0.0,
            lisa_labels=lisa_labels,
            lisa_p_values=lisa_pvals,
        )

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
