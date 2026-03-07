"""Classical entropy-quality correlation heatmap."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..data_loader import ENTROPY_WINDOWS
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def _get_metrics(df: pd.DataFrame) -> list[str]:
    return [
        metric
        for metric in ["psnr", "ssim", "sam", "ergas"]
        if metric in df.columns
    ]


def _filter_classical_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "type" in df.columns:
        return df[df["type"] == "Clássico"]
    return df


def _build_column_labels(metrics: list[str]) -> list[str]:
    return [
        f"e{window}x{metric.upper()}"
        for window in ENTROPY_WINDOWS
        for metric in metrics
    ]


def _compute_correlation_matrices(
    df: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    col_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    rho_matrix = np.full((len(methods), len(col_labels)), np.nan)
    p_matrix = np.full((len(methods), len(col_labels)), np.nan)

    for row_index, method in enumerate(methods):
        method_df = df[df["method"] == method]
        col_index = 0
        for window in ENTROPY_WINDOWS:
            entropy_col = f"entropy_{window}"
            for metric in metrics:
                if (
                    entropy_col in method_df.columns
                    and metric in method_df.columns
                ):
                    valid = method_df[[entropy_col, metric]].dropna()
                    if len(valid) >= 3:
                        rho, p_value = stats.spearmanr(
                            valid[entropy_col], valid[metric]
                        )
                        rho_matrix[row_index, col_index] = rho
                        p_matrix[row_index, col_index] = p_value
                col_index += 1

    return rho_matrix, p_matrix


def _mask_non_significant_correlations(
    rho_matrix: np.ndarray,
    p_matrix: np.ndarray,
) -> np.ndarray:
    p_values = p_matrix.ravel()
    valid_mask = ~np.isnan(p_values)
    corrected_p = np.full_like(p_values, np.nan, dtype=float)
    significant = np.full_like(p_values, False, dtype=bool)

    if np.any(valid_mask):
        reject, p_corr, _, _ = multipletests(
            p_values[valid_mask], method="fdr_bh"
        )
        corrected_p[valid_mask] = p_corr
        significant[valid_mask] = reject

    final = rho_matrix.copy()
    final[~significant.reshape(p_matrix.shape)] = np.nan
    final[np.abs(rho_matrix) < 0.1] = np.nan
    return final


def fig_classical_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """Heatmap of Spearman rho for classical methods only."""
    metrics = _get_metrics(df)
    if not metrics:
        log.warning("No metrics for fig7")
        return

    df = _filter_classical_rows(df)
    if df.empty:
        log.warning("No classical rows for correlation heatmap")
        return

    methods = sorted(df["method"].unique())
    col_labels = _build_column_labels(metrics)
    rho_matrix, p_matrix = _compute_correlation_matrices(
        df,
        methods,
        metrics,
        col_labels,
    )
    final = _mask_non_significant_correlations(rho_matrix, p_matrix)

    fig, ax = plt.subplots(
        figsize=(len(col_labels) * 0.8 + 1, len(methods) * 0.4 + 1)
    )
    sns.heatmap(
        pd.DataFrame(final, index=methods, columns=col_labels),
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        ax=ax,
        annot_kws={"size": FONT_SIZE - 2},
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    style_axes(
        ax,
        xlabel="Janela de entropia x métrica",
        ylabel="Método",
        grid=False,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save_figure(fig, output_dir, "fig_classical_correlation_heatmap")
    plt.close(fig)


def main() -> None:
    run_with_df(
        fig_classical_correlation_heatmap,
        "Classical correlation heatmap",
    )


if __name__ == "__main__":
    main()
