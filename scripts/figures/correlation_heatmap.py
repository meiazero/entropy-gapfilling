"""Figure 7: Correlation heatmap."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_loader import ENTROPY_WINDOWS
from figures.common import FONT_SIZE, save_figure, style_axes
from scipy import stats
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)


def fig7_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of Spearman rho with FDR and effect filtering."""
    metrics = [m for m in ["psnr", "ssim", "sam", "ergas"] if m in df.columns]
    if not metrics:
        log.warning("No metrics for fig7")
        return

    methods = sorted(df["method"].unique())

    col_labels = []
    for ws in ENTROPY_WINDOWS:
        for m in metrics:
            col_labels.append(f"e{ws}x{m.upper()}")

    rho_matrix = np.full((len(methods), len(col_labels)), np.nan)
    p_matrix = np.full((len(methods), len(col_labels)), np.nan)

    for i, method in enumerate(methods):
        mdf = df[df["method"] == method]
        col_idx = 0
        for ws in ENTROPY_WINDOWS:
            ecol = f"entropy_{ws}"
            for m in metrics:
                if ecol in mdf.columns and m in mdf.columns:
                    valid = mdf[[ecol, m]].dropna()
                    if len(valid) >= 3:
                        rho, p = stats.spearmanr(valid[ecol], valid[m])
                        rho_matrix[i, col_idx] = rho
                        p_matrix[i, col_idx] = p
                col_idx += 1

    p_vals = p_matrix.ravel()
    valid = ~np.isnan(p_vals)
    corr_p = np.full_like(p_vals, np.nan, dtype=float)
    sig = np.full_like(p_vals, False, dtype=bool)
    if np.any(valid):
        reject, p_corr, _, _ = multipletests(p_vals[valid], method="fdr_bh")
        corr_p[valid] = p_corr
        sig[valid] = reject
    sig_matrix = sig.reshape(p_matrix.shape)

    effect_mask = np.abs(rho_matrix) < 0.1
    final = rho_matrix.copy()
    final[~sig_matrix] = np.nan
    final[effect_mask] = np.nan

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
        title="Correlação Spearman (FDR, |rho|>=0,1)",
        ylabel="",
        grid=False,
        title_size=FONT_SIZE + 1,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save_figure(fig, output_dir, "fig7_correlation_heatmap")
    plt.close(fig)
