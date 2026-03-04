"""Figure 6: Boxplot PSNR by entropy bin."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..data_loader import (
    ENTROPY_WINDOWS,
    NOISE_ORDER,
    entropy_terciles,
    noise_label,
    select_top_n,
)
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = a.size, b.size
    return float((2.0 * u_stat / (n1 * n2)) - 1.0)


def fig6_psnr_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of PSNR per entropy tercile, top-k per category."""
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        for noise in noises:
            subset = df[df["noise_level"] == noise]
            if subset.empty:
                continue
            suffix = noise.replace("inf", "gap_only")
            binned = entropy_terciles(subset, entropy_col=ecol)

            top_methods: list[str] = []
            for mtype in ["Clássico", "DL"]:
                mdf = binned[binned["type"] == mtype]
                top_methods.extend(select_top_n(mdf, n=5, noise_filter=None))
            binned = binned[binned["method"].isin(top_methods)]
            if binned.empty:
                continue

            effect_labels: dict[str, str] = {}
            for method, mdf in binned.groupby("method", observed=True):
                low = mdf[mdf["entropy_bin"] == "baixa"]["psnr"].dropna()
                high = mdf[mdf["entropy_bin"] == "alta"]["psnr"].dropna()
                delta = _cliffs_delta(high.to_numpy(), low.to_numpy())
                if np.isnan(delta):
                    label = method
                else:
                    label = f"{method}\nΔ={delta:+.2f}"
                effect_labels[method] = label
            binned = binned.assign(
                method_label=binned["method"].map(effect_labels)
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(
                data=binned,
                x="method_label",
                y="psnr",
                hue="entropy_bin",
                hue_order=["baixa", "média", "alta"],
                palette="Set2",
                ax=ax,
                fliersize=3,
                linewidth=0.8,
                showmeans=True,
                meanprops={
                    "marker": "x",
                    "markeredgecolor": "#333",
                    "markersize": 4,
                },
            )
            style_axes(
                ax,
                title=(
                    f"PSNR por Faixa de Entropia ({ws}x{ws}) - "
                    f"{noise_label(noise)}"
                ),
                xlabel="Método (Δ = Cliff's d alta vs baixa)",
                ylabel="PSNR (dB)",
            )
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Entropia", loc="best", fontsize=FONT_SIZE - 2)
            plt.tight_layout()
            save_figure(fig, output_dir, f"psnr_entropy_{suffix}_e{ws}")
            plt.close(fig)


def main() -> None:
    run_with_df(fig6_psnr_entropy, "PSNR boxplot by entropy")


if __name__ == "__main__":
    main()
