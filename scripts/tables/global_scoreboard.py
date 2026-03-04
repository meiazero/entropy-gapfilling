"""Table 1: Global multi-metric scoreboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import NOISE_ORDER, noise_label
from .cli import run_table
from .common import (
    SETTINGS,
    ci95_half,
    format_pm,
    math_bold,
    tex_escape,
    wrap_table,
    write_tex,
)


def _bootstrap_ci_half_multi(
    values: pd.DataFrame,
    clusters: pd.Series | None,
    metrics: list[str],
    *,
    samples: int,
    seed: int = 42,
) -> dict[str, float]:
    if not metrics:
        return {}
    if clusters is None or clusters.isna().all():
        return {
            metric: ci95_half(values[metric].dropna()) for metric in metrics
        }

    data = values.copy()
    data["cluster"] = clusters
    data = data.dropna(subset=["cluster"])
    if data.empty:
        return {
            metric: ci95_half(values[metric].dropna()) for metric in metrics
        }

    grouped = data.groupby("cluster", sort=False)[metrics]
    cluster_sums = grouped.sum(min_count=1).to_numpy()
    cluster_counts = grouped.count().to_numpy()
    n_clusters = cluster_sums.shape[0]
    if n_clusters < 2:
        return {
            metric: ci95_half(values[metric].dropna()) for metric in metrics
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty((samples, len(metrics)), dtype=float)
    for i in range(samples):
        sampled_idx = rng.integers(0, n_clusters, size=n_clusters)
        sums = cluster_sums[sampled_idx].sum(axis=0)
        counts = cluster_counts[sampled_idx].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_means[i, :] = sums / counts

    lo = np.nanpercentile(boot_means, 2.5, axis=0)
    hi = np.nanpercentile(boot_means, 97.5, axis=0)
    ci_half = (hi - lo) / 2.0
    result: dict[str, float] = {}
    for idx, metric in enumerate(metrics):
        value = float(ci_half[idx]) if not np.isnan(ci_half[idx]) else 0.0
        result[metric] = value
    return result


def _compute_method_stats(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    cluster_col: str | None = "patch_id",
) -> pd.DataFrame:
    rows = []
    metrics_present = [m for m in metrics if m in df.columns]
    for method, grp in df.groupby("method", observed=True, sort=False):
        row: dict[str, object] = {"method": method}
        if "type" in grp.columns:
            row["type"] = grp["type"].iloc[0]
        clusters = grp[cluster_col] if cluster_col in grp.columns else None
        if metrics_present:
            means = grp[metrics_present].mean()
            ci_half = _bootstrap_ci_half_multi(
                grp[metrics_present],
                clusters,
                metrics_present,
                samples=int(SETTINGS.bootstrap_samples),
            )
            for m in metrics_present:
                mean_val = float(means[m]) if not np.isnan(means[m]) else np.nan
                row[f"{m}_mean"] = mean_val
                row[f"{m}_ci"] = ci_half.get(m, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_metric_cells(
    row: pd.Series,
    metrics: list[str],
) -> list[str]:
    cells: list[str] = []
    for metric in metrics:
        mean_val = row.get(f"{metric}_mean", np.nan)
        ci_val = row.get(f"{metric}_ci", 0.0)
        rank = int(row.get(f"{metric}_rank", 99))
        if np.isnan(mean_val):
            cells.append("--")
            continue
        base = format_pm(float(mean_val), float(ci_val))
        if rank == 1:
            cells.append(math_bold(base))
        else:
            cells.append(base)
    return cells


def _rank_metrics(
    stats_df: pd.DataFrame,
    metrics: list[str],
    higher_better: dict[str, bool],
) -> pd.DataFrame:
    for metric in metrics:
        col = f"{metric}_mean"
        if col not in stats_df.columns:
            continue
        if "type" in stats_df.columns:
            stats_df[f"{metric}_rank"] = stats_df.groupby("type")[col].rank(
                ascending=not higher_better[metric], method="min"
            )
        else:
            stats_df[f"{metric}_rank"] = stats_df[col].rank(
                ascending=not higher_better[metric], method="min"
            )
    return stats_df


def _append_noise_section(
    body: list[str],
    subset: pd.DataFrame,
    caption_noise: str,
    metrics: list[str],
    higher_better: dict[str, bool],
) -> None:
    stats_df = _compute_method_stats(subset, metrics)
    if stats_df.empty:
        return

    stats_df = _rank_metrics(stats_df, metrics, higher_better)
    stats_df = stats_df.sort_values("psnr_mean", ascending=False)
    if "type" in stats_df.columns:
        type_order = ["DL", "Clássico"]
        remaining = [
            t for t in stats_df["type"].unique().tolist() if t not in type_order
        ]
        ordered_types = [*type_order, *remaining]
        top_rows = []
        for method_type in ordered_types:
            typed = stats_df[stats_df["type"] == method_type]
            if typed.empty:
                continue
            top_rows.append(typed.head(3))
        if top_rows:
            stats_df = pd.concat(top_rows, ignore_index=True)
    else:
        stats_df = stats_df.head(3)

    if body:
        body.append(r"\midrule")
    section_title = tex_escape(f"Nível de ruído: {caption_noise}")
    body.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{section_title}}}}} \\")
    for _idx, row in stats_df.iterrows():
        method_str = tex_escape(str(row["method"]))
        type_str = str(row.get("type", ""))
        cells = [type_str, method_str]
        cells.extend(_format_metric_cells(row, metrics))
        body.append(" & ".join(cells) + r" \\")


def table_global_scoreboard(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR, SSIM, RMSE, SAM, ERGAS per method in one table."""
    metrics = ["psnr", "ssim", "rmse", "sam", "ergas"]
    higher_better = {
        "psnr": True,
        "ssim": True,
        "rmse": False,
        "sam": False,
        "ergas": False,
    }

    noise_levels = ["all"] + [
        n for n in NOISE_ORDER if n in df["noise_level"].unique()
    ]

    body: list[str] = []
    for noise in noise_levels:
        if noise == "all":
            subset = df
            caption_noise = "todos os níveis de ruído"
        else:
            subset = df[df["noise_level"] == noise]
            caption_noise = noise_label(noise)

        if subset.empty:
            continue

        _append_noise_section(
            body,
            subset,
            caption_noise,
            metrics,
            higher_better,
        )

    if not body:
        return

    header = (
        r"Tipo & Método & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ "
        r"& RMSE $\downarrow$ & SAM $\downarrow$ & ERGAS $\downarrow$"
    )
    tex = wrap_table(
        body,
        caption=(
            "Placar global multi-métrica por nível de ruído. "
            "\textbf{Negrito}: melhor por métrica dentro de cada grupo "
            "(DL e clássico)."
        ),
        label="tab:global-scoreboard",
        col_spec="llccccc",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex, output_dir / "global-scoreboard.tex")


def main() -> None:
    run_table(table_global_scoreboard, "Global scoreboard")


if __name__ == "__main__":
    main()
