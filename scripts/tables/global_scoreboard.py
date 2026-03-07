"""Table 1: Global multi-metric scoreboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import (
    NOISE_ORDER,
    display_method_name,
    filter_main_dl_scenario,
    noise_label,
)
from .cli import run_table
from .common import (
    bootstrap_ci_half,
    format_iqr,
    format_pm,
    iqr,
    math_bold,
    tex_escape,
    wrap_table,
    write_tex,
)


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
        if clusters is not None and clusters.notna().any():
            mask = clusters.notna()
            grp = grp.loc[mask]
            clusters = clusters.loc[mask]
        else:
            clusters = None
        if metrics_present:
            for m in metrics_present:
                vals = grp[m].dropna()
                if vals.empty:
                    row[f"{m}_median"] = np.nan
                    row[f"{m}_ci"] = 0.0
                    row[f"{m}_iqr"] = 0.0
                    continue
                row[f"{m}_median"] = float(vals.median())
                row[f"{m}_ci"] = bootstrap_ci_half(
                    vals,
                    clusters,
                    stat_fn=np.median,
                )
                row[f"{m}_iqr"] = iqr(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_metric_cells(
    row: pd.Series,
    metrics: list[str],
) -> list[str]:
    cells: list[str] = []
    for metric in metrics:
        median_val = row.get(f"{metric}_median", np.nan)
        ci_val = row.get(f"{metric}_ci", 0.0)
        rank = int(row.get(f"{metric}_rank", 99))
        if np.isnan(median_val):
            cells.append("--")
            continue
        base = format_pm(float(median_val), float(ci_val), ci_fmt=".3f")
        if rank == 1:
            cells.append(math_bold(base))
        else:
            cells.append(base)
    return cells


def _format_metric_cells_iqr(
    row: pd.Series,
    metrics: list[str],
) -> list[str]:
    cells: list[str] = []
    for metric in metrics:
        median_val = row.get(f"{metric}_median", np.nan)
        iqr_val = row.get(f"{metric}_iqr", 0.0)
        rank = int(row.get(f"{metric}_rank", 99))
        if np.isnan(median_val):
            cells.append("--")
            continue
        base = format_iqr(float(median_val), float(iqr_val))
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
        col = f"{metric}_median"
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


def _select_top_methods(stats_df: pd.DataFrame) -> pd.DataFrame:
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
            return pd.concat(top_rows, ignore_index=True)
        return stats_df
    return stats_df.head(3)


def _append_noise_section(
    body: list[str],
    subset: pd.DataFrame,
    caption_noise: str,
    metrics: list[str],
    higher_better: dict[str, bool],
    body_iqr: list[str] | None = None,
) -> None:
    stats_df = _compute_method_stats(subset, metrics)
    if stats_df.empty:
        return

    stats_df = _rank_metrics(stats_df, metrics, higher_better)
    stats_df = stats_df.sort_values("psnr_median", ascending=False)
    stats_df = _select_top_methods(stats_df)

    if body:
        body.append(r"\midrule")
    if body_iqr is not None and body_iqr:
        body_iqr.append(r"\midrule")
    section_title = tex_escape(f"Nível de ruído: {caption_noise}")
    body.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{section_title}}}}} \\")
    if body_iqr is not None:
        body_iqr.append(
            rf"\multicolumn{{7}}{{l}}{{\textbf{{{section_title}}}}} \\"
        )
    for _idx, row in stats_df.iterrows():
        method_str = tex_escape(display_method_name(str(row["method"])))
        type_str = str(row.get("type", ""))
        cells = [type_str, method_str]
        cells.extend(_format_metric_cells(row, metrics))
        body.append(" & ".join(cells) + r" \\")
        if body_iqr is not None:
            cells_iqr = [type_str, method_str]
            cells_iqr.extend(_format_metric_cells_iqr(row, metrics))
            body_iqr.append(" & ".join(cells_iqr) + r" \\")


def _table_caption(group_label: str, *, use_iqr: bool) -> str:
    spread = "mediana e IQR" if use_iqr else r"mediana e IC95\%"
    return (
        "Placar global multi-métrica "
        f"dos métodos {group_label} "
        "por nível de ruído "
        f"({spread}). "
        r"\textbf{Negrito}: melhor por métrica dentro do grupo."
    )


def _type_slug(method_type: str) -> str:
    return "classical" if method_type == "Clássico" else "dl"


def _write_type_table(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    method_type: str,
    metrics: list[str],
    higher_better: dict[str, bool],
) -> None:
    type_df = df[df["type"] == method_type]
    if method_type == "DL":
        type_df = filter_main_dl_scenario(type_df)
    if type_df.empty:
        return

    noise_levels = ["all"] + [
        n for n in NOISE_ORDER if n in type_df["noise_level"].unique()
    ]
    body: list[str] = []
    body_iqr: list[str] = []
    for noise in noise_levels:
        if noise == "all":
            subset = type_df
            caption_noise = "todos os níveis de ruído"
        else:
            subset = type_df[type_df["noise_level"] == noise]
            caption_noise = noise_label(noise)
        if subset.empty:
            continue

        _append_noise_section(
            body,
            subset,
            caption_noise,
            metrics,
            higher_better,
            body_iqr,
        )

    if not body:
        return

    group_label = "clássicos" if method_type == "Clássico" else "de DL"
    slug = _type_slug(method_type)
    header = (
        r"Tipo & Método & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ "
        r"& RMSE $\downarrow$ & SAM $\downarrow$ & ERGAS $\downarrow$"
    )
    tex = wrap_table(
        body,
        caption=_table_caption(group_label, use_iqr=False),
        label=f"tab:global-scoreboard-{slug}",
        col_spec="llccccc",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex, output_dir / f"global-scoreboard-{slug}.tex")

    if not body_iqr:
        return

    tex_iqr = wrap_table(
        body_iqr,
        caption=_table_caption(group_label, use_iqr=True),
        label=f"tab:global-scoreboard-{slug}-iqr",
        col_spec="llccccc",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex_iqr, output_dir / f"global-scoreboard-{slug}-iqr.tex")


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
    for method_type in ["Clássico", "DL"]:
        _write_type_table(
            df,
            output_dir,
            method_type=method_type,
            metrics=metrics,
            higher_better=higher_better,
        )


def main() -> None:
    run_table(table_global_scoreboard, "Global scoreboard")


if __name__ == "__main__":
    main()
