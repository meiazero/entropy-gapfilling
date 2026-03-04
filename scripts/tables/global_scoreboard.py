"""Table 1: Global multi-metric scoreboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import NOISE_ORDER, noise_label
from .cli import run_table
from .common import (
    bootstrap_ci_half,
    ranked_cell,
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
    for method, grp in df.groupby("method", observed=True):
        row: dict[str, object] = {"method": method}
        if "type" in grp.columns:
            row["type"] = grp["type"].iloc[0]
        clusters = grp[cluster_col] if cluster_col in grp.columns else None
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{m}_ci"] = bootstrap_ci_half(
                vals,
                clusters,
                stat_fn=np.mean,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def table_global_scoreboard(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR, SSIM, RMSE, SAM, ERGAS per method (all + per noise)."""
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

    for noise in noise_levels:
        if noise == "all":
            subset = df
            suffix = "all"
            caption_noise = "todos os níveis de ruído"
        else:
            subset = df[df["noise_level"] == noise]
            suffix = noise.replace("inf", "gap_only")
            caption_noise = noise_label(noise)

        if subset.empty:
            continue

        stats_df = _compute_method_stats(subset, metrics)
        if stats_df.empty:
            continue

        for m in metrics:
            col = f"{m}_mean"
            if col in stats_df.columns:
                stats_df[f"{m}_rank"] = stats_df[col].rank(
                    ascending=not higher_better[m], method="min"
                )

        stats_df = stats_df.sort_values("psnr_mean", ascending=False)

        body: list[str] = []
        for _, row in stats_df.iterrows():
            method_str = tex_escape(str(row["method"]))
            type_str = str(row.get("type", ""))
            cells = [type_str, method_str]
            for m in metrics:
                mean_val = row.get(f"{m}_mean", np.nan)
                ci_val = row.get(f"{m}_ci", 0.0)
                rank = int(row.get(f"{m}_rank", 99))
                if np.isnan(mean_val):
                    cells.append("--")
                else:
                    cells.append(ranked_cell(mean_val, ci_val, rank))
            body.append(" & ".join(cells) + r" \\")

        header = (
            r"Tipo & Método & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ "
            r"& RMSE $\downarrow$ & SAM $\downarrow$ & ERGAS $\downarrow$"
        )
        tex = wrap_table(
            body,
            caption=(
                f"Placar global multi-métrica ({caption_noise}). "
                r"\textbf{Negrito}: melhor; \underline{sublinhado}: 2º; "
                r"\textit{itálico}: 3º."
            ),
            label=f"tab:global-scoreboard-{suffix}",
            col_spec="llccccc",
            header=header,
            env="table*",
            resizebox=True,
        )
        write_tex(tex, output_dir / f"tab1_global_{suffix}.tex")


def main() -> None:
    run_table(table_global_scoreboard, "Global scoreboard")


if __name__ == "__main__":
    main()
