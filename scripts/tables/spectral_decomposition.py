"""Table 2: Spectral decomposition (RMSE per band)."""

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


def _build_spectral_stats(
    subset: pd.DataFrame,
    bands: list[str],
) -> pd.DataFrame:
    stats_rows: list[dict[str, object]] = []
    for method, grp in subset.groupby("method", observed=True):
        row: dict[str, object] = {
            "method": method,
            "type": grp["type"].iloc[0] if "type" in grp.columns else "",
        }
        clusters = grp["patch_id"] if "patch_id" in grp.columns else None
        for band in bands:
            if band in grp.columns:
                vals = grp[band].dropna()
                row[f"{band}_median"] = (
                    float(vals.median()) if len(vals) > 0 else np.nan
                )
                row[f"{band}_ci"] = bootstrap_ci_half(
                    vals,
                    clusters,
                    stat_fn=np.median,
                )
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    for band in bands:
        col = f"{band}_median"
        if col in stats_df.columns:
            stats_df[f"{band}_rank"] = stats_df[col].rank(
                ascending=True, method="min"
            )

    return stats_df.sort_values("rmse_b0_median", ascending=True)


def _format_spectral_body(
    stats_df: pd.DataFrame,
    bands: list[str],
) -> list[str]:
    body: list[str] = []
    for _, row in stats_df.iterrows():
        cells = [str(row.get("type", "")), tex_escape(str(row["method"]))]
        for band in bands:
            med_val = row.get(f"{band}_median", np.nan)
            ci_val = row.get(f"{band}_ci", 0.0)
            rank = int(row.get(f"{band}_rank", 99))
            if np.isnan(med_val):
                cells.append("--")
            else:
                cells.append(ranked_cell(med_val, ci_val, rank, ".4f"))
        body.append(" & ".join(cells) + r" \\")
    return body


def table_spectral_decomposition(df: pd.DataFrame, output_dir: Path) -> None:
    """RMSE per spectral band for each method."""
    bands = ["rmse_b0", "rmse_b1", "rmse_b2", "rmse_b3"]
    band_labels = ["B0 (Azul)", "B1 (Verde)", "B2 (Vermelho)", "B3 (NIR)"]

    noise_levels = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]

    for noise in noise_levels:
        subset = df[df["noise_level"] == noise]
        if subset.empty:
            continue

        suffix = noise.replace("inf", "gap_only")

        stats_df = _build_spectral_stats(subset, bands)
        if stats_df.empty:
            continue

        body = _format_spectral_body(stats_df, bands)

        header = "Tipo & Método & " + " & ".join(
            f"{bl} $\\downarrow$" for bl in band_labels
        )
        tex = wrap_table(
            body,
            caption=(
                rf"RMSE mediano por banda (IC95\%) ({noise_label(noise)}). "
                r"Menor é melhor."
            ),
            label=f"tab:spectral-rmse-{suffix}",
            col_spec="llcccc",
            header=header,
            resizebox=True,
        )
        write_tex(tex, output_dir / f"tab2_spectral_{suffix}.tex")


def main() -> None:
    run_table(table_spectral_decomposition, "Spectral decomposition")


if __name__ == "__main__":
    main()
