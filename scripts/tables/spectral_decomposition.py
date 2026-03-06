"""Table 2: Spectral decomposition (RMSE per band)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import NOISE_ORDER, noise_label
from .cli import run_table
from .common import (
    bootstrap_ci_half,
    iqr,
    ranked_cell,
    ranked_cell_iqr,
    tex_escape,
    wrap_table,
    write_tex,
)


def _build_spectral_stats(
    subset: pd.DataFrame,
    bands: list[str],
) -> pd.DataFrame:
    bands_present = [band for band in bands if band in subset.columns]
    if not bands_present:
        return pd.DataFrame()

    stats_rows: list[dict[str, object]] = []
    grouped = subset.groupby("method", observed=True, sort=False)
    for method, grp in grouped:
        row: dict[str, object] = {
            "method": method,
            "type": grp["type"].iloc[0] if "type" in grp.columns else "",
        }
        clusters = grp["patch_id"] if "patch_id" in grp.columns else None
        if clusters is not None and clusters.notna().any():
            mask = clusters.notna()
            grp = grp.loc[mask]
            clusters = clusters.loc[mask]
        else:
            clusters = None
        grp_bands = grp[bands_present]
        for band in bands_present:
            vals = grp_bands[band].dropna()
            if vals.empty:
                row[f"{band}_median"] = np.nan
                row[f"{band}_ci"] = 0.0
                row[f"{band}_iqr"] = 0.0
                continue
            row[f"{band}_median"] = float(vals.median())
            row[f"{band}_ci"] = bootstrap_ci_half(
                vals,
                clusters,
                stat_fn=np.median,
            )
            row[f"{band}_iqr"] = iqr(vals)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    if stats_df.empty:
        return stats_df

    for band in bands_present:
        col = f"{band}_median"
        stats_df[f"{band}_rank"] = stats_df[col].rank(
            ascending=True,
            method="min",
        )

    sort_col = f"{bands[0]}_median"
    if sort_col in stats_df.columns:
        return stats_df.sort_values(sort_col, ascending=True)
    return stats_df.sort_values("method", ascending=True)


def _format_spectral_body(
    stats_df: pd.DataFrame,
    bands: list[str],
    *,
    use_iqr: bool,
) -> list[str]:
    body: list[str] = []
    for row in stats_df.itertuples(index=False):
        row_map = row._asdict()
        cells = [
            str(row_map.get("type", "")),
            tex_escape(str(row_map["method"])),
        ]
        for band in bands:
            med_val = row_map.get(f"{band}_median", np.nan)
            if np.isnan(med_val):
                cells.append("--")
                continue
            rank = int(row_map.get(f"{band}_rank", 99))
            if use_iqr:
                iqr_val = float(row_map.get(f"{band}_iqr", 0.0))
                cells.append(
                    ranked_cell_iqr(float(med_val), iqr_val, rank, ".3f")
                )
            else:
                ci_val = float(row_map.get(f"{band}_ci", 0.0))
                cells.append(ranked_cell(float(med_val), ci_val, rank, ".3f"))
        body.append(" & ".join(cells) + r" \\")
    return body


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


def _append_spectral_section(
    body: list[str],
    body_iqr: list[str],
    *,
    stats_df: pd.DataFrame,
    bands: list[str],
    section_title: str,
) -> None:
    if body:
        body.append(r"\midrule")
    if body_iqr:
        body_iqr.append(r"\midrule")
    title = tex_escape(section_title)
    body.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\")
    body_iqr.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\")
    body.extend(_format_spectral_body(stats_df, bands, use_iqr=False))
    body_iqr.extend(_format_spectral_body(stats_df, bands, use_iqr=True))


def table_spectral_decomposition(df: pd.DataFrame, output_dir: Path) -> None:
    """RMSE per spectral band for each method."""
    bands = ["rmse_b0", "rmse_b1", "rmse_b2", "rmse_b3"]
    band_labels = ["B0 (Azul)", "B1 (Verde)", "B2 (Vermelho)", "B3 (NIR)"]

    noise_levels = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    grouped_noise = df.groupby(
        "noise_level",
        observed=True,
        sort=False,
    )

    body: list[str] = []
    body_iqr: list[str] = []
    for noise in noise_levels:
        try:
            subset = grouped_noise.get_group(noise)
        except KeyError:
            continue
        if subset.empty:
            continue

        stats_df = _build_spectral_stats(subset, bands)
        if stats_df.empty:
            continue

        stats_df = stats_df.sort_values("rmse_b0_median", ascending=True)
        stats_df = _select_top_methods(stats_df)

        _append_spectral_section(
            body,
            body_iqr,
            stats_df=stats_df,
            bands=bands,
            section_title=f"Nível de ruído: {noise_label(noise)}",
        )

    if not body:
        return

    header = "Tipo & Método & " + " & ".join(
        f"{bl} $\\downarrow$" for bl in band_labels
    )
    tex = wrap_table(
        body,
        caption=(
            r"RMSE mediano por banda (IC95\%) por nível de ruído. "
            r"Menor é melhor."
        ),
        label="tab:spectral-rmse",
        col_spec="llcccc",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex, output_dir / "spectral-rmse.tex")

    if not body_iqr:
        return

    tex_iqr = wrap_table(
        body_iqr,
        caption=(
            r"RMSE mediano por banda com IQR por nível de ruído. "
            r"Menor é melhor."
        ),
        label="tab:spectral-rmse-iqr",
        col_spec="llcccc",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex_iqr, output_dir / "spectral-rmse-iqr.tex")


def main() -> None:
    run_table(table_spectral_decomposition, "Spectral decomposition")


if __name__ == "__main__":
    main()
