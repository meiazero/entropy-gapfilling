"""Table 3b: Noise slope by entropy bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import ENTROPY_WINDOWS, entropy_terciles, select_top_n
from .cli import run_table
from .common import bootstrap_ci_half, iqr, tex_escape, wrap_table, write_tex


def _filter_clustered(
    mdf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if "patch_id" not in mdf.columns:
        return mdf, None
    clusters = mdf["patch_id"]
    if clusters.notna().any():
        mask = clusters.notna()
        return mdf.loc[mask], clusters.loc[mask]
    return mdf.iloc[0:0], None


def _noise_slope_value(
    mdf: pd.DataFrame,
    noise_levels: list[str],
    noise_numeric: np.ndarray,
) -> tuple[float, float]:
    def _slope_stat(
        s: pd.Series,
        data: pd.DataFrame = mdf,
    ) -> float:
        medians = []
        for nl in noise_levels:
            medians.append(s[data.loc[s.index, "noise_level"] == nl].median())
        if any(np.isnan(medians)):
            return float("nan")
        coef = np.polyfit(noise_numeric, np.array(medians), 1)
        return float(coef[0])

    slope = _slope_stat(mdf["psnr"])
    ci = bootstrap_ci_half(
        mdf["psnr"],
        mdf["patch_id"] if "patch_id" in mdf.columns else None,
        stat_fn=_slope_stat,
    )
    return slope, ci


def _noise_slope_iqr(
    mdf: pd.DataFrame,
    noise_levels: list[str],
    noise_numeric: np.ndarray,
) -> float:
    if "patch_id" not in mdf.columns:
        return float("nan")
    grouped = mdf.dropna(subset=["patch_id"]).groupby("patch_id")
    slopes = []
    for _pid, grp in grouped:
        medians = []
        for nl in noise_levels:
            vals = grp[grp["noise_level"] == nl]["psnr"].dropna()
            medians.append(vals.median())
        if any(np.isnan(medians)):
            continue
        coef = np.polyfit(noise_numeric, np.array(medians), 1)
        slopes.append(float(coef[0]))
    if not slopes:
        return float("nan")
    return iqr(pd.Series(slopes))


def _build_noise_slope_body(
    df_binned: pd.DataFrame,
    methods: list[str],
    entropy_bins: list[str],
    noise_levels: list[str],
    noise_numeric: np.ndarray,
    *,
    use_iqr: bool,
) -> list[str]:
    body: list[str] = []
    for method in methods:
        cells = [tex_escape(method)]
        for ebin in entropy_bins:
            mdf = df_binned[
                (df_binned["method"] == method)
                & (df_binned["entropy_bin"] == ebin)
                & (df_binned["noise_level"].isin(noise_levels))
            ]
            mdf, _clusters = _filter_clustered(mdf)
            if mdf.empty:
                cells.append("--")
                continue
            if use_iqr:
                slope = _noise_slope_value(mdf, noise_levels, noise_numeric)[0]
                slope_iqr = _noise_slope_iqr(mdf, noise_levels, noise_numeric)
                if np.isnan(slope) or np.isnan(slope_iqr):
                    cells.append("--")
                else:
                    cells.append(rf"${slope:.3f}\,(IQR\ {slope_iqr:.3f})$")
                continue

            slope, ci = _noise_slope_value(mdf, noise_levels, noise_numeric)
            if np.isnan(slope):
                cells.append("--")
            else:
                cells.append(rf"${slope:.3f}_{{\pm {ci:.3f}}}$")
        body.append(" & ".join(cells) + r" \\")
    return body


def _section_header_line(section_title: str, n_cols: int) -> str:
    return (
        rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{section_title}}}}} "
        r"\\"
    )


def table_noise_slope(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR slope vs noise level (20/30/40 dB) by entropy bin."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    entropy_bins = ["baixa", "média", "alta"]
    if not (top_classic or top_dl):
        return

    noise_levels = ["20", "30", "40"]
    noise_numeric = np.array([20.0, 30.0, 40.0])

    header = "Método & " + " & ".join(b.capitalize() for b in entropy_bins)
    body: list[str] = []
    body_iqr: list[str] = []

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        df_binned = entropy_terciles(df, entropy_col=ecol)
        if body:
            body.append(r"\midrule")
        if body_iqr:
            body_iqr.append(r"\midrule")
        section_title = tex_escape(f"Janela de entropia: {ws}x{ws}")
        section_line = _section_header_line(
            section_title, len(entropy_bins) + 1
        )
        body.append(section_line)
        body_iqr.append(section_line)

        if top_classic:
            classic_line = _section_header_line(
                tex_escape("Clássico"),
                len(entropy_bins) + 1,
            )
            body.append(classic_line)
            body_iqr.append(classic_line)
            section_body = _build_noise_slope_body(
                df_binned,
                top_classic,
                entropy_bins,
                noise_levels,
                noise_numeric,
                use_iqr=False,
            )
            section_body_iqr = _build_noise_slope_body(
                df_binned,
                top_classic,
                entropy_bins,
                noise_levels,
                noise_numeric,
                use_iqr=True,
            )
            body.extend(section_body)
            body_iqr.extend(section_body_iqr)

        if top_dl:
            if top_classic:
                body.append(r"\midrule")
                body_iqr.append(r"\midrule")
            dl_line = _section_header_line(
                tex_escape("DL"),
                len(entropy_bins) + 1,
            )
            body.append(dl_line)
            body_iqr.append(dl_line)
            section_body = _build_noise_slope_body(
                df_binned,
                top_dl,
                entropy_bins,
                noise_levels,
                noise_numeric,
                use_iqr=False,
            )
            section_body_iqr = _build_noise_slope_body(
                df_binned,
                top_dl,
                entropy_bins,
                noise_levels,
                noise_numeric,
                use_iqr=True,
            )
            body.extend(section_body)
            body_iqr.extend(section_body_iqr)

    if not body:
        return

    tex = wrap_table(
        body,
        caption=(
            r"Inclinação do PSNR vs ruído (20/30/40 dB, mediana, IC95\%), "
            "estratificada por faixa de entropia e janela de cálculo."
        ),
        label="tab:noise-slope-psnr",
        col_spec="l" + "c" * len(entropy_bins),
        header=header,
        env="table",
        resizebox=True,
    )
    write_tex(tex, output_dir / "psnr-noise-slope-entropy.tex")

    tex_iqr = wrap_table(
        body_iqr,
        caption=(
            "Inclinação do PSNR vs ruído (20/30/40 dB, mediana, IQR), "
            "estratificada por faixa de entropia e janela de cálculo."
        ),
        label="tab:noise-slope-psnr-iqr",
        col_spec="l" + "c" * len(entropy_bins),
        header=header,
        env="table",
        resizebox=True,
    )
    write_tex(tex_iqr, output_dir / "psnr-noise-slope-entropy-iqr.tex")


def main() -> None:
    run_table(table_noise_slope, "Noise slope by entropy")


if __name__ == "__main__":
    main()
