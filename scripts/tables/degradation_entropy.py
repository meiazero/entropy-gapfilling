"""Table 3: Degradation by entropy and noise."""

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


def _drop_pct_stat(sample: pd.Series, data: pd.DataFrame) -> float:
    labels = data.loc[sample.index, "noise_level"]
    inf_vals = sample[labels == "inf"]
    n20_vals = sample[labels == "20"]
    if inf_vals.empty or n20_vals.empty:
        return float("nan")
    inf_med = inf_vals.median()
    if inf_med == 0 or np.isnan(inf_med):
        return float("nan")
    return float((inf_med - n20_vals.median()) / inf_med * 100)


def _drop_pct_iqr(mdf: pd.DataFrame) -> float:
    if "patch_id" not in mdf.columns:
        return float("nan")
    grouped = mdf.dropna(subset=["patch_id"]).groupby("patch_id")
    values = []
    for _pid, grp in grouped:
        inf_vals = grp[grp["noise_level"] == "inf"]["psnr"].dropna()
        n20_vals = grp[grp["noise_level"] == "20"]["psnr"].dropna()
        if inf_vals.empty or n20_vals.empty:
            continue
        inf_med = inf_vals.median()
        if inf_med == 0 or np.isnan(inf_med):
            continue
        values.append(float((inf_med - n20_vals.median()) / inf_med * 100))
    if not values:
        return float("nan")
    return iqr(pd.Series(values))


def _section_header_line(section_title: str, n_cols: int) -> str:
    return (
        rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{section_title}}}}} "
        r"\\"
    )


def _build_degradation_rows(
    df_binned: pd.DataFrame,
    selected: list[str],
) -> tuple[list[str], list[str]]:
    body: list[str] = []
    body_iqr: list[str] = []
    for ebin in ["baixa", "média", "alta"]:
        cells = [ebin.capitalize()]
        cells_iqr = [ebin.capitalize()]
        for method in selected:
            mdf = df_binned[
                (df_binned["method"] == method)
                & (df_binned["entropy_bin"] == ebin)
            ]
            mdf, clusters = _filter_clustered(mdf)
            if mdf.empty:
                cells.append("--")
                cells_iqr.append("--")
                continue

            gap_only = mdf[mdf["noise_level"] == "inf"]["psnr"].dropna()
            noisy_20 = mdf[mdf["noise_level"] == "20"]["psnr"].dropna()
            if gap_only.empty or noisy_20.empty:
                cells.append("--")
                cells_iqr.append("--")
                continue

            drop_pct = _drop_pct_stat(mdf["psnr"], mdf)

            def _drop_stat(
                sample: pd.Series, data: pd.DataFrame = mdf
            ) -> float:
                return _drop_pct_stat(sample, data)

            ci = bootstrap_ci_half(
                mdf["psnr"],
                clusters,
                stat_fn=_drop_stat,
            )
            drop_iqr = _drop_pct_iqr(mdf)
            cells.append(f"${drop_pct:.1f}\\%_{{\\pm {ci:.1f}}}$")
            if np.isnan(drop_iqr):
                cells_iqr.append("--")
            else:
                cells_iqr.append(
                    f"${drop_pct:.1f}\\%\\,(IQR\\ {drop_iqr:.1f})$"
                )
        body.append(" & ".join(cells) + r" \\")
        body_iqr.append(" & ".join(cells_iqr) + r" \\")
    return body, body_iqr


def table_degradation_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR drop (%) from gap_only to 20dB, stratified by entropy."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl

    if not selected:
        return

    header = "Faixa de Entropia & " + " & ".join(
        tex_escape(m) for m in selected
    )
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
        section_line = _section_header_line(section_title, len(selected) + 1)
        body.append(section_line)
        body_iqr.append(section_line)

        rows, rows_iqr = _build_degradation_rows(df_binned, selected)
        body.extend(rows)
        body_iqr.extend(rows_iqr)

    if not body:
        return

    tex = wrap_table(
        body,
        caption=(
            r"Queda percentual no PSNR (mediana, IC95\%) "
            "(sem ruído → 20 dB), estratificada por faixa de entropia e "
            "janela de cálculo. Top-3 clássicos + top-3 DL."
        ),
        label="tab:degradation-psnr",
        col_spec="l" + "c" * len(selected),
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex, output_dir / "psnr-drop-entropy.tex")

    tex_iqr = wrap_table(
        body_iqr,
        caption=(
            "Queda percentual no PSNR (mediana, IQR) "
            "(sem ruído → 20 dB), estratificada por faixa de entropia e "
            "janela de cálculo. Top-3 clássicos + top-3 DL."
        ),
        label="tab:degradation-psnr-iqr",
        col_spec="l" + "c" * len(selected),
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex_iqr, output_dir / "psnr-drop-entropy-iqr.tex")


def main() -> None:
    run_table(table_degradation_entropy, "Degradation by entropy")


if __name__ == "__main__":
    main()
