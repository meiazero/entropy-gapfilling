"""Table 3b: Noise slope by entropy bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import ENTROPY_WINDOWS, entropy_terciles, select_top_n
from .cli import run_table
from .common import bootstrap_ci_half, tex_escape, wrap_table, write_tex


def _noise_slope_value(
    mdf: pd.DataFrame,
    noise_levels: list[str],
    noise_numeric: np.ndarray,
) -> tuple[float, float]:
    def _slope_stat(
        s: pd.Series,
        data: pd.DataFrame = mdf,
    ) -> float:
        means = []
        for nl in noise_levels:
            means.append(s[data.loc[s.index, "noise_level"] == nl].mean())
        if any(np.isnan(means)):
            return float("nan")
        coef = np.polyfit(noise_numeric, np.array(means), 1)
        return float(coef[0])

    slope = _slope_stat(mdf["psnr"])
    ci = bootstrap_ci_half(
        mdf["psnr"],
        mdf["patch_id"] if "patch_id" in mdf.columns else None,
        stat_fn=_slope_stat,
    )
    return slope, ci


def _build_noise_slope_body(
    df_binned: pd.DataFrame,
    selected: list[str],
    noise_levels: list[str],
    noise_numeric: np.ndarray,
) -> list[str]:
    body: list[str] = []
    for ebin in ["baixa", "média", "alta"]:
        cells = [ebin.capitalize()]
        for method in selected:
            mdf = df_binned[
                (df_binned["method"] == method)
                & (df_binned["entropy_bin"] == ebin)
                & (df_binned["noise_level"].isin(noise_levels))
            ]
            if mdf.empty:
                cells.append("--")
                continue
            slope, ci = _noise_slope_value(mdf, noise_levels, noise_numeric)
            if np.isnan(slope):
                cells.append("--")
            else:
                cells.append(rf"${slope:.3f}_{{\pm {ci:.3f}}}$")
        body.append(" & ".join(cells) + r" \\")
    return body


def table_noise_slope(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR slope vs noise level (20/30/40 dB) by entropy bin."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl
    if not selected:
        return

    noise_levels = ["20", "30", "40"]
    noise_numeric = np.array([20.0, 30.0, 40.0])

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        df_binned = entropy_terciles(df, entropy_col=ecol)
        body = _build_noise_slope_body(
            df_binned, selected, noise_levels, noise_numeric
        )

        header = "Faixa de Entropia & " + " & ".join(
            tex_escape(m) for m in selected
        )
        tex = wrap_table(
            body,
            caption=(
                rf"Inclinação do PSNR vs ruído (20/30/40 dB, IC95\%) por faixa "
                f"de entropia (janela {ws}x{ws})."
            ),
            label=f"tab:noise-slope-psnr-e{ws}",
            col_spec="l" + "c" * len(selected),
            header=header,
            resizebox=True,
        )
        write_tex(tex, output_dir / f"tab3_slope_entropy{ws}.tex")


def main() -> None:
    run_table(table_noise_slope, "Noise slope by entropy")


if __name__ == "__main__":
    main()
