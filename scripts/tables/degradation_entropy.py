"""Table 3: Degradation by entropy and noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from data_loader import ENTROPY_WINDOWS, entropy_terciles, select_top_n
from tables.cli import run_table
from tables.common import bootstrap_ci_half, tex_escape, wrap_table, write_tex


def table_degradation_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR drop (%) from gap_only to 20dB, stratified by entropy."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl

    if not selected:
        return

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        df_binned = entropy_terciles(df, entropy_col=ecol)

        body: list[str] = []
        for ebin in ["baixa", "média", "alta"]:
            cells = [ebin.capitalize()]
            for method in selected:
                mdf = df_binned[
                    (df_binned["method"] == method)
                    & (df_binned["entropy_bin"] == ebin)
                ]
                gap_only = mdf[mdf["noise_level"] == "inf"]["psnr"].dropna()
                noisy_20 = mdf[mdf["noise_level"] == "20"]["psnr"].dropna()
                clusters = (
                    mdf["patch_id"] if "patch_id" in mdf.columns else None
                )

                if gap_only.empty or noisy_20.empty:
                    cells.append("--")
                else:
                    drop_pct = (
                        (gap_only.mean() - noisy_20.mean())
                        / gap_only.mean()
                        * 100
                    )

                    def _drop_stat(
                        sample: pd.Series,
                        data: pd.DataFrame = mdf,
                    ) -> float:
                        idx = sample.index
                        labels = data.loc[idx, "noise_level"]
                        inf_vals = sample[labels == "inf"]
                        n20_vals = sample[labels == "20"]
                        if inf_vals.empty or n20_vals.empty:
                            return float("nan")
                        inf_mean = inf_vals.mean()
                        if inf_mean == 0 or np.isnan(inf_mean):
                            return float("nan")
                        return float(
                            (inf_mean - n20_vals.mean()) / inf_mean * 100
                        )

                    ci = bootstrap_ci_half(
                        mdf["psnr"],
                        clusters,
                        stat_fn=_drop_stat,
                    )
                    cells.append(f"${drop_pct:.1f}\\%_{{\\pm {ci:.1f}}}$")
            body.append(" & ".join(cells) + r" \\")

        header = "Faixa de Entropia & " + " & ".join(
            tex_escape(m) for m in selected
        )
        tex = wrap_table(
            body,
            caption=(
                rf"Queda percentual no PSNR (IC95\%) (sem ruído → 20 dB) "
                f"por faixa de entropia (janela {ws}x{ws}). "
                "Top-3 clássicos + top-3 DL."
            ),
            label=f"tab:degradation-psnr-e{ws}",
            col_spec="l" + "c" * len(selected),
            header=header,
            resizebox=True,
        )
        write_tex(tex, output_dir / f"tab3_degradation_entropy{ws}.tex")


def main() -> None:
    run_table(table_degradation_entropy, "Degradation by entropy")


if __name__ == "__main__":
    main()
