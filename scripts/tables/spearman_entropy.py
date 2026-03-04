"""Table 4: Spearman correlation by entropy window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..data_loader import ENTROPY_WINDOWS
from .cli import run_table
from .common import stars, tex_escape, wrap_table, write_tex


def _spearman_rows(
    df: pd.DataFrame,
    methods: list[str],
    ecol: str,
    metric_cols: list[str],
) -> list[tuple[str, str, float, float]]:
    rows: list[tuple[str, str, float, float]] = []
    for method in methods:
        mdf = df[df["method"] == method]
        for mcol in metric_cols:
            valid = mdf[[ecol, mcol]].dropna()
            if len(valid) < 3:
                rows.append((method, mcol, np.nan, np.nan))
                continue
            rho, p = stats.spearmanr(valid[ecol], valid[mcol])
            rows.append((method, mcol, float(rho), float(p)))
    return rows


def _fdr_correct(rows: list[tuple[str, str, float, float]]) -> np.ndarray:
    pvals_arr = np.array([r[3] for r in rows if not np.isnan(r[3])])
    corrected = np.full(len(rows), np.nan)
    if pvals_arr.size == 0:
        return corrected
    _reject, p_corr, _, _ = multipletests(pvals_arr, method="fdr_bh")
    idx = 0
    for i, (_, _, _, p) in enumerate(rows):
        if not np.isnan(p):
            corrected[i] = p_corr[idx]
            idx += 1
    return corrected


def _build_spearman_body(
    rows: list[tuple[str, str, float, float]],
    corrected: np.ndarray,
    methods: list[str],
    metric_cols: list[str],
) -> list[str]:
    body: list[str] = []
    for method in methods:
        cells = [tex_escape(method)]
        for mcol in metric_cols:
            rec = next(r for r in rows if r[0] == method and r[1] == mcol)
            rho = rec[2]
            if np.isnan(rho):
                cells.append("--")
                continue
            corr_p = corrected[rows.index(rec)]
            if abs(rho) < 0.1:
                cells.append(r"$\approx 0$")
            else:
                cells.append(f"${rho:.3f}${stars(corr_p)}")
        body.append(" & ".join(cells) + r" \\")
    return body


def table_spearman_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Spearman rho between entropy and multiple metrics."""
    metric_cols = ["psnr", "ssim", "sam", "ergas"]
    metric_cols = [m for m in metric_cols if m in df.columns]

    if not metric_cols:
        return

    methods = sorted(df["method"].unique())

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        rows = _spearman_rows(df, methods, ecol, metric_cols)
        corrected = _fdr_correct(rows)
        body = _build_spearman_body(rows, corrected, methods, metric_cols)

        header = "Método & " + " & ".join(
            f"$\\rho$({{\\scriptsize {m.upper()}}})" for m in metric_cols
        )
        tex = wrap_table(
            body,
            caption=(
                f"Correlação de Spearman entre entropia "
                f"({ws}x{ws}) e métricas de qualidade (FDR). "
                r"$^{*}p_{FDR}<0{,}05$; $^{**}p_{FDR}<0{,}01$; "
                r"$^{***}p_{FDR}<0{,}001$."
            ),
            label=f"tab:spearman-entropy-e{ws}",
            col_spec="l" + "c" * len(metric_cols),
            header=header,
            env="table*",
            resizebox=True,
        )
        write_tex(tex, output_dir / f"tab4_spearman_entropy{ws}.tex")


def main() -> None:
    run_table(table_spearman_entropy, "Spearman entropy correlation")


if __name__ == "__main__":
    main()
