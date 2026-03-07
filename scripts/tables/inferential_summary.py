"""Compact inferential summary tables for the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdi_pipeline.statistics import method_comparison, robust_regression

from ..data_loader import display_method_name, filter_main_dl_scenario
from .cli import run_table
from .common import tex_escape, wrap_table, write_tex


def _format_p_value(value: float) -> str:
    if np.isnan(value):
        return "--"
    if value < 1e-300:
        return "$<10^{-300}$"
    if value < 1e-3:
        exp = int(np.floor(np.log10(value)))
        mantissa = value / (10**exp)
        return rf"${mantissa:.2f}\times 10^{{{exp}}}$"
    return f"${value:.3f}$"


def _format_ci(lo: float, hi: float) -> str:
    return rf"$[{lo:.3f}, {hi:.3f}]$"


def _metric_p_expr(symbol: str, value: float) -> str:
    formatted = _format_p_value(value)
    if formatted == "--":
        return f"${symbol}=--$"
    inner = formatted[1:-1]
    if inner.startswith("<"):
        return rf"${symbol}{inner}$"
    return rf"${symbol}={inner}$"


def _top_pairwise_summary(
    posthoc: pd.DataFrame,
    *,
    top_methods: list[str] | None = None,
) -> tuple[str, str]:
    if posthoc.empty:
        return "nenhuma comparação disponível", "--"

    subset = posthoc
    if top_methods is not None:
        subset = posthoc[
            posthoc["method_a"].isin(top_methods)
            & posthoc["method_b"].isin(top_methods)
        ]
        if subset.empty:
            subset = posthoc

    significant = subset[subset["significant"]]
    if significant.empty:
        return "nenhum contraste significativo após Bonferroni", "--"

    top = significant.reindex(
        significant["cliffs_delta"].abs().sort_values(ascending=False).index
    ).iloc[0]
    label = (
        f"{display_method_name(str(top['method_a']))} vs "
        f"{display_method_name(str(top['method_b']))}"
    )
    stats = f"$d={top['cliffs_delta']:.3f}$, " + _metric_p_expr(
        "p_{corr}", float(top["p_corrected"])
    )
    return label, stats


def _build_rows(df: pd.DataFrame, *, method_type: str) -> list[str]:
    comp = method_comparison(df, "psnr")
    means = (
        df
        .groupby("method", observed=True)["psnr"]
        .mean()
        .sort_values(ascending=False)
    )
    top_methods = list(means.head(5).index)
    contrast_label, contrast_stats = _top_pairwise_summary(
        comp.posthoc,
        top_methods=top_methods,
    )

    rows = [
        r"Kruskal-Wallis global (PSNR)"
        + " & "
        + (
            rf"$H={comp.statistic:.2f}$, "
            + _metric_p_expr("p", comp.p_value)
            + ", "
            rf"$\varepsilon^2={comp.epsilon_squared:.3f}$"
        )
        + r" \\",
        r"Comparações par a par (top-5 por PSNR)"
        + " & "
        + tex_escape(contrast_label)
        + r" \\",
        r"Maior efeito par a par" + " & " + contrast_stats + r" \\",
    ]

    ent_cols = sorted(c for c in df.columns if c.startswith("entropy_"))
    reg = robust_regression(df, "psnr", ent_cols)
    if (
        reg.n >= 10
        and not reg.coefficients.empty
        and np.isfinite(reg.r_squared_adj)
    ):
        rows.append(
            r"Regressão robusta (PSNR)"
            + " & "
            + rf"$n={reg.n}$, $R^2_{{adj}}={reg.r_squared_adj:.3f}$"
            + r" \\"
        )

        coef_df = reg.coefficients.set_index("variable")
        for entropy_col in ent_cols:
            if entropy_col not in coef_df.index:
                continue
            coef = coef_df.loc[entropy_col]
            rows.append(
                tex_escape(f"Coeficiente {entropy_col}")
                + " & "
                + (
                    rf"$\beta={coef['beta']:.3f}$, "
                    rf"IC95\%={
                        _format_ci(float(coef['ci_lo']), float(coef['ci_hi']))[
                            1:-1
                        ]
                    }, "
                    rf"$p={_format_p_value(float(coef['p_value']))[1:-1]}$"
                )
                + r" \\"
            )

    return rows


def table_inferential_summary(df: pd.DataFrame, output_dir: Path) -> None:
    for method_type, slug, caption_label in [
        ("Clássico", "classical", "clássicos"),
        ("DL", "dl", "de DL"),
    ]:
        type_df = df[df["type"] == method_type].copy()
        if type_df.empty:
            continue
        if method_type == "DL":
            type_df = filter_main_dl_scenario(type_df)
        if type_df.empty:
            continue

        body = _build_rows(type_df, method_type=method_type)
        tex = wrap_table(
            body,
            caption=(
                f"Síntese inferencial do bloco {caption_label}: "
                "teste global de diferenças em PSNR, contraste par a par "
                "mais forte entre os métodos líderes e regressão robusta "
                "com entropia, método e nível de ruído."
            ),
            label=f"tab:inferential-summary-{slug}",
            col_spec="p{0.42\\linewidth}p{0.50\\linewidth}",
            header="Análise & Resultado",
            env="table",
            resizebox=False,
        )
        write_tex(tex, output_dir / f"inferential-summary-{slug}.tex")


def main() -> None:
    run_table(table_inferential_summary, "Inferential summary")


if __name__ == "__main__":
    main()
