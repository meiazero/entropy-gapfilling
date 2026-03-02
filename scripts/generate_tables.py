"""Generate LaTeX tables from experiment results (data-driven rewrite).

Produces 5 table types with variations per noise level, entropy window,
and satellite. Each table is saved as a standalone .tex file.

Usage:
    uv run python scripts/generate_tables.py
    uv run python scripts/generate_tables.py --output docs/tables
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from data_loader import (
    ENTROPY_WINDOWS,
    NOISE_ORDER,
    entropy_terciles,
    load_combined,
    noise_label,
    select_top_n,
)
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── LaTeX helpers ─────────────────────────────────────────────────────


def _write_tex(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    log.info("Saved %s", path)


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _format_pm(mean: float, ci_half: float, fmt: str = ".2f") -> str:
    return f"${mean:{fmt}}_{{\\pm {ci_half:{fmt}}}}$"


def _bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _underline(text: str) -> str:
    return f"\\underline{{{text}}}"


def _ranked_cell(
    value: float, ci_half: float, rank: int, fmt: str = ".2f"
) -> str:
    base = _format_pm(value, ci_half, fmt)
    if rank == 1:
        return _bold(base)
    if rank == 2:
        return _underline(base)
    if rank == 3:
        return f"\\textit{{{base}}}"
    return base


def _ranked_plain(value: float, rank: int, fmt: str = ".3f") -> str:
    base = f"${value:{fmt}}$"
    if rank == 1:
        return _bold(base)
    if rank == 2:
        return _underline(base)
    if rank == 3:
        return f"\\textit{{{base}}}"
    return base


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _ci95_half(values: pd.Series) -> float:
    """Compute half-width of 95% CI using t-distribution."""
    n = len(values)
    if n < 2:
        return 0.0
    se = float(values.std() / np.sqrt(n))
    return float(stats.t.ppf(0.975, n - 1) * se)


def _wrap_table(
    body: list[str],
    *,
    caption: str,
    label: str,
    col_spec: str,
    header: str,
    font_size: str = r"\footnotesize",
    env: str = "table",
    resizebox: bool = False,
) -> str:
    tabular = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines = [
        rf"\begin{{{env}}}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        font_size,
    ]
    if resizebox:
        lines += [r"\resizebox{\linewidth}{!}{%", *tabular, r"}"]
    else:
        lines += tabular
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


# ── Table 1: Global Multi-Metric Scoreboard ──────────────────────────


def _compute_method_stats(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compute mean ± CI95 for each method x metric."""
    rows = []
    for method, grp in df.groupby("method", observed=True):
        row: dict[str, object] = {"method": method}
        if "type" in grp.columns:
            row["type"] = grp["type"].iloc[0]
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{m}_ci"] = _ci95_half(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def table1_global_scoreboard(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 1: PSNR, SSIM, RMSE, SAM, ERGAS per method (all + per noise)."""
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

        # Compute ranks per metric
        for m in metrics:
            col = f"{m}_mean"
            if col in stats_df.columns:
                stats_df[f"{m}_rank"] = stats_df[col].rank(
                    ascending=not higher_better[m], method="min"
                )

        # Sort by PSNR descending
        stats_df = stats_df.sort_values("psnr_mean", ascending=False)

        body: list[str] = []
        for _, row in stats_df.iterrows():
            method_str = _tex_escape(str(row["method"]))
            type_str = str(row.get("type", ""))
            cells = [type_str, method_str]
            for m in metrics:
                mean_val = row.get(f"{m}_mean", np.nan)
                ci_val = row.get(f"{m}_ci", 0.0)
                rank = int(row.get(f"{m}_rank", 99))
                if np.isnan(mean_val):
                    cells.append("--")
                else:
                    cells.append(_ranked_cell(mean_val, ci_val, rank))
            body.append(" & ".join(cells) + r" \\")

        header = (
            r"Tipo & Método & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ "
            r"& RMSE $\downarrow$ & SAM $\downarrow$ & ERGAS $\downarrow$"
        )
        tex = _wrap_table(
            body,
            caption=(
                f"Placar global multi-métrica ({caption_noise}). "
                r"\textbf{Negrito}: melhor; \underline{sublinhado}: 2º; "
                r"\textit{itálico}: 3º."
            ),
            label=f"tab:global-{suffix}",
            col_spec="llccccc",
            header=header,
            env="table*",
            resizebox=True,
        )
        _write_tex(tex, output_dir / f"tab1_global_{suffix}.tex")


# ── Table 2: Spectral Deconstruction (RMSE per band) ─────────────────


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
        for band in bands:
            if band in grp.columns:
                vals = grp[band].dropna()
                row[f"{band}_mean"] = (
                    float(vals.mean()) if len(vals) > 0 else np.nan
                )
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    # Rank per band (lower is better)
    for band in bands:
        col = f"{band}_mean"
        if col in stats_df.columns:
            stats_df[f"{band}_rank"] = stats_df[col].rank(
                ascending=True, method="min"
            )

    return stats_df.sort_values("rmse_b0_mean", ascending=True)


def _format_spectral_body(
    stats_df: pd.DataFrame,
    bands: list[str],
) -> list[str]:
    body: list[str] = []
    for _, row in stats_df.iterrows():
        cells = [str(row.get("type", "")), _tex_escape(str(row["method"]))]
        for band in bands:
            mean_val = row.get(f"{band}_mean", np.nan)
            rank = int(row.get(f"{band}_rank", 99))
            if np.isnan(mean_val):
                cells.append("--")
            else:
                cells.append(_ranked_plain(mean_val, rank, ".4f"))
        body.append(" & ".join(cells) + r" \\")
    return body


def table2_spectral(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 2: RMSE per spectral band for each method."""
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
        tex = _wrap_table(
            body,
            caption=(
                f"RMSE por banda espectral ({noise_label(noise)}). "
                r"Menor é melhor."
            ),
            label=f"tab:spectral-{suffix}",
            col_spec="llcccc",
            header=header,
            resizebox=True,
        )
        _write_tex(tex, output_dir / f"tab2_spectral_{suffix}.tex")


# ── Table 3: Degradation by Entropy x Noise ──────────────────────────


def table3_degradation(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 3: PSNR drop (%) from gap_only to 20dB, stratified by entropy."""
    # Select top methods
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl

    if not selected:
        log.warning("No methods available for table3")
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
                gap_only = mdf[mdf["noise_level"] == "inf"]["psnr"].mean()
                noisy_20 = mdf[mdf["noise_level"] == "20"]["psnr"].mean()

                if np.isnan(gap_only) or np.isnan(noisy_20) or gap_only == 0:
                    cells.append("--")
                else:
                    drop_pct = (gap_only - noisy_20) / gap_only * 100
                    cells.append(f"${drop_pct:.1f}\\%$")
            body.append(" & ".join(cells) + r" \\")

        header = "Faixa de Entropia & " + " & ".join(
            _tex_escape(m) for m in selected
        )
        tex = _wrap_table(
            body,
            caption=(
                f"Queda percentual no PSNR (sem ruído → 20 dB) por faixa "
                f"de entropia (janela {ws}x{ws}). "
                f"Top-3 clássicos + top-3 DL."
            ),
            label=f"tab:degradation-e{ws}",
            col_spec="l" + "c" * len(selected),
            header=header,
            resizebox=True,
        )
        _write_tex(tex, output_dir / f"tab3_degradation_entropy{ws}.tex")


# ── Table 4: Spearman Correlation ─────────────────────────────────────


def table4_correlation(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 4: Spearman rho between entropy and multiple metrics."""
    metric_cols = ["psnr", "ssim", "sam", "ergas"]
    metric_cols = [m for m in metric_cols if m in df.columns]

    if not metric_cols:
        log.warning("No metric columns for table4")
        return

    methods = sorted(df["method"].unique())

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        body: list[str] = []
        for method in methods:
            mdf = df[df["method"] == method]
            cells = [_tex_escape(method)]
            for mcol in metric_cols:
                valid = mdf[[ecol, mcol]].dropna()
                if len(valid) < 3:
                    cells.append("--")
                    continue
                rho, p = stats.spearmanr(valid[ecol], valid[mcol])
                cells.append(f"${rho:.3f}${_stars(p)}")
            body.append(" & ".join(cells) + r" \\")

        header = "Método & " + " & ".join(
            f"$\\rho$({{\\scriptsize {m.upper()}}})" for m in metric_cols
        )
        tex = _wrap_table(
            body,
            caption=(
                f"Correlação de Spearman entre entropia "
                f"({ws}x{ws}) e métricas de qualidade. "
                r"$^{*}p<0{,}05$; $^{**}p<0{,}01$; $^{***}p<0{,}001$."
            ),
            label=f"tab:spearman-e{ws}",
            col_spec="l" + "c" * len(metric_cols),
            header=header,
            env="table*",
            resizebox=True,
        )
        _write_tex(tex, output_dir / f"tab4_spearman_entropy{ws}.tex")


# ── Table 5: Speed and Practical Viability ────────────────────────────


def table5_speed(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 5: PSNR vs inference time per method."""
    # Use gap_only for timing (noise doesn't affect elapsed_s)
    subset = df[df["noise_level"] == "inf"]
    if subset.empty:
        subset = df

    if "elapsed_s" not in subset.columns:
        log.warning("No elapsed_s column for table5")
        return

    rows = []
    for method, grp in subset.groupby("method", observed=True):
        psnr_mean = (
            float(grp["psnr"].mean()) if "psnr" in grp.columns else np.nan
        )
        time_mean = float(grp["elapsed_s"].mean())
        efficiency = psnr_mean / time_mean if time_mean > 0 else np.nan
        rows.append({
            "method": method,
            "type": grp["type"].iloc[0] if "type" in grp.columns else "",
            "psnr": psnr_mean,
            "time": time_mean,
            "efficiency": efficiency,
        })

    stats_df = pd.DataFrame(rows)

    # Rank: PSNR higher better, time lower better, efficiency higher better
    stats_df["psnr_rank"] = stats_df["psnr"].rank(ascending=False, method="min")
    stats_df["time_rank"] = stats_df["time"].rank(ascending=True, method="min")
    stats_df["eff_rank"] = stats_df["efficiency"].rank(
        ascending=False, method="min"
    )

    stats_df = stats_df.sort_values("efficiency", ascending=False)

    body: list[str] = []
    for _, row in stats_df.iterrows():
        psnr_cell = _ranked_plain(row["psnr"], int(row["psnr_rank"]), ".2f")
        if row["time"] < 0.01:
            time_str = f"${row['time']:.4f}$"
        elif row["time"] < 1:
            time_str = f"${row['time']:.3f}$"
        else:
            time_str = f"${row['time']:.2f}$"
        _ranked_plain(row["time"], int(row["time_rank"]), ".3f")
        eff_cell = _ranked_plain(row["efficiency"], int(row["eff_rank"]), ".1f")
        body.append(
            f"{row['type']} & {_tex_escape(str(row['method']))} & "
            f"{psnr_cell} & {time_str} & {eff_cell} \\\\"
        )

    tex = _wrap_table(
        body,
        caption=(
            "Velocidade e viabilidade prática (sem ruído). "
            "PSNR/s indica eficiência."
        ),
        label="tab:speed",
        col_spec="llccc",
        header=(
            r"Tipo & Método & PSNR (dB) $\uparrow$ "
            r"& Tempo (s/patch) $\downarrow$ & PSNR/s $\uparrow$"
        ),
        resizebox=True,
    )
    _write_tex(tex, output_dir / "tab5_speed.tex")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to paper_assets/tables/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    df = load_combined()
    if df.empty:
        log.error("No data loaded. Check paper_assets/ paths.")
        return

    output_dir = args.output or Path("paper_assets/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("Table 1: Global Scoreboard", table1_global_scoreboard),
        ("Table 2: Spectral Deconstruction", table2_spectral),
        ("Table 3: Degradation by Entropy", table3_degradation),
        ("Table 4: Spearman Correlation", table4_correlation),
        ("Table 5: Speed", table5_speed),
    ]

    for name, func in generators:
        try:
            log.info("Generating %s...", name)
            func(df, output_dir)
        except Exception:
            log.exception("Error generating %s", name)

    log.info("All tables saved to: %s", output_dir)


if __name__ == "__main__":
    main()
