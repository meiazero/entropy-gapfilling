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
from statsmodels.stats.multitest import multipletests

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


def _bootstrap_ci_half(
    values: pd.Series,
    clusters: pd.Series | None,
    *,
    stat_fn: callable,
    n_boot: int = 1000,
    seed: int = 42,
) -> float:
    vals = values.dropna()
    if len(vals) < 2:
        return 0.0
    if clusters is None or clusters.isna().all():
        return _ci95_half(vals)

    data = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    unique_clusters = data["cluster"].unique().tolist()
    if len(unique_clusters) < 2:
        return _ci95_half(vals)

    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_boot):
        sampled = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        sample = data[data["cluster"].isin(sampled)]["value"]
        stat = stat_fn(sample)
        if not np.isnan(stat):
            boot_stats.append(float(stat))
    if not boot_stats:
        return 0.0
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return float((hi - lo) / 2.0)


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
    *,
    cluster_col: str | None = "patch_id",
) -> pd.DataFrame:
    """Compute mean ± CI95 (cluster bootstrap) for each method x metric."""
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
            row[f"{m}_ci"] = _bootstrap_ci_half(
                vals,
                clusters,
                stat_fn=np.mean,
            )
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
            label=f"tab:global-scoreboard-{suffix}",
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
        clusters = grp["patch_id"] if "patch_id" in grp.columns else None
        for band in bands:
            if band in grp.columns:
                vals = grp[band].dropna()
                row[f"{band}_median"] = (
                    float(vals.median()) if len(vals) > 0 else np.nan
                )
                row[f"{band}_ci"] = _bootstrap_ci_half(
                    vals,
                    clusters,
                    stat_fn=np.median,
                )
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    # Rank per band (lower is better)
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
        cells = [str(row.get("type", "")), _tex_escape(str(row["method"]))]
        for band in bands:
            med_val = row.get(f"{band}_median", np.nan)
            ci_val = row.get(f"{band}_ci", 0.0)
            rank = int(row.get(f"{band}_rank", 99))
            if np.isnan(med_val):
                cells.append("--")
            else:
                cells.append(_ranked_cell(med_val, ci_val, rank, ".4f"))
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
                rf"RMSE mediano por banda (IC95\%) ({noise_label(noise)}). "
                r"Menor é melhor."
            ),
            label=f"tab:spectral-rmse-{suffix}",
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

                    ci = _bootstrap_ci_half(
                        mdf["psnr"],
                        clusters,
                        stat_fn=_drop_stat,
                    )
                    cells.append(f"${drop_pct:.1f}\\%_{{\\pm {ci:.1f}}}$")
            body.append(" & ".join(cells) + r" \\")

        header = "Faixa de Entropia & " + " & ".join(
            _tex_escape(m) for m in selected
        )
        tex = _wrap_table(
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
        _write_tex(tex, output_dir / f"tab3_degradation_entropy{ws}.tex")


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
    ci = _bootstrap_ci_half(
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


def table3_noise_slope(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Table 3b: PSNR slope vs noise level (20/30/40 dB) by entropy bin."""
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
            _tex_escape(m) for m in selected
        )
        tex = _wrap_table(
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
        _write_tex(tex, output_dir / f"tab3_slope_entropy{ws}.tex")


# ── Helpers: Spearman table ──────────────────────────────────────────


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
        cells = [_tex_escape(method)]
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
                cells.append(f"${rho:.3f}${_stars(corr_p)}")
        body.append(" & ".join(cells) + r" \\")
    return body


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

        rows = _spearman_rows(df, methods, ecol, metric_cols)
        corrected = _fdr_correct(rows)
        body = _build_spearman_body(rows, corrected, methods, metric_cols)

        header = "Método & " + " & ".join(
            f"$\\rho$({{\\scriptsize {m.upper()}}})" for m in metric_cols
        )
        tex = _wrap_table(
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
            "PSNR/s indica eficiência. "
            "Hardware: Rocky Linux 9.5, NVIDIA A100 80GB PCIe, "
            "driver 550.90.07, CUDA 12.4/12.6.2."
        ),
        label="tab:runtime-speed",
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
        ("Table 3b: Noise Slope", table3_noise_slope),
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
