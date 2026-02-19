"""Generate LaTeX tables from experiment results.

Produces 8 table types for the journal paper, each exported as a
standalone .tex file that can be included via \\input{}.

Usage:
    uv run python scripts/generate_tables.py --results results/paper_results
    uv run python scripts/generate_tables.py \
        --results results/paper_results --table 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_pipeline.aggregation import (
    load_results,
    summary_by_entropy_bin,
    summary_by_noise,
    summary_by_satellite,
)
from pdi_pipeline.statistics import (
    correlation_matrix,
    method_comparison,
    robust_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Method metadata for Table 1
METHOD_INFO = {
    "nearest": ("Spatial", "Nearest Neighbor", "None", "O(HW)"),
    "bilinear": ("Spatial", "Bilinear", "None", "O(N log N)"),
    "bicubic": ("Spatial", "Bicubic", "None", "O(N log N)"),
    "lanczos": ("Spatial", "Lanczos (PG)", "a=3", "O(HW log HW)"),
    "idw": ("Kernel", "IDW", "p=2.0", "O(NK)"),
    "rbf": ("Kernel", "RBF", "TPS kernel", "O(N^3)"),
    "spline": ("Kernel", "Thin Plate Spline", "None", "O(N^3)"),
    "kriging": (
        "Geostatistical",
        "Ordinary Kriging",
        "Auto variogram",
        "O(N^3)",
    ),
    "dct": ("Transform", "DCT", "iter=50, lam=0.05", "O(HW log HW)"),
    "wavelet": ("Transform", "Wavelet", "db4, iter=50", "O(HW)"),
    "tv": ("Transform", "Total Variation", "iter=100", "O(HW)"),
    "cs_dct": ("Compressive", "L1-DCT (CS)", "iter=100", "O(HW log HW)"),
    "cs_wavelet": ("Compressive", "L1-Wavelet (CS)", "iter=100", "O(HW)"),
    "non_local": (
        "Patch-based",
        "Non-Local Means",
        "h=0.1, p=7, s=21",
        "O(HW P^2)",
    ),
    "exemplar_based": ("Patch-based", "Exemplar-Based", "p=9", "O(HW P^2)"),
}


def _format_ranked_cell(value: float, ci_half: float, rank: int) -> str:
    """Format a metric cell with bold-best / underline-second styling.

    Args:
        value: Mean metric value.
        ci_half: Half-width of the 95% CI.
        rank: 1 = best, 2 = second-best, else plain.

    Returns:
        LaTeX-formatted string.
    """
    base = f"${value:.2f}_{{\\pm {ci_half:.2f}}}$"
    if rank == 1:
        return f"\\textbf{{{base}}}"
    if rank == 2:
        return f"\\underline{{{base}}}"
    return base


def _write_tex(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    log.info("Saved %s", path)


def _stars_for_p_value(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_table4_cell(
    method_df: pd.DataFrame,
    entropy_col: str,
    metric_col: str,
) -> str:
    row = method_df[
        (method_df["entropy_col"] == entropy_col)
        & (method_df["metric_col"] == metric_col)
    ]
    if row.empty:
        return "--"

    rr = row.iloc[0]
    rho = float(rr["spearman_rho"])
    p_col = (
        "spearman_p_corrected"
        if "spearman_p_corrected" in rr.index
        else "spearman_p"
    )
    stars = _stars_for_p_value(float(rr[p_col]))
    return f"${rho:.3f}${stars}"


def _collect_dl_rows(dl_base: Path) -> list[dict[str, object]]:
    dl_rows: list[dict[str, object]] = []
    if not dl_base.exists():
        return dl_rows

    for model_dir in sorted(dl_base.iterdir()):
        csv_path = model_dir / "results.csv"
        if not csv_path.exists():
            continue

        dl_df = pd.read_csv(csv_path)
        ok = dl_df[dl_df["status"] == "ok"]
        if ok.empty:
            continue

        dl_rows.append({
            "method": model_dir.name,
            "psnr": ok["psnr"].mean(),
            "ssim": ok["ssim"].mean(),
            "rmse": ok["rmse"].mean(),
            "type": "Deep Learning",
        })
    return dl_rows


def _add_metric_rankings(df: pd.DataFrame, metrics: list[str]) -> None:
    for metric_col in metrics:
        df[f"rank_{metric_col}"] = df[metric_col].rank(
            ascending=(metric_col == "rmse"), method="min"
        )


def _format_table8_metric(value: float, rank: int) -> str:
    if rank == 1:
        return f"\\textbf{{{value:.3f}}}"
    if rank == 2:
        return f"\\underline{{{value:.3f}}}"
    return f"${value:.3f}$"


def table1_method_overview(output_dir: Path) -> None:
    """Table 1: Method overview with category, parameters, complexity."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Overview of the 15 classical gap-filling methods "
            r"evaluated.}"
        ),
        r"\label{tab:methods}",
        r"\footnotesize",
        r"\begin{tabular}{llllc}",
        r"\toprule",
        r"Category & Method & Parameters & Complexity \\",
        r"\midrule",
    ]

    prev_cat = ""
    for _name, (cat, display, params, complexity) in METHOD_INFO.items():
        cat_str = cat if cat != prev_cat else ""
        prev_cat = cat
        lines.append(f"{cat_str} & {display} & {params} & {complexity} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table1_methods.tex")


def table2_overall_results(results_dir: Path, output_dir: Path) -> None:
    """Table 2: Mean PSNR +/- CI95% per method x noise level."""
    df = load_results(results_dir)
    noise_summary = summary_by_noise(df, metric="psnr")

    methods = sorted(noise_summary["method"].unique())
    noise_levels = ["inf", "40", "30", "20"]
    present_levels = [
        n for n in noise_levels if n in noise_summary["noise_level"].values
    ]

    ncols = len(present_levels)
    col_spec = "l" + "c" * ncols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean PSNR (dB) $\pm$ 95\% CI per method and noise level.}",
        r"\label{tab:overall}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Method & "
        + " & ".join(
            f"{n} dB" if n != "inf" else "No noise" for n in present_levels
        )
        + r" \\",
        r"\midrule",
    ]

    # Compute rankings per noise level
    rankings: dict[str, dict[str, int]] = {}
    for noise in present_levels:
        ndf = noise_summary[noise_summary["noise_level"] == noise]
        ranked = ndf.sort_values("mean", ascending=False)
        for rank, rr in enumerate(ranked.itertuples(), 1):
            rankings.setdefault(noise, {})[rr.method] = rank

    for method in methods:
        mdf = noise_summary[noise_summary["method"] == method]
        cells = [method]
        for noise in present_levels:
            row = mdf[mdf["noise_level"] == noise]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                ci_half = (r["ci95_hi"] - r["ci95_lo"]) / 2
                rank = rankings.get(noise, {}).get(method, 99)
                cells.append(_format_ranked_cell(r["mean"], ci_half, rank))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table2_overall.tex")


def table3_entropy_stratified(results_dir: Path, output_dir: Path) -> None:
    """Table 3: Mean PSNR per method x entropy bin at all window scales."""
    df = load_results(results_dir)

    entropy_windows = [c for c in df.columns if c.startswith("entropy_")]
    if not entropy_windows:
        entropy_windows = ["entropy_7"]

    bins = ["low", "medium", "high"]

    for ecol in sorted(entropy_windows):
        ws = ecol.split("_")[-1]
        ent_summary = summary_by_entropy_bin(
            df, entropy_col=ecol, metric="psnr"
        )

        if ent_summary.empty:
            log.warning("No data for table3 (%s)", ecol)
            continue

        methods = sorted(ent_summary["method"].unique())

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            (
                rf"\caption{{Mean PSNR (dB) $\pm$ 95\% CI stratified "
                rf"by entropy tercile ({ws}$\times${ws} window).}}"
            ),
            rf"\label{{tab:entropy_stratified_{ws}}}",
            r"\footnotesize",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & Low Entropy & Medium Entropy & High Entropy \\",
            r"\midrule",
        ]

        # Compute rankings per entropy bin
        bin_rankings: dict[str, dict[str, int]] = {}
        for b in bins:
            bdf = ent_summary[ent_summary["entropy_bin"] == b]
            ranked = bdf.sort_values("mean", ascending=False)
            for rank, rr in enumerate(ranked.itertuples(), 1):
                bin_rankings.setdefault(b, {})[rr.method] = rank

        for method in methods:
            mdf = ent_summary[ent_summary["method"] == method]
            cells = [method]
            for b in bins:
                row = mdf[mdf["entropy_bin"] == b]
                if row.empty:
                    cells.append("--")
                else:
                    r = row.iloc[0]
                    ci_half = (r["ci95_hi"] - r["ci95_lo"]) / 2
                    rank = bin_rankings.get(b, {}).get(method, 99)
                    cells.append(_format_ranked_cell(r["mean"], ci_half, rank))
            lines.append(" & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        _write_tex(
            "\n".join(lines),
            output_dir / f"table3_entropy_{ws}.tex",
        )


def table4_correlation(results_dir: Path, output_dir: Path) -> None:
    """Table 4: Spearman rho for entropy x metric with significance stars."""
    df = load_results(results_dir)

    entropy_cols = [c for c in df.columns if c.startswith("entropy_")]
    metric_cols = ["psnr", "ssim", "rmse", "sam"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    if not entropy_cols or not metric_cols:
        log.warning("Missing columns for table4")
        return

    corr_df = correlation_matrix(df, entropy_cols, metric_cols)
    if corr_df.empty:
        return

    methods = sorted(corr_df["method"].unique())

    ncols = len(entropy_cols) * len(metric_cols)
    col_spec = "l" + "c" * ncols

    # Build compound header
    header_parts = ["Method"]
    for ecol in entropy_cols:
        ws = ecol.split("_")[-1]
        for mcol in metric_cols:
            header_parts.append(f"$\\rho_{{H_{{{ws}}}}}$({mcol.upper()})")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spearman correlation between entropy and quality metrics.}",
        r"\label{tab:correlation}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        mdf = corr_df[corr_df["method"] == method]
        cells = [method]
        for ecol in entropy_cols:
            for mcol in metric_cols:
                cells.append(_format_table4_cell(mdf, ecol, mcol))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table4_correlation.tex")


def table5_kruskal_wallis(results_dir: Path, output_dir: Path) -> None:
    """Table 5: Kruskal-Wallis + Dunn post-hoc summary."""
    df = load_results(results_dir)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Kruskal-Wallis test and significant pairwise "
            r"differences (Dunn post-hoc, Bonferroni correction).}"
        ),
        r"\label{tab:kruskal}",
        r"\footnotesize",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            r"Metric & H statistic & $p$-value & $\epsilon^2$ "
            r"& Significant pairs \\",
        ),
        r"\midrule",
    ]

    for metric in ["psnr", "ssim", "rmse", "sam"]:
        if metric not in df.columns:
            continue
        result = method_comparison(df, metric_col=metric)
        n_sig = 0
        if not result.posthoc.empty:
            n_sig = int(result.posthoc["significant"].sum())
        p_str = (
            "$< 10^{-10}$"
            if result.p_value < 1e-10
            else f"${result.p_value:.2e}$"
        )
        lines.append(
            f"{metric.upper()} & ${result.statistic:.1f}$ & "
            f"{p_str} & ${result.epsilon_squared:.4f}$ & {n_sig} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table5_kruskal.tex")


def table6_regression(results_dir: Path, output_dir: Path) -> None:
    """Table 6: Robust regression coefficients, p-values, R2-adj, VIF."""
    df = load_results(results_dir)

    entropy_cols = sorted(c for c in df.columns if c.startswith("entropy_"))
    if not entropy_cols:
        log.warning("No entropy columns for table6")
        return

    for metric in ["psnr", "ssim", "rmse"]:
        if metric not in df.columns:
            continue

        result = robust_regression(
            df, metric_col=metric, entropy_cols=entropy_cols
        )
        if result.coefficients.empty:
            log.warning("Regression failed for %s", metric)
            continue

        coef = result.coefficients

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Robust regression (RLM/HuberT) for {metric.upper()}. "
            rf"$R^2_{{adj}} = {result.r_squared_adj:.4f}$, $n = {result.n}$.}}",
            rf"\label{{tab:regression_{metric}}}",
            r"\tiny",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Variable & $\beta$ & Std. Err. & $z$ & $p$ & 95\% CI \\",
            r"\midrule",
        ]

        for row in coef.itertuples():
            var = str(row.variable).replace("_", r"\_")
            p_str = (
                "$< 10^{-10}$"
                if row.p_value < 1e-10
                else f"${row.p_value:.2e}$"
            )
            ci_str = f"[{row.ci_lo:.4f}, {row.ci_hi:.4f}]"
            lines.append(
                f"{var} & ${row.beta:.4f}$ & "
                f"${row.std_err:.4f}$ & "
                f"${row.z_value:.2f}$ & "
                f"{p_str} & {ci_str} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
        ])

        # VIF sub-table
        if not result.vif.empty:
            lines.append(r"\vspace{0.5em}")
            lines.append(r"\begin{tabular}{lr}")
            lines.append(r"\toprule")
            lines.append(r"Variable & VIF \\")
            lines.append(r"\midrule")
            for vrow in result.vif.itertuples():
                var = str(vrow.variable).replace("_", r"\_")
                lines.append(f"{var} & ${vrow.vif:.2f}$ \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")

        lines.append(r"\end{table}")

        _write_tex(
            "\n".join(lines),
            output_dir / f"table6_regression_{metric}.tex",
        )


def table7_satellite(results_dir: Path, output_dir: Path) -> None:
    """Table 7: Mean PSNR per method x satellite."""
    df = load_results(results_dir)
    sat_summary = summary_by_satellite(df, metric="psnr")

    if sat_summary.empty:
        log.warning("No data for table7")
        return

    methods = sorted(sat_summary["method"].unique())
    satellites = sorted(sat_summary["satellite"].unique())

    ncols = len(satellites)
    col_spec = "l" + "c" * ncols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Mean PSNR (dB) $\pm$ 95\% CI per method "
            r"and satellite sensor.}"
        ),
        r"\label{tab:satellite}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Method & "
        + " & ".join(s.replace("_", r"\_") for s in satellites)
        + r" \\",
        r"\midrule",
    ]

    # Rankings per satellite
    sat_rankings: dict[str, dict[str, int]] = {}
    for sat in satellites:
        sdf = sat_summary[sat_summary["satellite"] == sat]
        ranked = sdf.sort_values("mean", ascending=False)
        for rank, rr in enumerate(ranked.itertuples(), 1):
            sat_rankings.setdefault(sat, {})[rr.method] = rank

    for method in methods:
        mdf = sat_summary[sat_summary["method"] == method]
        cells = [method]
        for sat in satellites:
            row = mdf[mdf["satellite"] == sat]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                ci_half = (r["ci95_hi"] - r["ci95_lo"]) / 2
                rank = sat_rankings.get(sat, {}).get(method, 99)
                cells.append(_format_ranked_cell(r["mean"], ci_half, rank))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table7_satellite.tex")


def table8_dl_comparison(results_dir: Path, output_dir: Path) -> None:
    """Table 8: Classical vs. DL method comparison.

    Loads DL evaluation results from results_dir/../dl_eval/ and merges
    with classical results for a side-by-side comparison.
    """
    import pandas as pd

    df_classical = load_results(results_dir)

    # Aggregate classical: mean per method
    classical_summary = (
        df_classical.groupby("method")[["psnr", "ssim", "rmse"]]
        .mean()
        .reset_index()
    )
    classical_summary["type"] = "Classical"

    # Load DL results
    dl_base = results_dir.parent / "dl_eval"
    dl_rows = _collect_dl_rows(dl_base)

    if not dl_rows:
        log.warning(
            "No DL results found at %s. "
            "Run DL evaluation first to include in comparison.",
            dl_base,
        )

    dl_summary = pd.DataFrame(dl_rows)
    combined = pd.concat([classical_summary, dl_summary], ignore_index=True)
    combined = combined.sort_values("psnr", ascending=False)

    metrics_display = ["psnr", "ssim", "rmse"]
    present = [m for m in metrics_display if m in combined.columns]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Performance comparison: classical interpolation "
            r"vs.\ deep learning methods.}"
        ),
        r"\label{tab:dl_comparison}",
        r"\footnotesize",
        r"\begin{tabular}{ll" + "c" * len(present) + "}",
        r"\toprule",
        "Type & Method & " + " & ".join(m.upper() for m in present) + r" \\",
        r"\midrule",
    ]

    # Rankings for bold/underline
    _add_metric_rankings(combined, present)

    prev_type = ""
    for row in combined.itertuples():
        row_type = str(row.type)
        type_str = row_type if row_type != prev_type else ""
        prev_type = row_type
        method_str = str(row.method).replace("_", r"\_")
        cells = [type_str, method_str]
        for m_col in present:
            val = float(getattr(row, m_col))
            rank = int(getattr(row, f"rank_{m_col}"))
            cells.append(_format_table8_metric(val, rank))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_tex("\n".join(lines), output_dir / "table8_dl_comparison.tex")


ALL_TABLES = {
    1: lambda rd, od: table1_method_overview(od),
    2: table2_overall_results,
    3: table3_entropy_stratified,
    4: table4_correlation,
    5: table5_kruskal_wallis,
    6: table6_regression,
    7: table7_satellite,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment results.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to experiment results directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for tables. Default: results_dir/tables/",
    )
    parser.add_argument(
        "--table",
        type=int,
        default=None,
        help="Generate only this table number (1-7).",
    )
    return parser.parse_args(argv)


def _setup_file_logging(log_path: Path) -> None:
    """Add a file handler to the root logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    results_dir = args.results
    output_dir = args.output or results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(results_dir / "tables.log")

    if args.table is not None:
        if args.table not in ALL_TABLES:
            log.error("Invalid table number: %d", args.table)
            return
        ALL_TABLES[args.table](results_dir, output_dir)
    else:
        for num, func in ALL_TABLES.items():
            try:
                func(results_dir, output_dir)
            except Exception:
                log.exception("Error generating table %d", num)

    log.info("Tables saved to: %s", output_dir)


if __name__ == "__main__":
    main()
