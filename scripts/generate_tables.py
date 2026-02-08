"""Generate LaTeX tables from experiment results.

Produces 5 table types for the journal paper, each exported as a
standalone .tex file that can be included via \\input{}.

Usage:
    uv run python scripts/generate_tables.py --results results/paper_results
    uv run python scripts/generate_tables.py --results results/paper_results --table 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pdi_pipeline.aggregation import (
    load_results,
    summary_by_entropy_bin,
    summary_by_noise,
)
from pdi_pipeline.statistics import correlation_matrix, method_comparison

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


def table1_method_overview(output_dir: Path) -> None:
    """Table 1: Method overview with category, parameters, complexity."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Overview of the 15 classical gap-filling methods evaluated.}",
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
        for rank, (_, rr) in enumerate(ranked.iterrows(), 1):
            rankings.setdefault(noise, {})[rr["method"]] = rank

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
    """Table 3: Mean PSNR per method x entropy bin."""
    df = load_results(results_dir)
    ent_summary = summary_by_entropy_bin(
        df, entropy_col="entropy_7", metric="psnr"
    )

    if ent_summary.empty:
        log.warning("No data for table3")
        return

    methods = sorted(ent_summary["method"].unique())
    bins = ["low", "medium", "high"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean PSNR (dB) $\pm$ 95\% CI stratified by entropy tercile (7$\times$7 window).}",
        r"\label{tab:entropy_stratified}",
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
        for rank, (_, rr) in enumerate(ranked.iterrows(), 1):
            bin_rankings.setdefault(b, {})[rr["method"]] = rank

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

    _write_tex("\n".join(lines), output_dir / "table3_entropy.tex")


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
        r"\caption{Spearman correlation between local entropy and metrics. Significance: * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$ (FDR-corrected).}",
        r"\label{tab:correlation}",
        r"\tiny",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        mdf = corr_df[corr_df["method"] == method]
        cells = [method]
        for ecol in entropy_cols:
            for mcol in metric_cols:
                row = mdf[
                    (mdf["entropy_col"] == ecol) & (mdf["metric_col"] == mcol)
                ]
                if row.empty:
                    cells.append("--")
                else:
                    r = row.iloc[0]
                    rho = r["spearman_rho"]
                    p = r["spearman_p"]
                    stars = ""
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                    cells.append(f"${rho:.3f}${stars}")
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
        r"\caption{Kruskal-Wallis test and significant pairwise differences (Dunn post-hoc, Bonferroni correction).}",
        r"\label{tab:kruskal}",
        r"\footnotesize",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & H statistic & $p$-value & $\epsilon^2$ & Significant pairs \\",
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


ALL_TABLES = {
    1: lambda rd, od: table1_method_overview(od),
    2: table2_overall_results,
    3: table3_entropy_stratified,
    4: table4_correlation,
    5: table5_kruskal_wallis,
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
        help="Generate only this table number (1-5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    results_dir = args.results
    output_dir = args.output or results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

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
