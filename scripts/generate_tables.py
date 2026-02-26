"""Generate LaTeX tables from experiment results.

Produces table types for the journal paper, each exported as a
standalone .tex file that can be included via \\input{}.

Usage:
    uv run python scripts/generate_tables.py --results results/paper_results
    uv run python scripts/generate_tables.py \
        --results results/paper_results --table 2
    uv run python scripts/generate_tables.py \
        --dl-results dl_models
    uv run python scripts/generate_tables.py \
        --results results/paper_results --dl-results dl_models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "nearest": ("Espacial", "Vizinho Mais Próximo", "None", r"$O(HW)$"),
    "bilinear": ("Espacial", "Bilinear", "None", r"$O(N \log N)$"),
    "bicubic": ("Espacial", "Bicúbico", "None", r"$O(N \log N)$"),
    "lanczos": (
        "Espacial",
        "Lanczos (PG)",
        "a=3",
        r"$O(HW \log HW)$",
    ),
    "idw": ("Kernel", "IDW", "p=2.0", r"$O(NK)$"),
    "rbf": ("Kernel", "RBF", "kernel TPS", r"$O(N^3)$"),
    "spline": ("Kernel", "Spline de Placa Fina", "None", r"$O(N^3)$"),
    "kriging": (
        "Geoestatístico",
        "Krigagem Ordinária",
        "Variograma automático",
        r"$O(N^3)$",
    ),
    "dct": (
        "Transformada",
        "DCT",
        "iter=50, lam=0.05",
        r"$O(HW \log HW)$",
    ),
    "wavelet": ("Transformada", "Wavelet", "db4, iter=50", r"$O(HW)$"),
    "tv": ("Transformada", "Variação Total", "iter=100", r"$O(HW)$"),
    "cs_dct": (
        "Compressivo",
        "L1-DCT (CS)",
        "iter=100",
        r"$O(HW \log HW)$",
    ),
    "cs_wavelet": (
        "Compressivo",
        "L1-Wavelet (CS)",
        "iter=100",
        r"$O(HW)$",
    ),
    "non_local": (
        "Baseado em Recortes",
        "Médias Não-Locais",
        "h=0.1, p=7, s=21",
        r"$O(HW P^2)$",
    ),
    "exemplar_based": (
        "Baseado em Recortes",
        "Baseado em Exemplar",
        "p=9",
        r"$O(HW P^2)$",
    ),
}


# ------------------------------------------------------------------
# LaTeX table boilerplate
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LatexTableConfig:
    """Specification for a LaTeX ``tabular`` environment."""

    caption: str
    label: str
    col_spec: str
    header: str
    font_size: str = r"\footnotesize"
    env: str = "table"
    resizebox: bool = False


def _render_latex_table(
    config: LatexTableConfig,
    body_lines: list[str],
    extra_after_tabular: list[str] | None = None,
) -> str:
    """Render a complete LaTeX table from *config* and body rows."""
    tabular_lines = [
        rf"\begin{{tabular}}{{{config.col_spec}}}",
        r"\toprule",
        config.header + r" \\",
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    lines = [
        rf"\begin{{{config.env}}}[htbp]",
        r"\centering",
        rf"\caption{{{config.caption}}}",
        rf"\label{{{config.label}}}",
        config.font_size,
    ]
    if config.resizebox:
        lines += [r"\resizebox{\linewidth}{!}{%", *tabular_lines, r"}"]
    else:
        lines += tabular_lines
    if extra_after_tabular:
        lines.extend(extra_after_tabular)
    lines.append(rf"\end{{{config.env}}}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Shared formatting helpers
# ------------------------------------------------------------------


def _format_ranked_cell(value: float, ci_half: float, rank: int) -> str:
    """Format a metric cell with bold-best / underline-second."""
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


def _format_table8_metric(value: float, rank: int) -> str:
    if rank == 1:
        return f"\\textbf{{{value:.3f}}}"
    if rank == 2:
        return f"\\underline{{{value:.3f}}}"
    return f"${value:.3f}$"


# ------------------------------------------------------------------
# Ranking + row generation (shared by tables 2, 3, 7)
# ------------------------------------------------------------------


def _compute_group_rankings(
    summary: pd.DataFrame,
    group_col: str,
    groups: list[str],
) -> dict[str, dict[str, int]]:
    """Rank methods within each *group_col* value by descending mean."""
    rankings: dict[str, dict[str, int]] = {}
    for group_val in groups:
        gdf = summary[summary[group_col] == group_val]
        ranked = gdf.sort_values("mean", ascending=False)
        for rank, rr in enumerate(ranked.itertuples(), 1):
            rankings.setdefault(group_val, {})[rr.method] = rank
    return rankings


def _build_ranked_rows(
    summary: pd.DataFrame,
    group_col: str,
    groups: list[str],
    methods: list[str],
    rankings: dict[str, dict[str, int]],
) -> list[str]:
    """Build LaTeX row strings for a ranked method x group table."""
    rows: list[str] = []
    for method in methods:
        mdf = summary[summary["method"] == method]
        cells: list[str] = [method]
        for g in groups:
            row = mdf[mdf[group_col] == g]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                ci_half = (r["ci95_hi"] - r["ci95_lo"]) / 2
                rank = rankings.get(g, {}).get(method, 99)
                cells.append(_format_ranked_cell(r["mean"], ci_half, rank))
        rows.append(" & ".join(cells) + r" \\")
    return rows


# ------------------------------------------------------------------
# O(1) correlation cell lookup (table 4)
# ------------------------------------------------------------------


def _build_correlation_lookup(
    corr_df: pd.DataFrame,
) -> dict[tuple[str, str, str], Any]:
    """Index *corr_df* for O(1) cell access."""
    lookup: dict[tuple[str, str, str], Any] = {}
    for row in corr_df.itertuples():
        key = (row.method, row.entropy_col, row.metric_col)
        lookup[key] = row
    return lookup


def _format_correlation_cell(
    lookup: dict[tuple[str, str, str], Any],
    method: str,
    entropy_col: str,
    metric_col: str,
) -> str:
    rr = lookup.get((method, entropy_col, metric_col))
    if rr is None:
        return "--"
    rho = float(rr.spearman_rho)
    p_val = float(getattr(rr, "spearman_p_corrected", None) or rr.spearman_p)
    stars = _stars_for_p_value(p_val)
    return f"${rho:.3f}${stars}"


# ------------------------------------------------------------------
# DL result collection (table 8)
# ------------------------------------------------------------------


def _collect_dl_rows(
    dl_base: Path,
) -> list[dict[str, object]]:
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
            "type": "Aprendizado Profundo",
        })
    return dl_rows


def _add_metric_rankings(df: pd.DataFrame, metrics: list[str]) -> None:
    for metric_col in metrics:
        df[f"rank_{metric_col}"] = df[metric_col].rank(
            ascending=(metric_col == "rmse"), method="min"
        )


# ------------------------------------------------------------------
# Table generators
# ------------------------------------------------------------------


def table1_method_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 1: Method overview (no data needed)."""
    body: list[str] = []
    prev_cat = ""
    for _name, (cat, display, params, _cplx) in METHOD_INFO.items():
        cat_str = cat if cat != prev_cat else ""
        prev_cat = cat
        body.append(f"{cat_str} & {display} & {params} \\\\")

    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                "Visão geral dos 15 métodos clássicos de preenchimento de "
                "lacunas avaliados."
            ),
            label="tab:methods",
            col_spec="lll",
            header=(r"Categoria & Método & Parâmetros"),
            resizebox=True,
        ),
        body,
    )
    _write_tex(tex, output_dir / "methods.tex")


def table2_overall_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 2: Mean PSNR +/- CI95% per method x noise level."""
    noise_summary = summary_by_noise(df, metric="psnr")
    if noise_summary.empty:
        log.warning("No data for table2")
        return
    noise_summary = noise_summary.copy()
    noise_summary["noise_level"] = noise_summary["noise_level"].astype(str)

    methods = sorted(noise_summary["method"].unique())
    noise_levels = ["inf", "40", "30", "20"]
    present = [
        n for n in noise_levels if n in noise_summary["noise_level"].values
    ]

    if present:
        header = "Método & " + " & ".join(
            f"{n} dB" if n != "inf" else "Sem ruído" for n in present
        )
        rankings = _compute_group_rankings(
            noise_summary, "noise_level", present
        )
        body = _build_ranked_rows(
            noise_summary, "noise_level", present, methods, rankings
        )
    else:
        header = "Method"
        body = [f"{method} \\\\" for method in methods]

    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                r"PSNR médio (dB) $\pm$ IC de 95\% por método "
                r"e nível de ruído. Maior PSNR é melhor."
            ),
            label="tab:psnr-method-noise",
            col_spec="l" + "c" * len(present),
            header=header,
        ),
        body,
    )
    _write_tex(tex, output_dir / "psnr-method-noise.tex")


def table3_entropy_stratified(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 3: Mean PSNR per method x entropy bin."""
    if "entropy_15" in df.columns:
        entropy_windows = ["entropy_15"]
    elif "entropy_7" in df.columns:
        entropy_windows = ["entropy_7"]
    else:
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
        rankings = _compute_group_rankings(ent_summary, "entropy_bin", bins)
        body = _build_ranked_rows(
            ent_summary, "entropy_bin", bins, methods, rankings
        )

        label = (
            "tab:psnr-entropy-tercile"
            if ws == "15"
            else f"tab:psnr-entropy-tercile-{ws}"
        )
        tex = _render_latex_table(
            LatexTableConfig(
                caption=(
                    r"PSNR médio (dB) $\pm$ IC de 95\% estratificado "
                    rf"por tercil de entropia "
                    rf"(janela {ws}$\times${ws}). Maior PSNR é melhor."
                ),
                label=label,
                col_spec="lccc",
                header=(
                    "Método & Entropia Baixa & Entropia Média & Entropia Alta"
                ),
                resizebox=True,
            ),
            body,
        )
        _write_tex(tex, output_dir / "psnr-entropy-tercile.tex")


def table4_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 4: Spearman rho for entropy x metric."""
    entropy_cols = [c for c in df.columns if c.startswith("entropy_")]
    metric_cols = [c for c in ["psnr"] if c in df.columns]

    if not entropy_cols or not metric_cols:
        log.warning("Missing columns for table4")
        return

    corr_df = correlation_matrix(df, entropy_cols, metric_cols)
    if corr_df.empty:
        return

    methods = sorted(corr_df["method"].unique())
    lookup = _build_correlation_lookup(corr_df)

    header_parts = [
        r"$\rho_{H_{s}}(\mathrm{PSNR})$",
        *[m.replace("_", r"\_") for m in methods],
    ]

    body: list[str] = []
    for ecol in entropy_cols:
        ws = ecol.split("_")[-1]
        row_label = rf"$\rho_{{H_{{{ws}}}}}(\mathrm{{PSNR}})$"
        cells: list[str] = [row_label]
        for method in methods:
            cells.append(_format_correlation_cell(lookup, method, ecol, "psnr"))
        body.append(" & ".join(cells) + r" \\")

    ncols = len(methods)
    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                "Correlação de Spearman entre entropia e PSNR. "
                "Maior PSNR é melhor."
            ),
            label="tab:spearman-heatmap",
            col_spec="l" + "c" * ncols,
            header=" & ".join(header_parts),
            env="table*",
            resizebox=True,
        ),
        body,
    )
    _write_tex(tex, output_dir / "spearman-heatmap.tex")


def table5_kruskal_wallis(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 5: Kruskal-Wallis + Dunn post-hoc summary."""
    body: list[str] = []
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
        body.append(
            f"{metric.upper()} & ${result.statistic:.1f}$ & "
            f"{p_str} & ${result.epsilon_squared:.4f}$ "
            f"& {n_sig} \\\\"
        )

    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                "Teste de Kruskal-Wallis e diferenças par a par significativas "
                "(pós-hoc de Dunn, "
                "correção de Bonferroni)."
            ),
            label="tab:kruskal",
            col_spec="lcccc",
            header=(
                r"Métrica & Estatística H & Valor de $p$ "
                r"& $\epsilon^2$ & Pares significativos"
            ),
        ),
        body,
    )
    _write_tex(tex, output_dir / "table5_kruskal.tex")


def table6_regression(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 6: Robust regression coefficients."""
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

        body: list[str] = []
        for row in result.coefficients.itertuples():
            var = str(row.variable).replace("_", r"\_")
            p_str = (
                "$< 10^{-10}$"
                if row.p_value < 1e-10
                else f"${row.p_value:.2e}$"
            )
            ci_str = f"[{row.ci_lo:.4f}, {row.ci_hi:.4f}]"
            body.append(
                f"{var} & ${row.beta:.4f}$ & "
                f"${row.std_err:.4f}$ & "
                f"${row.z_value:.2f}$ & "
                f"{p_str} & {ci_str} \\\\"
            )

        extra: list[str] = []
        if not result.vif.empty:
            extra.append(r"\vspace{0.5em}")
            extra.append(r"\begin{tabular}{lr}")
            extra.append(r"\toprule")
            extra.append(r"Variável & FIV \\")
            extra.append(r"\midrule")
            for vrow in result.vif.itertuples():
                var = str(vrow.variable).replace("_", r"\_")
                extra.append(f"{var} & ${vrow.vif:.2f}$ \\\\")
            extra.append(r"\bottomrule")
            extra.append(r"\end{tabular}")

        mu = metric.upper()
        label = (
            "tab:robust-regression"
            if metric == "psnr"
            else f"tab:robust-regression-{metric}"
        )
        tex = _render_latex_table(
            LatexTableConfig(
                caption=(
                    f"Regressão robusta (RLM/HuberT) para {mu}. "
                    f"$R^2_{{adj}} = {result.r_squared_adj:.4f}$"
                    f", $n = {result.n}$."
                ),
                label=label,
                col_spec="lrrrrr",
                header=(
                    r"Variável & $\beta$ & Erro Padrão "
                    r"& $z$ & $p$ & IC 95\%"
                ),
                font_size=r"\tiny",
                env="table*",
            ),
            body,
            extra_after_tabular=extra or None,
        )
        filename = (
            "robust-regression.tex"
            if metric == "psnr"
            else f"table6_regression_{metric}.tex"
        )
        _write_tex(tex, output_dir / filename)


def table7_satellite(df: pd.DataFrame, output_dir: Path) -> None:
    """Table 7: Mean PSNR per method x satellite."""
    sat_summary = summary_by_satellite(df, metric="psnr")

    if sat_summary.empty:
        log.warning("No data for table7")
        return

    methods = sorted(sat_summary["method"].unique())
    satellites = sorted(sat_summary["satellite"].unique())

    header = "Método & " + " & ".join(s.replace("_", r"\_") for s in satellites)
    rankings = _compute_group_rankings(sat_summary, "satellite", satellites)
    body = _build_ranked_rows(
        sat_summary, "satellite", satellites, methods, rankings
    )

    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                r"PSNR médio (dB) $\pm$ IC de 95\% por método "
                r"e sensor de satélite. Maior PSNR é melhor."
            ),
            label="tab:psnr-satellite",
            col_spec="l" + "c" * len(satellites),
            header=header,
            env="table*",
        ),
        body,
    )
    _write_tex(tex, output_dir / "psnr-satellite.tex")


def table8_dl_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> None:
    """Table 8: Classical vs. DL method comparison."""
    classical_summary = (
        df
        .groupby("method", observed=True)[["psnr", "ssim", "rmse"]]
        .mean()
        .reset_index()
    )
    classical_summary["type"] = "Clássico"

    dl_base = (
        results_dir.parent / "dl_eval" if results_dir is not None else Path(".")
    )
    dl_rows = _collect_dl_rows(dl_base)

    if not dl_rows:
        log.info(
            "No DL results at %s. Skipping table8.",
            dl_base,
        )
        return

    dl_summary = pd.DataFrame(dl_rows)
    combined = pd.concat([classical_summary, dl_summary], ignore_index=True)
    combined = combined.sort_values("psnr", ascending=False)

    present = [m for m in ["psnr", "ssim", "rmse"] if m in combined.columns]

    _add_metric_rankings(combined, present)

    body: list[str] = []
    prev_type = ""
    for row in combined.itertuples():
        row_type = str(row.type)
        type_str = row_type if row_type != prev_type else ""
        prev_type = row_type
        method_str = str(row.method).replace("_", r"\_")
        cells: list[str] = [type_str, method_str]
        for m_col in present:
            val = float(getattr(row, m_col))
            rank = int(getattr(row, f"rank_{m_col}"))
            cells.append(_format_table8_metric(val, rank))
        body.append(" & ".join(cells) + r" \\")

    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                "Comparação de desempenho: interpolação clássica "
                r"vs.\ métodos de aprendizado profundo. "
                "Maior PSNR e SSIM são melhores; menor RMSE é melhor."
            ),
            label="tab:dl-results",
            col_spec="ll" + "c" * len(present),
            header=(
                "Tipo & Método & " + " & ".join(m.upper() for m in present)
            ),
        ),
        body,
    )
    _write_tex(tex, output_dir / "dl-results.tex")


# ------------------------------------------------------------------
# DL metrics table (table 9)
# ------------------------------------------------------------------

_DL_MODELS: tuple[str, ...] = ("ae", "vae", "gan", "unet", "vit")

_DL_COLUMN_SPECS: list[tuple[str, str, bool, str]] = [
    ("val_psnr", "PSNR (dB)", True, ".2f"),
    ("val_ssim", "SSIM", True, ".4f"),
    ("val_rmse", "RMSE", False, ".4f"),
    ("val_pixel_acc_002", r"Acur. @0,02", True, ".4f"),
    ("val_f1_002", r"F1 @0,02", True, ".4f"),
    ("val_pixel_acc_005", r"Acur. @0,05", True, ".4f"),
    ("val_f1_005", r"F1 @0,05", True, ".4f"),
    ("val_pixel_acc_01", r"Acur. @0,10", True, ".4f"),
    ("val_f1_01", r"F1 @0,10", True, ".4f"),
]


def _load_dl_history(dl_results_dir: Path, model: str) -> dict | None:
    path = dl_results_dir / f"{model}_history.json"
    if not path.exists():
        log.debug("History not found: %s", path)
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _format_dl_metric(value: float, rank: int, fmt: str) -> str:
    formatted = f"{value:{fmt}}"
    if rank == 1:
        return f"\\textbf{{{formatted}}}"
    if rank == 2:
        return f"\\underline{{{formatted}}}"
    return formatted


def table_dl_metrics(dl_results_dir: Path, output_dir: Path) -> None:
    """Table DL: Final-epoch metrics summary for all DL models."""
    histories = {m: _load_dl_history(dl_results_dir, m) for m in _DL_MODELS}
    available = {m: h for m, h in histories.items() if h and h.get("epochs")}
    if not available:
        log.warning("No DL histories found. Skipping table_dl_metrics.")
        return

    # Build rows: model -> {col_key: value}
    rows: dict[str, dict[str, float | None]] = {}
    for model, hist in available.items():
        last = hist["epochs"][-1]
        rows[model] = {key: last.get(key) for key, _, _, _ in _DL_COLUMN_SPECS}

    # Compute per-column ranks
    col_ranks: dict[str, dict[str, int]] = {}
    for key, _, higher_is_better, _ in _DL_COLUMN_SPECS:
        vals = {
            m: v
            for m, v in ((m, rows[m].get(key)) for m in available)
            if v is not None and np.isfinite(v)
        }
        sorted_models = sorted(
            vals, key=lambda m: vals[m], reverse=higher_is_better
        )
        col_ranks[key] = {m: i + 1 for i, m in enumerate(sorted_models)}

    body: list[str] = []
    for model in available:
        cells: list[str] = [model.upper()]
        for key, _, _, fmt in _DL_COLUMN_SPECS:
            val = rows[model].get(key)
            if val is None or not np.isfinite(val):
                cells.append("--")
            else:
                rank = col_ranks[key].get(model, 99)
                cells.append(_format_dl_metric(val, rank, fmt))
        body.append(" & ".join(cells) + r" \\")

    col_header = "Modelo & " + " & ".join(
        label for _, label, _, _ in _DL_COLUMN_SPECS
    )
    ncols = 1 + len(_DL_COLUMN_SPECS)
    tex = _render_latex_table(
        LatexTableConfig(
            caption=(
                "Métricas de avaliação dos modelos de aprendizado profundo "
                r"na última época de treinamento. "
                "Maior PSNR, SSIM e acurácia são melhores; menor RMSE é melhor."
                r"\textbf{Negrito}: melhor; \underline{sublinhado}: segundo "
                "melhor."
            ),
            label="tab:dl-metrics",
            col_spec="l" + "c" * (ncols - 1),
            header=col_header,
            font_size=r"\footnotesize",
            env="table*",
        ),
        body,
    )
    _write_tex(tex, output_dir / "dl-metrics.tex")


# ------------------------------------------------------------------
# Dispatch table and CLI
# ------------------------------------------------------------------


def _make_table_dispatch(
    df: pd.DataFrame,
    results_dir: Path,
    output_dir: Path,
) -> dict[int, tuple[str, Any]]:
    """Build dispatch table mapping number -> (name, callable).

    Each callable takes no arguments (partial application).
    """
    return {
        1: (
            "method_overview",
            lambda: table1_method_overview(df, output_dir),
        ),
        2: (
            "overall_results",
            lambda: table2_overall_results(df, output_dir),
        ),
        3: (
            "entropy_stratified",
            lambda: table3_entropy_stratified(df, output_dir),
        ),
        4: (
            "correlation",
            lambda: table4_correlation(df, output_dir),
        ),
        7: (
            "satellite",
            lambda: table7_satellite(df, output_dir),
        ),
        8: (
            "dl_comparison",
            lambda: table8_dl_comparison(
                df, output_dir, results_dir=results_dir
            ),
        ),
    }


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate LaTeX tables from experiment results."),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to classical experiment results directory.",
    )
    parser.add_argument(
        "--dl-results",
        type=Path,
        default=None,
        help="Path to DL models directory containing *_history.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output directory for tables. Default: results_dir/tables/"),
    )
    parser.add_argument(
        "--table",
        type=int,
        default=None,
        help="Generate only this classical table number (1-8).",
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

    if args.results is None and args.dl_results is None:
        log.error("Provide at least one of --results or --dl-results.")
        return

    base_dir = args.results or args.dl_results
    output_dir = args.output or base_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(base_dir / "tables.log")

    if args.results is not None:
        results_dir = args.results
        df = load_results(results_dir)
        dispatch = _make_table_dispatch(df, results_dir, output_dir)

        if args.table is not None:
            if args.table not in dispatch:
                log.error("Invalid table number: %d", args.table)
                return
            _, fn = dispatch[args.table]
            fn()
        else:
            for num, (name, fn) in dispatch.items():
                try:
                    fn()
                except Exception:
                    log.exception("Error generating table %d (%s)", num, name)

    if args.dl_results is not None:
        try:
            table_dl_metrics(args.dl_results, output_dir)
        except Exception:
            log.exception("Error generating DL metrics table")

    log.info("Tables saved to: %s", output_dir)


if __name__ == "__main__":
    main()
