"""DL model comparison heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import display_method_name, load_combined
from .cli import run_no_df
from .common import FONT_SIZE, save_figure, style_axes


def _build_dl_comparison_frame(
    df: pd.DataFrame,
    metric_keys: list[tuple[str, str, bool]],
) -> tuple[pd.DataFrame, list[str]]:
    grouped = df.groupby("method", observed=True)
    rows: list[dict[str, float | None]] = []
    model_labels: list[str] = []
    ranking: list[tuple[str, float]] = []

    for method, method_df in grouped:
        row = {
            label: float(method_df[key].median())
            if key in method_df.columns
            else np.nan
            for key, label, _ in metric_keys
        }
        rows.append(row)
        model_labels.append(method)
        ranking.append((method, row.get("PSNR", float("nan"))))

    ordered_methods = [
        method
        for method, _ in sorted(ranking, key=lambda item: item[1], reverse=True)
    ]
    raw_df = pd.DataFrame(rows, index=model_labels)
    raw_df = raw_df.loc[ordered_methods]
    display_labels = [display_method_name(method) for method in raw_df.index]
    raw_df.index = display_labels
    return raw_df, display_labels


def _normalize_dl_metrics(
    raw_df: pd.DataFrame,
    higher_better: dict[str, bool],
) -> pd.DataFrame:
    normed = pd.DataFrame(
        index=raw_df.index, columns=raw_df.columns, dtype=float
    )
    for col in raw_df.columns:
        col_min = raw_df[col].min()
        col_max = raw_df[col].max()
        rng = col_max - col_min
        if rng < 1e-12:
            normed[col] = 0.5
        elif higher_better.get(col, True):
            normed[col] = (raw_df[col] - col_min) / rng
        else:
            normed[col] = (col_max - raw_df[col]) / rng
    return normed


def _format_dl_cell_value(col: str, raw_val: float | None) -> str:
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "N/D"
    if col == "PSNR":
        return f"{raw_val:.2f}"
    return f"{raw_val:.4f}"


def _render_dl_comparison_heatmap(
    raw_df: pd.DataFrame,
    normed: pd.DataFrame,
    col_labels: list[str],
    model_labels: list[str],
    scenario: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            len(col_labels) * 1.0 + 0,
            len(model_labels) * 0.6 + 0.8,
        ),
        constrained_layout=True,
    )

    im = ax.imshow(
        normed.values.astype(float),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=FONT_SIZE)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=FONT_SIZE)

    for r_idx in range(len(model_labels)):
        for c_idx, col in enumerate(col_labels):
            raw_val = raw_df.iloc[r_idx][col]
            cell_text = _format_dl_cell_value(col, raw_val)
            bg_val = normed.iloc[r_idx][col]
            bg = 0.5 if pd.isna(bg_val) else float(bg_val)
            text_color = "black" if 0.25 < bg < 0.85 else "white"
            ax.text(
                c_idx,
                r_idx,
                cell_text,
                ha="center",
                va="center",
                fontsize=FONT_SIZE - 1,
                color=text_color,
            )

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Pontuação norm.")
    style_axes(
        ax,
        xlabel="Métrica de teste",
        ylabel="Arquitetura",
        grid=False,
    )
    save_figure(fig, output_dir, f"dl_comparison_{scenario}")
    plt.close(fig)


def fig_dl_comparison(output_dir: Path) -> None:
    """Heatmap comparing DL models using evaluation medians."""
    df = load_combined()
    if df.empty or "type" not in df.columns:
        return

    df = df[df["type"] == "DL"].copy()
    if df.empty:
        return
    if "entropy_scenario" in df.columns and "entropy_all" in set(
        df["entropy_scenario"]
    ):
        df = df[df["entropy_scenario"] == "entropy_all"]
    if df.empty:
        return

    metric_keys = [
        ("psnr", "PSNR", True),
        ("ssim", "SSIM", True),
        ("rmse", "RMSE", False),
        ("sam", "SAM", False),
        ("ergas", "ERGAS", False),
    ]

    scenario = "entropy_all"
    raw_df, model_labels = _build_dl_comparison_frame(df, metric_keys)
    col_labels = [label for _, label, _ in metric_keys]
    higher_better = {label: higher for _, label, higher in metric_keys}
    normed = _normalize_dl_metrics(raw_df, higher_better)
    _render_dl_comparison_heatmap(
        raw_df,
        normed,
        col_labels,
        model_labels,
        scenario,
        output_dir,
    )

    source_pdf = output_dir / f"dl_comparison_{scenario}.pdf"
    if source_pdf.exists():
        source_pdf.replace(output_dir / "fig_dl_comparison.pdf")


def main() -> None:
    run_no_df(fig_dl_comparison, "DL comparison heatmap")


if __name__ == "__main__":
    main()
