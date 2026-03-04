"""Figure 11: DL model comparison heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import load_all_dl_histories
from .cli import run_no_df
from .common import FONT_SIZE, save_figure, style_axes


def _select_best_epoch(hist: dict) -> dict:
    epochs = hist.get("epochs", [])
    if not epochs:
        return {}
    valid_loss = [e for e in epochs if e.get("val_loss") is not None]
    if valid_loss:
        return min(valid_loss, key=lambda e: e.get("val_loss"))
    valid_psnr = [e for e in epochs if e.get("val_psnr") is not None]
    if valid_psnr:
        return max(valid_psnr, key=lambda e: e.get("val_psnr"))
    return epochs[-1]


def _build_dl_comparison_frame(
    models: dict[str, dict[str, list[dict[str, float | None]]]],
    metric_keys: list[tuple[str, str, bool]],
) -> tuple[pd.DataFrame, list[str]]:
    model_labels: list[str] = []
    rows: list[dict[str, float | None]] = []
    for model, hist in models.items():
        best = _select_best_epoch(hist)
        if not best:
            continue
        row = {label: best.get(key) for key, label, _ in metric_keys}
        rows.append(row)
        model_labels.append(model.upper())
    return pd.DataFrame(rows, index=model_labels), model_labels


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
            len(col_labels) * 1.0 + 0.8,
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
    scenario_title = scenario.replace("_", " ").title()
    style_axes(
        ax,
        title=f"Comparação DL (melhor época) - {scenario_title}",
        grid=False,
        title_size=FONT_SIZE + 1,
        title_pad=8,
    )
    save_figure(fig, output_dir, f"dl_comparison_{scenario}")
    plt.close(fig)


def fig11_dl_comparison(output_dir: Path) -> None:
    """Heatmap comparing all DL models across final-epoch metrics."""
    histories = load_all_dl_histories()

    metric_keys = [
        ("val_psnr", "PSNR", True),
        ("val_ssim", "SSIM", True),
        ("val_rmse", "RMSE", False),
        ("val_sam", "SAM", False),
        ("val_ergas", "ERGAS", False),
        ("val_f1_002", "F1@0.02", True),
        ("val_f1_005", "F1@0.05", True),
        ("val_f1_01", "F1@0.10", True),
    ]

    for scenario, models in histories.items():
        if not models:
            continue

        raw_df, model_labels = _build_dl_comparison_frame(models, metric_keys)
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


def main() -> None:
    run_no_df(fig11_dl_comparison, "DL comparison heatmap")


if __name__ == "__main__":
    main()
