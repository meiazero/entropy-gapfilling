"""Shared helpers for figure generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STYLE_PATH = _PROJECT_ROOT / "images" / "style.mplstyle"

DPI = 300
FONT_SIZE = 8

_PUB_FONT = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


@dataclass
class FigureSettings:
    png_only: bool = False
    bootstrap_samples: int = 500


SETTINGS = FigureSettings()


def configure_settings(*, png_only: bool, bootstrap_samples: int) -> None:
    SETTINGS.png_only = png_only
    SETTINGS.bootstrap_samples = bootstrap_samples


def setup_style() -> None:
    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    plt.rcParams.update(_PUB_FONT)
    sns.set_palette("Set2")


def style_axes(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    grid_axis: str = "both",
    grid_alpha: float = 0.3,
    grid_linewidth: float = 0.5,
    title_size: int | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=title_size or FONT_SIZE + 1)
    if grid:
        ax.grid(
            True, axis=grid_axis, alpha=grid_alpha, linewidth=grid_linewidth
        )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    if not SETTINGS.png_only:
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    log.info("Saved %s", name)
