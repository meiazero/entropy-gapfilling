"""Shared helpers for LaTeX table generation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


@dataclass
class TableSettings:
    bootstrap_samples: int = 1000


SETTINGS = TableSettings()


def configure_settings(*, bootstrap_samples: int) -> None:
    SETTINGS.bootstrap_samples = bootstrap_samples


def write_tex(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    log.info("Saved %s", path)


def tex_escape(value: str) -> str:
    return (
        str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    )


def format_pm(mean: float, ci_half: float, fmt: str = ".2f") -> str:
    return f"${mean:{fmt}}_{{\\pm {ci_half:{fmt}}}}$"


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def underline(text: str) -> str:
    return f"\\underline{{{text}}}"


def ranked_cell(
    value: float,
    ci_half: float,
    rank: int,
    fmt: str = ".2f",
) -> str:
    base = format_pm(value, ci_half, fmt)
    if rank == 1:
        return bold(base)
    if rank == 2:
        return underline(base)
    if rank == 3:
        return f"\\textit{{{base}}}"
    return base


def ranked_plain(value: float, rank: int, fmt: str = ".3f") -> str:
    base = f"${value:{fmt}}$"
    if rank == 1:
        return bold(base)
    if rank == 2:
        return underline(base)
    if rank == 3:
        return f"\\textit{{{base}}}"
    return base


def stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def ci95_half(values: pd.Series) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    se = float(values.std() / np.sqrt(n))
    return float(stats.t.ppf(0.975, n - 1) * se)


def bootstrap_ci_half(
    values: pd.Series,
    clusters: pd.Series | None,
    *,
    stat_fn: Callable[[pd.Series], float],
    n_boot: int | None = None,
    seed: int = 42,
) -> float:
    samples = n_boot if n_boot is not None else int(SETTINGS.bootstrap_samples)
    vals = values.dropna()
    if len(vals) < 2:
        return 0.0
    if clusters is None or clusters.isna().all():
        return ci95_half(vals)

    data = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    unique_clusters = data["cluster"].unique().tolist()
    if len(unique_clusters) < 2:
        return ci95_half(vals)

    grouped = data.groupby("cluster", sort=False)
    cluster_values = [grp["value"].to_numpy() for _, grp in grouped]
    cluster_index = [grp.index.to_numpy() for _, grp in grouped]

    rng = np.random.default_rng(seed)
    boot_stats = []
    n_clusters = len(cluster_values)
    for _ in range(samples):
        sampled_idx = rng.integers(0, n_clusters, size=n_clusters)
        values_concat = np.concatenate([cluster_values[i] for i in sampled_idx])
        index_concat = np.concatenate([cluster_index[i] for i in sampled_idx])
        stat = stat_fn(pd.Series(values_concat, index=index_concat))
        if not np.isnan(stat):
            boot_stats.append(float(stat))
    if not boot_stats:
        return 0.0
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return float((hi - lo) / 2.0)


def wrap_table(
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
