"""Table 0b: Classical methods by family."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from pdi_pipeline.config import MethodConfig, load_config
from pdi_pipeline.methods.registry import list_aliases

from ..data_loader import CATEGORY_LABELS
from .cli import run_table
from .common import tex_escape, write_tex

_DEFAULT_CONFIG = Path("config/paper_results.yaml")
_CONFIG_ENV_VARS = ("PDI_METHODS_CONFIG", "PDI_EXPERIMENT_CONFIG")
_RESULTS_ENV_VAR = "PDI_RESULTS_DIR"

_METHOD_LABELS: dict[str, str] = {
    "nearest": "Nearest Neighbor",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "lanczos": "Lanczos (PG)",
    "idw": "IDW",
    "rbf": "RBF",
    "spline": "Thin Plate Spline",
    "kriging": "Ordinary Kriging",
    "dct": "DCT",
    "wavelet": "Wavelet",
    "tv": "Total Variation",
    "cs_dct": "L1-DCT (CS)",
    "cs_wavelet": "L1-Wavelet (CS)",
    "non_local": "Non-Local Means",
    "exemplar": "Exemplar-Based",
}


_PARAM_SYMBOLS: dict[str, str] = {
    "lam": r"\lambda",
    "lambda": r"\lambda",
    "lambda_param": r"\lambda",
    "beta": r"\beta",
    "mu": r"\mu",
    "sigma": r"\sigma",
}


def _resolve_canonical(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_param_key(key: str) -> str:
    symbol = _PARAM_SYMBOLS.get(key)
    if symbol is None:
        return tex_escape(key)
    return f"${symbol}$"


def _format_params(method: str, params: dict[str, Any]) -> str:
    if not params:
        return "--"
    if method == "lanczos" and "a" in params:
        return f"a={_format_param_value(params['a'])}"
    if method == "idw" and "power" in params:
        return f"p={_format_param_value(params['power'])}"
    if method == "rbf" and "kernel" in params:
        kernel = str(params["kernel"])
        if kernel in {"thin_plate_spline", "tps"}:
            return "kernel TPS"
        return f"kernel {kernel}"
    if method == "tv" and "max_iterations" in params:
        return f"iter={_format_param_value(params['max_iterations'])}"
    if method == "non_local":
        h_rel = params.get("h_rel")
        patch_size = params.get("patch_size")
        patch_distance = params.get("patch_distance")
        if (
            h_rel is not None
            and patch_size is not None
            and patch_distance is not None
        ):
            return (
                f"h={_format_param_value(h_rel)}, "
                f"p={_format_param_value(patch_size)}, "
                f"s={_format_param_value(patch_distance)}"
            )
    items = [
        f"{_format_param_key(key)}={tex_escape(_format_param_value(value))}"
        for key, value in params.items()
    ]
    if len(items) > 3:
        return ", \\newline ".join(items)
    return ", ".join(items)


def _group_methods(
    methods: list[MethodConfig],
) -> dict[str, list[MethodConfig]]:
    grouped: dict[str, list[MethodConfig]] = {
        key: [] for key in CATEGORY_LABELS
    }
    for method in methods:
        grouped.setdefault(method.category, []).append(method)
    return grouped


def _resolve_config_path() -> Path:
    for key in _CONFIG_ENV_VARS:
        candidate = os.environ.get(key)
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    results_dir = os.environ.get(_RESULTS_ENV_VAR)
    if results_dir:
        base = Path(results_dir)
        for name in ("config.yaml", "config.yml", "experiment.yaml"):
            path = base / name
            if path.exists():
                return path

    return _DEFAULT_CONFIG


def table_methods(df: pd.DataFrame, output_dir: Path) -> None:
    """Classical methods grouped by category."""
    classic = df[df.get("type", "") == "Clássico"]
    if classic.empty:
        return

    config = load_config(_resolve_config_path())
    aliases = list_aliases()
    grouped = _group_methods(config.methods)

    rows: list[str] = []
    ordered_categories = [key for key in CATEGORY_LABELS if key in grouped]
    for cat_index, category in enumerate(ordered_categories):
        methods = grouped.get(category, [])
        if not methods:
            continue
        for idx, method in enumerate(methods):
            canonical = _resolve_canonical(method.name, aliases)
            label = tex_escape(_METHOD_LABELS.get(canonical, canonical))
            params = _format_params(canonical, method.params)
            category_label = CATEGORY_LABELS[category] if idx == 0 else ""
            category_label = (
                tex_escape(category_label) if category_label else ""
            )
            rows.append(
                " ".join([
                    " & ".join([category_label, label, params]),
                    r"\\",
                ])
            )
        if cat_index < len(ordered_categories) - 1:
            rows.append(r"\midrule")

    if not rows:
        return

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Visão geral dos 15 métodos clássicos de preenchimento de lacunas avaliados.}",  # noqa: E501
        r"\label{tab:methods}",
        r"\footnotesize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llp{3cm}}",
        r"\toprule",
        r"Categoria & Método & Parâmetros \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(lines)
    write_tex(tex, output_dir / "methods.tex")


def main() -> None:
    run_table(table_methods, "Methods")


if __name__ == "__main__":
    main()
