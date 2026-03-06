"""Table 0b: Classical methods by family."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data_loader import CATEGORY_LABELS
from .cli import run_table
from .common import tex_escape, wrap_table, write_tex


def _format_methods_list(methods: list[str]) -> str:
    items = [tex_escape(m) for m in methods]
    return ", ".join(items)


def table_methods(df: pd.DataFrame, output_dir: Path) -> None:
    """Classical methods grouped by category."""
    classic = df[df.get("type", "") == "Clássico"]
    if classic.empty or "method" not in classic.columns:
        return

    if "method_category" in classic.columns:
        grouped = classic.groupby("method_category", observed=True)[
            "method"
        ].unique()
        categories = grouped.to_dict()
    else:
        categories = {"outros": classic["method"].unique().tolist()}

    ordered_keys = [key for key in CATEGORY_LABELS if key in categories]
    remaining = [key for key in categories if key not in ordered_keys]
    ordered = [*ordered_keys, *sorted(remaining)]

    body: list[str] = []
    for key in ordered:
        methods = sorted(categories.get(key, []))
        if not methods:
            continue
        label = CATEGORY_LABELS.get(key, key)
        body.append(
            " & ".join([tex_escape(label), _format_methods_list(methods)])
            + r" \\"
        )

    if not body:
        return

    header = "Família & Métodos"
    tex = wrap_table(
        body,
        caption="Métodos clássicos avaliados, agrupados por família.",
        label="tab:methods",
        col_spec="lp{0.7\\linewidth}",
        header=header,
        resizebox=True,
    )
    write_tex(tex, output_dir / "methods.tex")


def main() -> None:
    run_table(table_methods, "Methods")


if __name__ == "__main__":
    main()
