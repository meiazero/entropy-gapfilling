"""Table 0c: Executed DL configurations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data_loader import load_all_dl_histories
from .cli import run_table
from .common import tex_escape, wrap_table, write_tex

LOSS_BY_MODEL: dict[str, str] = {
    "ae": "GapPixelLoss (MSE, apenas pixels de lacuna)",
    "vae": "GapPixelLoss (MSE, apenas pixels de lacuna)",
    "unet": "GapPixelLoss (MSE, apenas pixels de lacuna)",
    "vit": "GapPixelLoss (MSE, apenas pixels de lacuna)",
    "gan": "GapPixelLoss (L1, apenas pixels de lacuna)",
}


def _mode_value(values: pd.Series) -> str:
    if values.empty:
        return "N/A"
    mode = values.mode(dropna=True)
    if mode.empty:
        return "N/A"
    return str(mode.iloc[0])


def table_dl_architectures(df: pd.DataFrame, output_dir: Path) -> None:
    """Executed DL configurations from experiment results."""
    dl = df[df.get("type", "") == "DL"]
    if dl.empty or "method" not in dl.columns:
        return

    if "architecture" not in dl.columns:
        dl = dl.copy()
        dl["architecture"] = "N/A"
    if "entropy_scenario" not in dl.columns:
        dl = dl.copy()
        dl["entropy_scenario"] = "N/A"

    histories = load_all_dl_histories()
    body: list[str] = []
    for method, grp in dl.groupby("method", observed=True, sort=False):
        architecture = _mode_value(grp["architecture"])
        scenario_key = _mode_value(grp["entropy_scenario"])
        loss = LOSS_BY_MODEL.get(method, "GapPixelLoss (MSE)")

        meta = (
            histories.get(scenario_key, {}).get(method, {}).get("metadata", {})
        )
        epochs = meta.get("epochs")
        batch = meta.get("batch_size")
        lr = meta.get("lr")
        wd = meta.get("weight_decay")

        body.append(
            " & ".join([
                tex_escape(method),
                tex_escape(architecture),
                tex_escape(loss),
                str(epochs) if epochs is not None else "--",
                str(batch) if batch is not None else "--",
                f"{lr:.4f}" if lr is not None else "--",
                f"{wd:.4f}" if wd is not None else "--",
            ])
            + r" \\"
        )

    if not body:
        return

    header = "Modelo & Arquitetura & Função de perda & Épocas & Batch & LR & WD"
    tex = wrap_table(
        body,
        caption=(
            "Configurações DL executadas nos experimentos, com "
            "características da arquitetura e do treino."
        ),
        label="tab:dl-architectures",
        col_spec="l p{0.36\\linewidth} l r r r r",
        header=header,
        env="table*",
        resizebox=True,
    )
    write_tex(tex, output_dir / "dl-architectures.tex")


def main() -> None:
    run_table(table_dl_architectures, "DL architectures")


if __name__ == "__main__":
    main()
