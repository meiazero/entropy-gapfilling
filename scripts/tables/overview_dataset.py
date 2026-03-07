"""Dataset overview table from manifest data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .common import bold, tex_escape, write_tex

log = logging.getLogger(__name__)

_DEFAULT_MANIFEST = Path("preprocessed/manifest.csv")

_SENSOR_LABELS = {
    "sentinel2": "Sentinel-2",
    "landsat8": "Landsat-8",
    "landsat9": "Landsat-9",
    "modis": "MODIS",
}

_SENSOR_METADATA: dict[str, dict[str, str]] = {
    "sentinel2": {
        "resolution": r"10\,m",
    },
    "landsat8": {
        "resolution": r"30\,m",
    },
    "landsat9": {
        "resolution": r"30\,m",
    },
    "modis": {
        "resolution": r"500\,m",
    },
}


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


class ManifestNotFoundError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Manifest not found: {path}")


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ManifestNotFoundError(path)
    return pd.read_csv(path, usecols=["satellite", "split"])


def table_overview_dataset(_df: object, output_dir: Path) -> None:
    """Generate dataset composition table from manifest.csv."""
    data = _load_manifest(_DEFAULT_MANIFEST)
    counts = data.groupby(["satellite", "split"], observed=True).size()

    sensors = ["sentinel2", "landsat8", "landsat9", "modis"]

    rows: list[str] = []
    totals = {"total": 0, "train": 0, "val": 0, "test": 0}
    for sensor in sensors:
        total = int(counts.get((sensor, "train"), 0))
        total += int(counts.get((sensor, "val"), 0))
        total += int(counts.get((sensor, "test"), 0))
        train = int(counts.get((sensor, "train"), 0))
        val = int(counts.get((sensor, "val"), 0))
        test = int(counts.get((sensor, "test"), 0))
        totals["total"] += total
        totals["train"] += train
        totals["val"] += val
        totals["test"] += test

        meta = _SENSOR_METADATA.get(sensor, {})
        rows.append(
            " & ".join([
                tex_escape(_SENSOR_LABELS.get(sensor, sensor)),
                meta.get("resolution", "--"),
                _format_int(train),
                _format_int(val),
                _format_int(test),
                _format_int(total),
            ])
            + r" \\"
        )

    rows.append(
        " & ".join([
            bold("Total"),
            "--",
            bold(_format_int(totals["train"])),
            bold(_format_int(totals["val"])),
            bold(_format_int(totals["test"])),
            bold(_format_int(totals["total"])),
        ])
        + r" \\"
    )

    lines = [
        r"\begin{table}[t]",
        r" \centering",
        r" \caption{Estatística do conjunto de dados por sensor de satélite.",
        r" Todos os patches são de $64 \times 64$ pixels com 4 bandas "
        r"espectrais (vermelha, azul, verde e IR).",
        r" Divisão dos dados: 80\% treino, 10\% validação e 10\% teste.}",
        r" \label{tab:dataset-stats}",
        r"\resizebox{\linewidth}{!}{%",
        r" \begin{tabular}{lrrrrr}",
        r" \toprule",
        (
            r" Sensor & Resolução & \#Patches (Treino) & "
            r"\#Patches (Validação) & \#Patches (Teste) & "
            r"\#Patches (Total)\\"
        ),
        r" \midrule",
        *rows,
        r" \bottomrule",
        r" \end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tex(tex, output_dir / "dataset-stats.tex")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset overview table."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/tables"),
        help="Output directory. Defaults to docs/tables/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to the manifest.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    global _DEFAULT_MANIFEST
    _DEFAULT_MANIFEST = args.manifest
    table_overview_dataset(None, args.output)
    log.info("Saved dataset overview table to: %s", args.output)


if __name__ == "__main__":
    main()
