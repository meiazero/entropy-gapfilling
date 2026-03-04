"""Dataset overview table from manifest data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_MANIFEST = Path("preprocessed/manifest.csv")

_SENSOR_LABELS = {
    "sentinel2": "Sentinel-2",
    "landsat8": "Landsat-8",
    "landsat9": "Landsat-9",
    "modis": "MODIS",
}


class ManifestNotFoundError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Manifest not found: {path}")


def _ensure_scripts_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ManifestNotFoundError(path)
    return pd.read_csv(path, usecols=["satellite", "split"])


def table_overview_dataset(_df: object, output_dir: Path) -> None:
    """Generate dataset composition table from manifest.csv."""
    _ensure_scripts_on_path()
    from tables.common import bold, tex_escape, wrap_table, write_tex

    data = _load_manifest(_DEFAULT_MANIFEST)
    counts = data.groupby(["satellite", "split"], observed=True).size()

    sensors = ["sentinel2", "landsat8", "landsat9", "modis"]

    rows = []
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
        rows.append(
            " & ".join([
                tex_escape(_SENSOR_LABELS.get(sensor, sensor)),
                str(total),
                str(train),
                str(val),
                str(test),
            ])
            + r" \\"
        )

    rows.append(
        " & ".join([
            bold("Total"),
            bold(str(totals["total"])),
            bold(str(totals["train"])),
            bold(str(totals["val"])),
            bold(str(totals["test"])),
        ])
        + r" \\"
    )

    tex = wrap_table(
        rows,
        caption=(
            "Estatísticas do conjunto de dados (recortes $64\\times64$) "
            "por sensor e partição."
        ),
        label="tab:dataset-stats",
        col_spec="lrrrr",
        header="Sensor & Total & Treino & Val. & Teste",
        font_size=r"\footnotesize",
    )
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
    _ensure_scripts_on_path()
    args = _parse_args()
    global _DEFAULT_MANIFEST
    _DEFAULT_MANIFEST = args.manifest
    table_overview_dataset(None, args.output)
    log.info("Saved dataset overview table to: %s", args.output)


if __name__ == "__main__":
    main()
