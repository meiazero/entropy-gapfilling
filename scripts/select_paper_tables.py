"""Copy only the curated paper tables into docs/tables."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    source_dir = root / "paper_assets" / "tables"
    output_dir = root / "docs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        "dataset-stats.tex",
        "methods.tex",
        "dl-architectures.tex",
        "global-scoreboard-classical.tex",
        "global-scoreboard-dl.tex",
        "inferential-summary-classical.tex",
        "inferential-summary-dl.tex",
        "spectral-rmse-classical.tex",
        "spectral-rmse-dl.tex",
        "runtime-speed-classical.tex",
        "runtime-speed-dl.tex",
        "psnr-drop-entropy-classical.tex",
        "psnr-drop-entropy-dl.tex",
        "psnr-noise-slope-entropy-classical.tex",
        "psnr-noise-slope-entropy-dl.tex",
        "spearman-entropy-classical-15.tex",
        "spearman-entropy-dl-15.tex",
    ]

    copied = 0
    for name in selected:
        src = source_dir / name
        dst = output_dir / name
        if not src.exists():
            print(f"WARN missing table: {name}")
            continue
        shutil.copy2(src, dst)
        copied += 1

    print("Copied", copied, "of", len(selected), "tables")


if __name__ == "__main__":
    main()
