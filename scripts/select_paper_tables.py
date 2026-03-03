"""Select best table variants for the article.

Decision rule: pick the noise level that maximizes median PSNR and the
entropy window default (15 when available). Copy only the corresponding
tables to docs/tables with stable filenames and labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from data_loader import ENTROPY_WINDOWS, NOISE_ORDER, load_combined


def _best_noise(df: pd.DataFrame) -> str:
    if "noise_level" not in df.columns:
        return "inf"
    med = (
        df
        .groupby("noise_level", observed=True)["psnr"]
        .median()
        .reindex([n for n in NOISE_ORDER if n in df["noise_level"].unique()])
    )
    if med.empty:
        return "inf"
    return str(med.idxmax())


def _select_noise(df: pd.DataFrame, *, preferred: str) -> str:
    if "noise_level" not in df.columns:
        return "inf"
    available = set(df["noise_level"].dropna().astype(str).unique().tolist())
    if preferred in available:
        return preferred
    return _best_noise(df)


def _best_entropy_window(df: pd.DataFrame) -> int:
    if 15 in ENTROPY_WINDOWS and "entropy_15" in df.columns:
        return 15
    return ENTROPY_WINDOWS[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select paper table assets")
    parser.add_argument(
        "--noise",
        default="inf",
        help=(
            "Noise level to use as reference (default: inf). "
            "If not available, falls back to the level with best median PSNR."
        ),
    )
    return parser.parse_args()


def _rewrite_label(content: str, new_label: str) -> str:
    lines = []
    for line in content.splitlines():
        if line.strip().startswith("\\label{"):
            lines.append(f"\\label{{{new_label}}}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _copy_with_label(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        print(f"WARN missing table: {src.name}")
        return
    content = src.read_text(encoding="utf-8")
    dst.write_text(_rewrite_label(content, label), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    tables_dir = root / "paper_assets" / "tables"
    output_dir = root / "docs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous best outputs
    for old in output_dir.glob("tab_*_best.tex"):
        old.unlink(missing_ok=True)

    df = load_combined()
    if df.empty:
        print("No data loaded. Skipping selection.")
        return

    best_noise = _select_noise(df, preferred=str(args.noise))
    best_entropy = _best_entropy_window(df)
    suffix = best_noise.replace("inf", "gap_only")

    selection = {
        f"tab1_global_{suffix}.tex": (
            "tab_global_best.tex",
            "tab:global-scoreboard-best",
        ),
        f"tab2_spectral_{suffix}.tex": (
            "tab_spectral_best.tex",
            "tab:spectral-rmse-best",
        ),
        f"tab3_degradation_entropy{best_entropy}.tex": (
            "tab_degradation_best.tex",
            "tab:degradation-psnr-best",
        ),
        f"tab3_slope_entropy{best_entropy}.tex": (
            "tab_slope_best.tex",
            "tab:noise-slope-psnr-best",
        ),
        f"tab4_spearman_entropy{best_entropy}.tex": (
            "tab_spearman_best.tex",
            "tab:spearman-entropy-best",
        ),
        "tab5_speed.tex": (
            "tab_speed_best.tex",
            "tab:runtime-speed",
        ),
    }

    copied = 0
    for src_name, (dst_name, label) in selection.items():
        src = tables_dir / src_name
        dst = output_dir / dst_name
        if src.exists():
            copied += 1
        _copy_with_label(src, dst, label)

    print("Selected noise:", best_noise)
    print("Selected entropy window:", best_entropy)
    print("Copied", copied, "of", len(selection), "tables")


if __name__ == "__main__":
    main()
