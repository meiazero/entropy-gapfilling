"""Alias script for the PSNR entropy boxplot figure."""

from __future__ import annotations

from figures.cli import run_with_df
from figures.psnr_entropy import fig6_psnr_entropy


def main() -> None:
    run_with_df(fig6_psnr_entropy, "PSNR boxplot by entropy")


if __name__ == "__main__":
    main()
