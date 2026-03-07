"""No-op helper kept for Makefile compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No additional paper figures are required."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_assets/figures"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("No extra figures generated.")


if __name__ == "__main__":
    main()
