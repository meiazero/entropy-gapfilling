"""No-op helper kept for Makefile compatibility in a PDF-only paper flow."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No additional paper figures are required "
        + "beyond the PDF set."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("No extra figures generated.")


if __name__ == "__main__":
    main()
