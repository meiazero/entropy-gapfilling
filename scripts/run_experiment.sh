#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR does not exist: $REPO_DIR" >&2
    exit 1
fi

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found in $REPO_DIR" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found in PATH. Install uv and retry." >&2
    exit 1
fi

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(--config "config/paper_results.yaml")
fi

CONFIG_PATH=""
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--config" ]]; then
        if (( i + 1 >= ${#ARGS[@]} )); then
            echo "ERROR: --config requires a path." >&2
            exit 1
        fi
        CONFIG_PATH="${ARGS[$((i + 1))]}"
    fi
done

if [[ -n "$CONFIG_PATH" && ! -f "$REPO_DIR/$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $REPO_DIR/$CONFIG_PATH" >&2
    exit 1
fi

MANIFEST="${MANIFEST:-preprocessed/manifest.csv}"
if [[ "${ALLOW_MISSING_MANIFEST:-0}" != "1" && ! -f "$REPO_DIR/$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $REPO_DIR/$MANIFEST" >&2
    echo "Run: uv run python scripts/preprocess_dataset.py --resume" >&2
    exit 1
fi

echo "Checking Python environment with uv..."
uv run python - <<'PY'
import importlib
import sys

if sys.version_info < (3, 12):
    raise SystemExit("Python >= 3.12 is required.")

missing = [
    name
    for name in ("numpy", "pandas", "tqdm", "pdi_pipeline")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(f"Missing modules: {', '.join(missing)}")
PY

echo "Starting experiment..."
cd "$REPO_DIR"
uv run python scripts/run_experiment.py "${ARGS[@]}"
