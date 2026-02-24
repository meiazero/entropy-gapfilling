#!/bin/bash
# ---------------------------------------------------------------------------
# experiment_classical.sbatch - Run the full classical gap-filling experiment.
#
# Executes all 15 classical interpolation methods across all satellites,
# noise levels, and seeds defined in the config YAML. This is a CPU-bound
# job (no GPU required).
#
# Override the config file via PDI_CONFIG:
#   PDI_CONFIG=config/quick_validation.yaml sbatch scripts/slurm/experiment_classical.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J pdi-experiment-classical
#SBATCH -p cpuq
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdi312

export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

CONFIG="${PDI_CONFIG:-config/paper_results.yaml}"

echo "=== Environment ==="
echo "Config: $CONFIG"
python --version

python -m pip install --no-build-isolation -e .

echo "=== Preprocessing (resume-safe) ==="
MANIFEST_PATH="$REPO_DIR/preprocessed/manifest.csv"
if [[ -f "$MANIFEST_PATH" ]]; then
    echo "Manifest already exists at $MANIFEST_PATH - skipping preprocessing."
else
    DATA_ROOT_ARG=()
    if [[ -n "${PDI_DATA_ROOT:-}" ]]; then
        DATA_ROOT_ARG=(--data-root "$PDI_DATA_ROOT")
    fi
    python scripts/preprocess_dataset.py --resume "${DATA_ROOT_ARG[@]}"
fi

echo "=== Classical Experiment ==="
python scripts/run_experiment.py --config "$CONFIG"
