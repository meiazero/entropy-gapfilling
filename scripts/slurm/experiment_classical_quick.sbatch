#!/bin/bash
# ---------------------------------------------------------------------------
# experiment_classical_quick.sbatch - Quick validation of classical methods.
#
# Uses quick_validation.yaml (1 seed, 2 patches) for fast smoke testing.
# Useful to verify the pipeline before submitting the full experiment.
# ---------------------------------------------------------------------------
#SBATCH -J pdi-experiment-classical-quick
#SBATCH -p cpuq
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdi312

export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

echo "=== Environment ==="
echo "Config: config/quick_validation.yaml"
python --version

python -m pip install -e .

echo "=== Preprocessing (resume-safe) ==="
DATA_ROOT_ARG=()
if [[ -n "${PDI_DATA_ROOT:-}" ]]; then
    DATA_ROOT_ARG=(--data-root "$PDI_DATA_ROOT")
fi
python scripts/preprocess_dataset.py --resume "${DATA_ROOT_ARG[@]}"

echo "=== Classical Experiment (quick) ==="
python scripts/run_experiment.py --config config/quick_validation.yaml
