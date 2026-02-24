#!/bin/bash
#SBATCH -J pdi-preprocess
#SBATCH -p cpuq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdi312

# Prevent pip from reading or writing to the user home directory.
export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export PIP_USER=0

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

DATA_ROOT_ARG=()
if [[ -n "${PDI_DATA_ROOT:-}" ]]; then
    DATA_ROOT_ARG=(--data-root "$PDI_DATA_ROOT")
fi

echo "=== Preprocess ==="
python --version
python -m pip install -e .
python scripts/preprocess_dataset.py --resume "${DATA_ROOT_ARG[@]}"
