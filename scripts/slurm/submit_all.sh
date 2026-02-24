#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh - Submit all 5 DL training jobs to SLURM.
#
# Each model runs as an independent job and can execute in parallel if
# multiple GPUs are available.
#
# Usage:
#   bash scripts/slurm/submit_all.sh
#   PDI_CONFIG=config/quick_validation.yaml bash scripts/slurm/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for model in ae vae gan unet vit; do
    echo "Submitting ${model}..."
    sbatch "${SCRIPT_DIR}/train_${model}.sbatch"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
