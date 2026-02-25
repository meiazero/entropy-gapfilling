#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh - Submit the full DL pipeline as independent SLURM jobs.
#
# Dependency chain:
#   preprocess -> [ae, vae, gan, unet, vit in parallel]
#
# Usage (from the repo root or login node):
#   bash scripts/slurm/submit_all.sh
#   PDI_CONFIG=config/quick_validation.yaml bash scripts/slurm/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONFIG="${PDI_CONFIG:-config/paper_results.yaml}"

mkdir -p "$REPO_DIR/logs"

echo "=== PDI Pipeline ==="
echo "Repo:   $REPO_DIR"
echo "Config: $CONFIG"
echo ""

PREP_JID=$(REPO_DIR="$REPO_DIR" sbatch --parsable "$SCRIPT_DIR/preprocess.sbatch")
echo "Submitted preprocess:  job $PREP_JID"

for model in ae vae gan unet vit; do
    JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
        --parsable \
        --dependency=afterok:"$PREP_JID" \
        "$SCRIPT_DIR/train_${model}.sbatch")
    echo "Submitted train-${model}: job $JID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
