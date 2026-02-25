#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh - Submit the full DL pipeline as independent SLURM jobs.
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

AE_JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
    --parsable \
    "$SCRIPT_DIR/train_ae.sbatch")
echo "Submitted train-ae:    job $AE_JID"

VAE_JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
    --parsable \
    "$SCRIPT_DIR/train_vae.sbatch")
echo "Submitted train-vae:   job $VAE_JID"

GAN_JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
    --parsable \
    "$SCRIPT_DIR/train_gan.sbatch")
echo "Submitted train-gan:   job $GAN_JID"

UNET_JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
    --parsable \
    "$SCRIPT_DIR/train_unet.sbatch")
echo "Submitted train-unet:  job $UNET_JID"

VIT_JID=$(REPO_DIR="$REPO_DIR" PDI_CONFIG="$CONFIG" sbatch \
    --parsable \
    "$SCRIPT_DIR/train_vit.sbatch")
echo "Submitted train-vit:   job $VIT_JID"

echo ""
echo "Monitor with: squeue -u \$USER"
