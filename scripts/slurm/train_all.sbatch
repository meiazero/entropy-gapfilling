#!/bin/bash
#SBATCH -J pdi-train-all
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdi312

# Prevent pip from reading user config (which may contain 'user = true') and
# prevent user site-packages (~/.local) from being added to sys.path.
# Without these, torch 2.5.x from ~/.local could shadow the pinned 2.3.x.
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

export PDI_DATA_ROOT="${PDI_DATA_ROOT:-}"
SATELLITE="${SATELLITE:-sentinel2}"
MANIFEST="${MANIFEST:-preprocessed/manifest.csv}"
CKPT_DIR="${CKPT_DIR:-dl_models/checkpoints}"

AE_EPOCHS="${AE_EPOCHS:-50}"
VAE_EPOCHS="${VAE_EPOCHS:-60}"
GAN_EPOCHS="${GAN_EPOCHS:-100}"
UNET_EPOCHS="${UNET_EPOCHS:-60}"
VIT_EPOCHS="${VIT_EPOCHS:-100}"

AE_BATCH="${AE_BATCH:-32}"
VAE_BATCH="${VAE_BATCH:-32}"
GAN_BATCH="${GAN_BATCH:-16}"
UNET_BATCH="${UNET_BATCH:-32}"
VIT_BATCH="${VIT_BATCH:-32}"

AE_LR="${AE_LR:-1e-3}"
VAE_LR="${VAE_LR:-1e-3}"
GAN_LR="${GAN_LR:-2e-4}"
UNET_LR="${UNET_LR:-1e-3}"
VIT_LR="${VIT_LR:-1e-4}"

VAE_BETA="${VAE_BETA:-0.001}"
GAN_LAMBDA_L1="${GAN_LAMBDA_L1:-10.0}"
GAN_LAMBDA_ADV="${GAN_LAMBDA_ADV:-0.1}"
UNET_WD="${UNET_WD:-1e-4}"
VIT_WD="${VIT_WD:-0.05}"

AE_PATIENCE="${AE_PATIENCE:-10}"
VAE_PATIENCE="${VAE_PATIENCE:-10}"
GAN_PATIENCE="${GAN_PATIENCE:-15}"
UNET_PATIENCE="${UNET_PATIENCE:-12}"
VIT_PATIENCE="${VIT_PATIENCE:-15}"

DATA_ROOT_ARG=()
if [[ -n "${PDI_DATA_ROOT:-}" ]]; then
    DATA_ROOT_ARG=(--data-root "$PDI_DATA_ROOT")
fi

echo "=== Environment ==="
python --version
nvidia-smi

python -m pip install -e .

set +e
python - <<'PY'
import importlib
import sys

spec = importlib.util.find_spec("torch")
if spec is None:
    sys.exit(10)

try:
    import torch
except Exception:
    sys.exit(11)

if not torch.cuda.is_available():
    sys.exit(12)
PY
TORCH_STATUS=$?
set -e

case $TORCH_STATUS in
    10|11|12)
        python -m pip install --upgrade \
            "torch==2.3.1+cu121" \
            "torchvision==0.18.1+cu121" \
            --index-url https://download.pytorch.org/whl/cu121
        ;;
    0)
        ;;
esac

# Preprocess (resume-safe). Remove if you already have the NPY cache.
python scripts/preprocess_dataset.py --resume "${DATA_ROOT_ARG[@]}"

mkdir -p "$CKPT_DIR"

python -m dl_models.ae.train \
    --manifest "$MANIFEST" --satellite "$SATELLITE" \
    --output "$CKPT_DIR/ae_best.pth" \
    --epochs "$AE_EPOCHS" --batch-size "$AE_BATCH" --lr "$AE_LR" \
    --patience "$AE_PATIENCE" --device cuda

python -m dl_models.vae.train \
    --manifest "$MANIFEST" --satellite "$SATELLITE" \
    --output "$CKPT_DIR/vae_best.pth" \
    --epochs "$VAE_EPOCHS" --batch-size "$VAE_BATCH" --lr "$VAE_LR" \
    --beta "$VAE_BETA" --patience "$VAE_PATIENCE" --device cuda

python -m dl_models.gan.train \
    --manifest "$MANIFEST" --satellite "$SATELLITE" \
    --output "$CKPT_DIR/gan_best.pth" \
    --epochs "$GAN_EPOCHS" --batch-size "$GAN_BATCH" --lr "$GAN_LR" \
    --lambda-l1 "$GAN_LAMBDA_L1" --lambda-adv "$GAN_LAMBDA_ADV" \
    --patience "$GAN_PATIENCE" --device cuda

python -m dl_models.unet.train \
    --manifest "$MANIFEST" --satellite "$SATELLITE" \
    --output "$CKPT_DIR/unet_best.pth" \
    --epochs "$UNET_EPOCHS" --batch-size "$UNET_BATCH" --lr "$UNET_LR" \
    --weight-decay "$UNET_WD" --patience "$UNET_PATIENCE" --device cuda

python -m dl_models.vit.train \
    --manifest "$MANIFEST" --satellite "$SATELLITE" \
    --output "$CKPT_DIR/vit_best.pth" \
    --epochs "$VIT_EPOCHS" --batch-size "$VIT_BATCH" --lr "$VIT_LR" \
    --weight-decay "$VIT_WD" --patience "$VIT_PATIENCE" --device cuda
