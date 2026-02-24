#!/bin/bash
# ---------------------------------------------------------------------------
# train_ae.sbatch - Train the Autoencoder model on a GPU node.
#
# Reads hyperparameters from the YAML config (dl_training section).
# Override the config file via PDI_CONFIG:
#   PDI_CONFIG=config/quick_validation.yaml sbatch scripts/slurm/train_ae.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J pdi-train-ae
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

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
nvidia-smi

python -m pip install -e .

set +e
python -c "import torch; assert torch.cuda.is_available()"
TORCH_STATUS=$?
set -e

if [[ $TORCH_STATUS -ne 0 ]]; then
    python -m pip install --upgrade \
        "torch==2.3.1+cu121" \
        "torchvision==0.18.1+cu121" \
        --index-url https://download.pytorch.org/whl/cu121
fi

echo "=== Training AE ==="
python scripts/slurm/train_model.py --config "$CONFIG" --model ae
