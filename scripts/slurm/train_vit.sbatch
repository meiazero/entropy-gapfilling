#!/bin/bash
# ---------------------------------------------------------------------------
# train_vit.sbatch - Train the ViT (MAE-style) model on a GPU node.
#
# Reads hyperparameters from the YAML config (dl_training section).
# Override the config file via PDI_CONFIG:
#   PDI_CONFIG=config/quick_validation.yaml sbatch scripts/slurm/train_vit.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J pdi-train-vit
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdi312

export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

CONFIG="${PDI_CONFIG:-config/paper_results.yaml}"

echo "=== Environment ==="
echo "Config: $CONFIG"
python --version
nvidia-smi


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

echo "=== Training ViT ==="
python scripts/slurm/train_model.py --config "$CONFIG" --model vit

DL_RESULTS_DIR=$(python -c "
import yaml, pathlib
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
ckpt_dir = pathlib.Path(cfg.get('dl_training', {}).get('output_dir', 'dl_models/checkpoints'))
print(ckpt_dir.parent if ckpt_dir.name == 'checkpoints' else ckpt_dir)
")

echo "=== Generating DL Figures (results: $DL_RESULTS_DIR) ==="
python scripts/generate_figures.py --dl-results "$DL_RESULTS_DIR"

echo "=== Generating DL Tables (results: $DL_RESULTS_DIR) ==="
python scripts/generate_tables.py --dl-results "$DL_RESULTS_DIR"
