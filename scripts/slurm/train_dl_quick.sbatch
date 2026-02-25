#!/bin/bash
# ---------------------------------------------------------------------------
# train_dl_quick.sbatch - Quick validation of all 5 DL models.
#
# Uses quick_validation.yaml (2 epochs, batch_size 4, 2 patches) for fast
# smoke testing. Trains all models sequentially in a single short job.
# Useful to verify the training pipeline before submitting full runs.
# ---------------------------------------------------------------------------
#SBATCH -J pdi-train-dl-quick
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

CONFIG="config/quick_validation.yaml"

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

echo "=== Quick DL Training (all models, 2 epochs) ==="
for model in ae vae gan unet vit; do
    echo "--- Training ${model} ---"
    python scripts/slurm/train_model.py --config "$CONFIG" --model "$model"
done

DL_RESULTS_DIR=$(python -c "
import yaml, pathlib
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
ckpt_dir = pathlib.Path(cfg.get('dl_training', {}).get('output_dir', 'results/dl_models/checkpoints'))
print(ckpt_dir.parent if ckpt_dir.name == 'checkpoints' else ckpt_dir)
")

echo "=== Plotting DL training curves ==="
HIST_FILES=$(ls "$DL_RESULTS_DIR"/*_history.json 2>/dev/null || true)
if [ -n "$HIST_FILES" ]; then
    mkdir -p results/dl_plots
    python -m dl_models.plot_training \
        --history $HIST_FILES \
        --output results/dl_plots
fi

echo "=== Generating DL Figures (results: $DL_RESULTS_DIR) ==="
python scripts/generate_figures.py --dl-results "$DL_RESULTS_DIR"
