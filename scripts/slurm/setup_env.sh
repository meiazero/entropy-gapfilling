#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx pdi312; then
    conda create -n pdi312 python=3.12 -y
fi

conda activate pdi312
pip install --upgrade pip

cd "$REPO_DIR"
pip install -e .

# Ensure CUDA-enabled PyTorch is installed for GPU training.
pip install --upgrade --force-reinstall \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Environment ready: $(python --version)"
