#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"
ENV_NAME="pdi312"

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.12 -y
fi

conda activate "${ENV_NAME}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    echo "ERROR: Failed to activate conda environment '${ENV_NAME}'."
    exit 1
fi

export PIP_NO_USER=1

python -m pip install --upgrade pip

cd "$REPO_DIR"
python -m pip install -e .

# Ensure CUDA-enabled PyTorch is installed for GPU training.
python -m pip install --upgrade \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch

print("Python:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Environment ready: $(python --version)"
