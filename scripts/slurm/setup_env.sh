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
export PIP_USER=0
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip

cd "$REPO_DIR"
python -m pip install -e .

# Ensure CUDA-enabled PyTorch is installed for GPU training and matches
# project constraints.
python -m pip install --upgrade --force-reinstall \
    "torch==2.3.1+cu121" \
    "torchvision==0.18.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch

print("Python:", torch.__version__)
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
if cuda_ok:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available in this session.")
PY

echo "Environment ready: $(python --version)"
