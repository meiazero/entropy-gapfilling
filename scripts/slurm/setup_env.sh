#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_env.sh - Configure the conda environment on the cluster.
#
# Target cluster:
#   OS:  Rocky Linux 9.5
#   CPU: AMD EPYC 7513 (2x 32-core, 2.6 GHz)
#   GPU: 2x NVIDIA A100 80 GB PCIe (compute capability 8.0)
#   RAM: 512 GB DDR4-3200
#   Interconnect: InfiniBand HDR100
#
# Usage:
#   bash scripts/slurm/setup_env.sh [REPO_DIR]
#
# REPO_DIR defaults to the current working directory.
#
# Module names can be overridden via environment variables before calling
# this script, e.g.:
#   CONDA_MODULE=miniconda3/23.x.y bash scripts/slurm/setup_env.sh
#   Run 'module avail miniconda3' and 'module avail cuda' on the cluster
#   to find the exact names if the defaults below do not match.
# ---------------------------------------------------------------------------

REPO_DIR="${1:-$PWD}"
ENV_NAME="pdi312"

CONDA_MODULE="${CONDA_MODULE:-miniconda3/py312_25.1.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.2}"

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
module load "${CONDA_MODULE}" || {
    echo "ERROR: Cannot load module '${CONDA_MODULE}'."
    echo "       Run: module avail miniconda3"
    echo "       Then re-run: CONDA_MODULE=<name> bash $0"
    exit 1
}

module load "${CUDA_MODULE}" || {
    echo "ERROR: Cannot load module '${CUDA_MODULE}'."
    echo "       Run: module avail cuda"
    echo "       Then re-run: CUDA_MODULE=<name> bash $0"
    exit 1
}

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# Create environment if it does not exist
# ---------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.12 -y
fi

conda activate "${ENV_NAME}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
    echo "ERROR: Failed to activate conda environment '${ENV_NAME}'."
    exit 1
fi

# Prevent pip from installing into the user home directory.
export PIP_NO_USER=1
export PIP_USER=0
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# Install project dependencies
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"
python -m pip install -e .

# Install CUDA-enabled PyTorch.
# pyproject.toml constrains torch>=2.2,<2.4, so 2.3.x is the latest allowed
# release. Official PyTorch 2.3.x wheels are published against CUDA 12.1
# (cu121). The NVIDIA driver on this cluster reports CUDA 12.4 (nvidia-smi),
# so cu121 binaries are compatible (12.1 <= 12.4). The loaded CUDA toolkit is
# 12.6.2 (nvcc), but the runtime constraint is determined by the driver (12.4).
# The A100 (sm_80) is fully supported by all CUDA 12.x toolchains.
python -m pip install --upgrade \
    "torch==2.3.1+cu121" \
    "torchvision==0.18.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# Verify the installation
# ---------------------------------------------------------------------------
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)

if cuda_ok:
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name}  (capability {cap[0]}.{cap[1]})")
else:
    print("WARNING: CUDA is not available in this session.")
    print("         Submit via the 'gpuq' partition to reach a GPU node.")
PY

echo "Environment ready: $(python --version)"
