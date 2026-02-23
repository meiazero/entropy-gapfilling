#!/bin/bash
#SBATCH -J pdi-hello-cuda
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
if conda env list | awk '{print $1}' | grep -qx pdi312; then
    conda activate pdi312
else
    echo "WARN: conda env 'pdi312' not found. Using base environment."
    echo "      Create it with: conda create -n pdi312 python=3.12 -y"
fi

echo "=== Python ==="
python --version
which python
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
PY

echo "=== Conda envs ==="
conda info --envs
conda info -a | sed -n '1,120p'

echo "=== CUDA module ==="
module list 2>&1 | grep -i cuda || true

echo "=== CUDA toolchain ==="
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc: not found"
python - <<'PY'
import os
for key in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"):
    print(f"{key}={os.environ.get(key, '')}")
PY

echo "=== NVIDIA SMI ==="
nvidia-smi

echo "=== PyTorch CUDA Check ==="
python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"WARN: torch import failed: {exc}")
    print("Install torch in the active environment to test CUDA.")
else:
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda.device_count:", torch.cuda.device_count())
        print("cuda.device_name[0]:", torch.cuda.get_device_name(0))
        print("cuda.capability[0]:", torch.cuda.get_device_capability(0))
        print("cuda.mem_allocated:", torch.cuda.memory_allocated(0))
        print("cuda.mem_reserved:", torch.cuda.memory_reserved(0))
PY

echo "=== Pip summary ==="
python -m pip --version
python -m pip show torch torchvision 2>/dev/null || true
