#!/bin/bash
# ---------------------------------------------------------------------------
# hello_cuda.sbatch - Cluster diagnostic / environment probe.
#
# Intentionally does NOT use set -e so every section runs regardless of
# individual command failures. The goal is to collect enough output to
# configure setup_env.sh for this cluster.
#
# Before submitting, verify the partition name and GPU resource spec:
#   sinfo -o "%P %a %G"    <- lists partitions and GRES available
#
# If 'gpuq' or 'gpu:a100:1' are wrong, override at submission time:
#   sbatch -p <partition> --gres=gpu:1 scripts/slurm/hello_cuda.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J pdi-hello-cuda
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

SEP() { printf '\n=== %s ===\n' "$*"; }

SEP "SLURM environment"
echo "HOST:      $(hostname -f)"
echo "NODE:      ${SLURMD_NODENAME:-unknown}"
echo "JOB_ID:    ${SLURM_JOB_ID:-unknown}"
echo "PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "DATE:      $(date --iso-8601=seconds)"

SEP "OS"
cat /etc/os-release

SEP "Kernel"
uname -a

SEP "CPU"
lscpu | grep -E "^(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|NUMA)" || lscpu

SEP "Memory"
free -h

SEP "SLURM partitions and GRES"
sinfo -o "%P %a %l %D %c %m %G" 2>&1 || echo "sinfo: not available"

# ---------------------------------------------------------------------------
# Module discovery - do not abort if a module is unavailable
# ---------------------------------------------------------------------------
SEP "Available modules - miniconda / anaconda"
module avail miniconda3 2>&1 || true
module avail anaconda3  2>&1 || true

SEP "Available modules - CUDA"
module avail cuda 2>&1 || true

SEP "Available modules - Python"
module avail python 2>&1 || true

SEP "Load modules (best-effort)"
CONDA_MODULE="${CONDA_MODULE:-miniconda3/py312_25.1.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.2}"

module load "${CONDA_MODULE}" 2>&1 \
    && echo "OK: loaded ${CONDA_MODULE}" \
    || echo "WARN: could not load '${CONDA_MODULE}'"

module load "${CUDA_MODULE}" 2>&1 \
    && echo "OK: loaded ${CUDA_MODULE}" \
    || echo "WARN: could not load '${CUDA_MODULE}'"

SEP "Currently loaded modules"
module list 2>&1

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
SEP "Conda"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda --version
    echo "--- envs ---"
    conda info --envs
    echo "--- info ---"
    conda info -a | head -40
else
    echo "conda: not found in PATH after module load"
fi

# ---------------------------------------------------------------------------
# CUDA toolchain
# ---------------------------------------------------------------------------
SEP "CUDA toolchain"
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc: not found"

SEP "CUDA / GPU environment variables"
python3 - 2>/dev/null <<'PY'
import os
for key in (
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH", "PATH",
):
    print(f"{key}={os.environ.get(key, '')}")
PY

SEP "nvidia-smi"
nvidia-smi 2>&1 || echo "nvidia-smi: not available"

SEP "nvidia-smi - GPU query"
nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
    --format=csv \
    2>&1 || true

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
SEP "Python (system)"
python3 --version 2>&1 || echo "python3: not found"
which python3 2>/dev/null || true

# ---------------------------------------------------------------------------
# PyTorch check
# ---------------------------------------------------------------------------
SEP "PyTorch CUDA check"
python3 - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
    print("Run setup_env.sh first to install torch.")
else:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
    avail = torch.cuda.is_available()
    print("cuda.is_available:", avail)
    if avail:
        print("cuda.device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            mem  = torch.cuda.get_device_properties(i).total_memory // 1024**3
            print(f"  GPU {i}: {name}  cap={cap[0]}.{cap[1]}  mem={mem} GB")
    else:
        print("WARNING: CUDA not available - check GPU allocation and driver.")
PY

SEP "pip - torch / torchvision"
python3 -m pip show torch torchvision 2>&1 || echo "pip show: torch/torchvision not installed"

echo ""
echo "--- diagnostic complete ---"
