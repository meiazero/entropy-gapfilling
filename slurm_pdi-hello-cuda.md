[emanuel.pires@apollo entropy-gapfilling]$ cat slurm_pdi-hello-cuda_1881.out
=== Python ===
Python 3.12.12
/home/emanuel.pires/.conda/envs/pdi312/bin/python
sys.executable: /home/emanuel.pires/.conda/envs/pdi312/bin/python
sys.version: 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:16:04) [GCC 11.2.0]
=== Conda envs ===

# conda environments:

#

pdi312 \* /home/emanuel.pires/.conda/envs/pdi312
base /softwares/miniconda3/py312_25.1.1

     active environment : pdi312
    active env location : /home/emanuel.pires/.conda/envs/pdi312
            shell level : 2
       user config file : /home/emanuel.pires/.condarc

populated config files : /softwares/miniconda3/py312_25.1.1/.condarc
conda version : 25.1.1
conda-build version : not installed
python version : 3.12.9.final.0
solver : libmamba (default)
virtual packages : **archspec=1=zen3
**conda=25.1.1=0
**cuda=12.4=0
**glibc=2.34=0
**linux=5.14.0=0
**unix=0=0
base environment : /softwares/miniconda3/py312_25.1.1 (read only)
conda av data dir : /softwares/miniconda3/py312_25.1.1/etc/conda
conda av metadata url : None
channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
https://repo.anaconda.com/pkgs/main/noarch
https://repo.anaconda.com/pkgs/r/linux-64
https://repo.anaconda.com/pkgs/r/noarch
package cache : /softwares/miniconda3/py312_25.1.1/pkgs
/home/emanuel.pires/.conda/pkgs
envs directories : /home/emanuel.pires/.conda/envs
/softwares/miniconda3/py312_25.1.1/envs
platform : linux-64
user-agent : conda/25.1.1 requests/2.32.3 CPython/3.12.9 Linux/5.14.0-503.14.1.el9_5.x86_64 rocky/9.5 glibc/2.34 solver/libmamba conda-libmamba-solver/25.1.1 libmambapy/2.0.5 aau/0.5.0 c/. s/. e/.
UID:GID : 1074:1074
netrc file : None
offline mode : False

# conda environments:

#

pdi312 \* /home/emanuel.pires/.conda/envs/pdi312
base /softwares/miniconda3/py312_25.1.1

sys.version: 3.12.9 | packaged by Anaconda, Inc. | (m...
sys.prefix: /softwares/miniconda3/py312_25.1.1
sys.executable: /softwares/miniconda3/py312_25.1.1/bin/python
conda location: /softwares/miniconda3/py312_25.1.1/lib/python3.12/site-packages/conda
conda-build: None
conda-content-trust: /softwares/miniconda3/py312_25.1.1/bin/conda-content-trust
conda-env: /softwares/miniconda3/py312_25.1.1/bin/conda-env
user site dirs: ~/.local/lib/python3.12

ACLOCAL_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/share/aclocal:/usr/share/aclocal
CIO_TEST: <not set>
CMAKE_PREFIX_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/gcc-runtime-11.5.0-6ukpvpt6ul7zicuo5huj5pwxjpulekhy
CONDA_DEFAULT_ENV: pdi312
CONDA_EXE: /softwares/miniconda3/py312_25.1.1/bin/conda
CONDA_PREFIX: /home/emanuel.pires/.conda/envs/pdi312
CONDA_PREFIX_1: /softwares/miniconda3/py312_25.1.1
CONDA_PROMPT_MODIFIER: (pdi312)
CONDA_PYTHON_EXE: /softwares/miniconda3/py312_25.1.1/bin/python
CONDA_ROOT: /softwares/miniconda3/py312_25.1.1
CONDA_SHLVL: 2
CURL_CA_BUNDLE: <not set>
DEBUGINFOD_IMA_CERT_PATH: /etc/keys/ima:
FPATH: /usr/share/lmod/lmod/init/ksh_funcs
LD_LIBRARY_PATH: /softwares/miniconda3/py312_25.1.1/lib
LD_PRELOAD: <not set>
MANPATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/share/man:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf/share/man:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/share/man:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy/share/man:/usr/share/man:/softwares/miniconda3/py312_25.1.1/share/man:/usr/share/lmod/lmod/share/man:/opt/clmgr/man:/opt/sgi/share/man:/opt/clmgr/share/man:/opt/clmgr/lib/cm-cli/man::
MODULEPATH: /softwares/spack/modulefiles/mvapich2/2.3.7-1-vuvlmfb/Core:/softwares/spack/modulefiles/Core:/softwares/modulefiles:/etc/modulefiles:/usr/share/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core
PATH: /home/emanuel.pires/.conda/envs/pdi312/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy/bin:/softwares/miniconda3/py312_25.1.1/bin:/softwares/miniconda3/py312_25.1.1/condabin:/home/emanuel.pires/.local/bin:/home/emanuel.pires/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/bin:/usr/bin:/usr/local/bin:/sbin:/usr/local/sbin:/usr/sbin:/opt/c3/bin
PKG_CONFIG_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/lib/pkgconfig:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf/lib/pkgconfig:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/lib/pkgconfig:/usr/share/pkgconfig:/usr/lib64/pkgconfig
PYTHONPATH: /softwares/miniconda3/py312_25.1.1/bin
REQUESTS_CA_BUNDLE: <not set>
SSL_CERT_FILE: <not set>
**LMOD_REF_COUNT_ACLOCAL_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/share/aclocal:1;/usr/share/aclocal:1
**LMOD_REF_COUNT_CMAKE_PREFIX_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/gcc-runtime-11.5.0-6ukpvpt6ul7zicuo5huj5pwxjpulekhy:1
**LMOD_REF_COUNT_LD_LIBRARY_PATH: /softwares/miniconda3/py312_25.1.1/lib:1
**LMOD_REF_COUNT_MANPATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/share/man:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf/share/man:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/share/man:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy/share/man:1;/usr/share/man:1;/softwares/miniconda3/py312_25.1.1/share/man:1;/usr/share/lmod/lmod/share/man:1;/opt/clmgr/man:1;/opt/sgi/share/man:1;/opt/clmgr/share/man:1;/opt/clmgr/lib/cm-cli/man:1;:5
**LMOD_REF_COUNT_MODULEPATH: /softwares/spack/modulefiles/mvapich2/2.3.7-1-vuvlmfb/Core:1;/softwares/spack/modulefiles/Core:1;/softwares/modulefiles:1;/etc/modulefiles:1;/usr/share/modulefiles:1;/usr/share/modulefiles/Linux:1;/usr/share/modulefiles/Core:1;/usr/share/lmod/lmod/modulefiles/Core:1
**LMOD_REF_COUNT_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah/bin:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/bin:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/bin:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy/bin:1;/softwares/miniconda3/py312_25.1.1/bin:1;/softwares/miniconda3/py312_25.1.1/condabin:1;/home/emanuel.pires/.local/bin:1;/home/emanuel.pires/bin:1;/opt/clmgr/sbin:1;/opt/clmgr/bin:1;/opt/sgi/sbin:1;/opt/sgi/bin:1;/bin:1;/usr/bin:1;/usr/local/bin:1;/sbin:1;/usr/local/sbin:1;/usr/sbin:1;/opt/c3/bin:1
**LMOD_REF_COUNT_PKG_CONFIG_PATH: /softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/lib/pkgconfig:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/zlib-ng-2.2.1-hfqhpdelqmqxzacwt75r34xpku5pmejf/lib/pkgconfig:1;/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/lib/pkgconfig:1;/usr/share/pkgconfig:1;/usr/lib64/pkgconfig:1
**LMOD_REF_COUNT_PYTHONPATH: /softwares/miniconda3/py312_25.1.1/bin:1
=== CUDA module === 4) libiconv/1.17-fgdoeld 8) cuda/12.6.2-hbmugou
=== CUDA toolchain ===
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
CUDA_HOME=/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah
CUDA_PATH=
LD_LIBRARY_PATH=/softwares/miniconda3/py312_25.1.1/lib
PATH=/home/emanuel.pires/.conda/envs/pdi312/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/cuda-12.6.2-hbmugoui54orn2dhx4m33nsmfpmtboah/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libxml2-2.13.4-uehhn5mfs6wewahxsi7yjadwts7h66l4/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/xz-5.4.6-fctvfzrvlzn4qdegpvfzm6pzdewax6ch/bin:/softwares/spack/packages/linux-rocky9-zen3/gcc-11.5.0/libiconv-1.17-fgdoelds5of3jx4qg5tfd737gbl56kyy/bin:/softwares/miniconda3/py312_25.1.1/bin:/softwares/miniconda3/py312_25.1.1/condabin:/home/emanuel.pires/.local/bin:/home/emanuel.pires/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/bin:/usr/bin:/usr/local/bin:/sbin:/usr/local/sbin:/usr/sbin:/opt/c3/bin
=== NVIDIA SMI ===
Mon Feb 23 23:38:20 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07 Driver Version: 550.90.07 CUDA Version: 12.4 |
|-----------------------------------------+------------------------+----------------------+
| GPU Name Persistence-M | Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap | Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|=========================================+========================+======================|
| 0 NVIDIA A100 80GB PCIe Off | 00000000:C3:00.0 Off | 0 |
| N/A 48C P0 70W / 300W | 1MiB / 81920MiB | 18% Default |
| | | Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes: |
| GPU GI CI PID Type Process name GPU Memory |
| ID ID Usage |
|=========================================================================================|
| No running processes found |
+-----------------------------------------------------------------------------------------+
=== PyTorch CUDA Check ===
torch.**version**: 2.3.1+cu121
torch.version.cuda: 12.1
torch.backends.cudnn.version: 8902
cuda.is_available: True
cuda.device_count: 1
cuda.device_name[0]: NVIDIA A100 80GB PCIe
cuda.capability[0]: (8, 0)
cuda.mem_allocated: 0
cuda.mem_reserved: 0
=== Pip summary ===
pip 26.0.1 from /home/emanuel.pires/.local/lib/python3.12/site-packages/pip (python 3.12)
Name: torch
Version: 2.5.1+cu121
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: /home/emanuel.pires/.local/lib/python3.12/site-packages
Requires: filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-nccl-cu12, nvidia-nvtx-cu12, setuptools, sympy, triton, typing-extensions
Required-by: pdi-entropy-gapfilling, torchvision

---

Name: torchvision
Version: 0.18.1
Summary: image and video datasets and models for torch deep learning
Home-page: https://github.com/pytorch/vision
Author: PyTorch Core Team
Author-email: soumith@pytorch.org
License: BSD
Location: /home/emanuel.pires/.local/lib/python3.12/site-packages
Requires: numpy, pillow, torch
Required-by: pdi-entropy-gapfilling
