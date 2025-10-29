# Installing `transformer-engine[pytorch]` with GPU support

These steps document the environment configuration that allowed `uv pip install 'transformer-engine[pytorch]'` to succeed in the `cp4` conda environment on 2025-10-22.

## 1. Install CUDA developer wheels via conda or pip

- Preferred (conda-forge):
  ```
  conda install -c conda-forge \
      cuda-cccl cuda-nvcc cuda-cudart-dev \
      cuda-cublas-dev cuda-curand-dev \
      cuda-cusparse-dev cuda-cusolver-dev \
      cuda-cusparselt-dev cuda-cufft-dev \
      cuda-nvtx-dev cuda-nvjitlink-dev \
      cuda-nvshmem-dev
  ```
- Pip alternative (if conda packages are unavailable):
  ```
  pip install \
      nvidia-cuda-cccl-cu12 nvidia-cuda-runtime-cu12 \
      nvidia-cublas-cu12 nvidia-curand-cu12 \
      nvidia-cusparse-cu12 nvidia-cusolver-cu12 \
      nvidia-cusparselt-cu12 nvidia-cufft-cu12 \
      nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
      nvidia-nvshmem-cu12 nvidia-cudnn-cu12
  ```

## 2. Export build environment variables

```bash
export CONDA_PREFIX=/home/chaichontat/miniforge3/envs/cp4
NV_SITE="$CONDA_PREFIX/lib/python3.13/site-packages"
CUDA_HOME="$NV_SITE/nvidia/cuda_runtime"
CUDA_INC="$CUDA_HOME/include"
CUDA_LIB="$CUDA_HOME/lib"
CCCL_INC="$NV_SITE/nvidia/cuda_cccl/include"

export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export NVTE_CUDA_INCLUDE_DIR="$CCCL_INC"
export NVTE_CUDA_LIB_DIR="$CUDA_LIB"
export NVTE_CUDA_ARCHS="70;80;89;90;100;120"

extra_inc="$NV_SITE/nvidia/cublas/include:\
$NV_SITE/nvidia/cudnn/include:\
$NV_SITE/nvidia/cusparse/include:\
$NV_SITE/nvidia/cusolver/include:\
$NV_SITE/nvidia/cusparselt/include:\
$NV_SITE/nvidia/cufft/include:\
$NV_SITE/nvidia/curand/include:\
$NV_SITE/nvidia/cuda_nvrtc/include:\
$NV_SITE/nvidia/nvtx/include:\
$NV_SITE/nvidia/nccl/include:\
$NV_SITE/triton/backends/nvidia/include"

extra_lib="$CUDA_LIB:\
$NV_SITE/nvidia/cublas/lib:\
$NV_SITE/nvidia/cudnn/lib:\
$NV_SITE/nvidia/cusparse/lib:\
$NV_SITE/nvidia/cusolver/lib:\
$NV_SITE/nvidia/cusparselt/lib:\
$NV_SITE/nvidia/cufft/lib:\
$NV_SITE/nvidia/curand/lib:\
$NV_SITE/nvidia/cuda_nvrtc/lib:\
$NV_SITE/nvidia/nvtx/lib:\
$NV_SITE/nvidia/nccl/lib:\
$NV_SITE/nvidia/nvjitlink/lib:\
$NV_SITE/nvidia/nvshmem/lib"

export CPATH="$CCCL_INC:$CUDA_INC:$extra_inc:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CPATH"
export LIBRARY_PATH="$extra_lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$extra_lib:${LD_LIBRARY_PATH:-}"
```

## 3. Install with uv

Ensure the same shell session has the environment variables from §2 exported, then run:

```bash
mkdir -p .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

uv pip install 'transformer-engine[pytorch]'
```

`uv` resolves the CUDA-enabled `transformer-engine` wheels without building from source, provided the CUDA developer libraries from §1 are available. After installation, confirm the resolved packages:

```bash
uv pip show transformer-engine
uv pip show transformer-engine-cu12
uv pip show transformer-engine-torch
```

All three should report version `2.8.0` under `/home/chaichontat/miniforge3/envs/cp4/lib/python3.13/site-packages`.
