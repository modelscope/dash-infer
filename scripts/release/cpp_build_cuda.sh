#!/bin/bash
set -e -x

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')

mkdir -p local_cuda_libs
ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so local_cuda_libs/libnvidia-ml.so.1
ln -sf /usr/local/cuda-${CUDA_VERSION}/compat/libcuda.so.1 local_cuda_libs/libcuda.so.1
export LD_LIBRARY_PATH=${PWD}/local_cuda_libs:${LD_LIBRARY_PATH}

bash build.sh
