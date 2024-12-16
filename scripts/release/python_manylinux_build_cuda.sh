#!/bin/bash
set -e -x

ALL_VERSION="3.8 3.9 3.10 3.11"
BUILD_VERSION=${@:-$ALL_VERSION}
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')

echo " going to build python wheels with version: ${BUILD_VERSION}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( dirname -- "$( dirname -- "${SCRIPT_DIR}" )" )

source activate ds_py
pushd $SCRIPT_DIR

# 捕获arch命令的输出
architecture=$(arch)

export PLAT=manylinux2014_x86_64
export AS_PLATFORM=cuda

mkdir -p local_cuda_libs
ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so local_cuda_libs/libnvidia-ml.so.1
ln -sf /usr/local/cuda-${CUDA_VERSION}/compat/libcuda.so.1 local_cuda_libs/libcuda.so.1
export LD_LIBRARY_PATH=${PWD}/local_cuda_libs:${LD_LIBRARY_PATH}

if [ -z "$PLAT" ] || [ -z "$AS_PLATFORM" ];
then
	echo " please set PLAT and AS_PLATFORM  env, PLAT can be manylinux_2_28_aarch64 or manylinux2014_x86_64"
	exit 1
fi

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        # TODO: add lib path to build lib path
        auditwheel repair "$wheel" --plat "$PLAT" -w ${REPO_ROOT}/python/wheelhouse/
    fi
}

build_wheel_for_python() {
    local python_version=$1
    local env_name="py${python_version//.}env"

    if ! conda env list | grep "^${env_name}\s" > /dev/null; then
        echo "Conda environment '${env_name}' does not exist. Creating..."
        conda create -n "$env_name" python="$python_version" -y
    else
        echo "Conda environment '${env_name}' already exists. Skipping creation."
    fi

    conda activate "$env_name"
    conda install pybind11 -y

    pip install -r ${REPO_ROOT}/python/requirements_dev_cuda.txt -i https://mirrors.aliyun.com/pypi/simple/
    ln -sf ${REPO_ROOT}/python/dashinfer .
    # python ${REPO_ROOT}/python/setup.py bdist_wheel
    pip wheel ${REPO_ROOT}/python --no-deps -w ${REPO_ROOT}/python/wheelhouse/ --verbose

    conda deactivate
    # conda remove --name "$env_name" --all -y
}

# rm -rf build

mkdir -p ${REPO_ROOT}/python/wheelhouse/

for python_version in $BUILD_VERSION; do
    build_wheel_for_python ${python_version}  2>&1 | tee wheel_build_log_py${python_version//.}.txt
done


# Bundle external shared libraries into the wheels
for whl in ${REPO_ROOT}/python/wheelhouse/*.whl; do
    repair_wheel "$whl"
done

echo "Build finished, please check in wheelhouse/* for manylinux whl package"
popd
