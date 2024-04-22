#!/bin/bash
set -e -x

source activate ds_py

# Env should passed by caller.
export AS_RELEASE_VERSION="1.0.2"
repo_root=/root/workspace/DashInfer

# 捕获arch命令的输出
architecture=$(arch)

# 使用if-else结构进行条件判断
if [ "${architecture}" == "aarch64" ]; then
    export PLAT=manylinux_2_28_aarch64
    export AS_PLATFORM=armclang
else
    export PLAT=manylinux2014_x86_64
    export AS_PLATFORM=x86
fi

if [ -z "$PLAT" ] || [ -z "$AS_PLATFORM" ];
then
	echo " please set PLAT and AS_PLATFORM  env, PLAT can be manylinux_2_28_aarch64 or manylinux2014_x86_64"
	exit 1
fi

export AS_PYTHON_MANYLINUX=ON

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        # TODO: add lib path to build lib path
        auditwheel repair "$wheel" --plat "$PLAT" -w ${repo_root}/python/wheelhouse/
    fi
}

build_wheel_for_python() {
    local python_version=$1
    local env_name="py${python_version//.}env"

    if ! conda env list | grep "^${env_name}\s" > /dev/null; then
        echo "Conda environment '${env_name}' does not exist. Creating..."
        conda create -n "$env_name" python="$python_version" -y
    else
        echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
    fi

    conda activate "$env_name"
    conda install pybind11 -y

    pip install -r ${repo_root}/python/dev-requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    python ${repo_root}/python/setup.py bdist_wheel
    pip wheel ${repo_root}/python --no-deps -w ${repo_root}/python/wheelhouse/ --log wheel_log.txt

    conda deactivate
    conda remove --name "$env_name" --all -y
}

# ccache for faster build
yum install -y atlas-devel
pip3 install auditwheel

# rm -rf build

mkdir -p ${repo_root}/python/wheelhouse/

build_wheel_for_python 3.8
build_wheel_for_python 3.9
build_wheel_for_python 3.10
build_wheel_for_python 3.11

# Bundle external shared libraries into the wheels
for whl in ${repo_root}/python/wheelhouse/*.whl; do
    repair_wheel "$whl"
done

echo "Build finished, please check in wheelhouse/* for manylinux whl package"
