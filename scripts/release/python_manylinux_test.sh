#!/bin/bash
set -e -x

ALL_VERSION="3.8 3.9 3.10 3.11"
TEST_VERSION=${@:-$ALL_VERSION}

echo " going to test with python version: ${TEST_VERSION}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( dirname -- "$( dirname -- "${SCRIPT_DIR}" )" )

source activate test_py
pushd $SCRIPT_DIR

run_python_example() {
    local python_version=$1
    local env_name="py${python_version//.}env"

    if ! conda env list | grep "^${env_name}\s" > /dev/null; then
        echo "Conda environment '${env_name}' does not exist. Creating..."
        conda create -n "$env_name" python="$python_version" -y
    else
        echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
    fi

    conda activate "$env_name"
    pip uninstall dashinfer dashinfer-allspark -y

    # 捕获arch命令的输出
    architecture=$(arch)

    # 使用if-else结构进行条件判断
    if [ "${architecture}" == "x86_64" ]; then
        conda install -y pytorch-cpu -c pytorch
    fi

    pip install -r ${REPO_ROOT}/examples/python/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    pip install ${REPO_ROOT}/python/wheelhouse/dashinfer-${AS_RELEASE_VERSION}-cp${python_version//.}-cp${python_version//.}-manylinux*.whl

    cd ${REPO_ROOT}/examples/python/0_basic
    # python basic_example_chatglm2.py
    # python basic_example_chatglm3.py
    # python basic_example_llama2.py
    python basic_example_qwen_v10.py
    python basic_example_qwen_v15.py
    # python basic_example_qwen_v20.py
    cd ${REPO_ROOT}

    cd ${REPO_ROOT}/examples/python/1_performance
    # python performance_test_llama2.py --config_file config_llama2_7b.json --device_ids 0 1
    # python performance_test_qwen_v15.py --device_ids 0
    # python performance_test_qwen_v15.py --device_ids 0 1
    cd ${REPO_ROOT}

    conda deactivate
    # conda remove --name "$env_name" --all -y
}

for python_version in $TEST_VERSION; do
    run_python_example $python_version  2>&1 | tee whl_test_log_py${python_version//.}.txt
done

popd
