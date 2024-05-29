#!/bin/bash
set -e -x

source activate test_py

root_path=/root/workspace/DashInfer

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

    pip install -r ${root_path}/examples/python/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    pip install ${root_path}/python/wheelhouse/dashinfer-${AS_RELEASE_VERSION}-cp${python_version//.}-cp${python_version//.}-manylinux*.whl

    cd ${root_path}/examples/python/0_basic
    # python basic_example_chatglm2.py
    # python basic_example_chatglm3.py
    # python basic_example_llama2.py
    python basic_example_qwen_v10.py
    python basic_example_qwen_v15.py
    # python basic_example_qwen_v20.py
    cd ${root_path}

    cd ${root_path}/examples/python/1_performance
    # python performance_test_llama2.py --config_file config_llama2_7b.json --device_ids 0 1
    # python performance_test_qwen_v15.py --device_ids 0
    python performance_test_qwen_v15.py --device_ids 0 1
    cd ${root_path}

    conda deactivate
    # conda remove --name "$env_name" --all -y
}

run_python_example 3.8  2>&1 | tee whl_test_log_py38.txt
run_python_example 3.9  2>&1 | tee whl_test_log_py39.txt
run_python_example 3.10 2>&1 | tee whl_test_log_py310.txt
run_python_example 3.11 2>&1 | tee whl_test_log_py311.txt
