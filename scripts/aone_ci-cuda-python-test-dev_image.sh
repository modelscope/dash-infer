#!/bin/bash

# pull conan data from OSS if conan setup fails in CI
CONAN_SETUP="pull"

## How to build this cache: (cd  ~/.conan/; tar zcf data.tgz data; mv data.tgz $OLDPWD), then use oss utils push to oss dir.
CONAN_DATA_URL="http://ait-public.oss-cn-hangzhou-zmf.aliyuncs.com/hci_team/public/HIE-Allspark/cicd/cuda/conan_data.tar.gz"

export CUDA1104_DOCKER_VER=2.9
export CUDA1204_DOCKER_VER=2.10

pwd

cat > pull_conan.sh << EOF #!/bin/bash
rm -rf /root/.conan/data
wget -P /root/.conan/ $CONAN_DATA_URL
tar -xzf /root/.conan/conan_data.tar.gz -C /root/.conan
EOF

chmod +x pull_conan.sh

mkdir -p $NAS_PATH/modelscope_cache
chmod 0777 $NAS_PATH/modelscope_cache
mkdir -p ~/ci_ccache_dir/

build_cuda_package() {
  local cuda_version="$1"
cat > run_unit_test.sh << EOF
#!/bin/bash
set -x

if ! command -v numactl &> /dev/null; then
    echo "numactl is not installed"
    echo "Installing numactl..."
    # 使用yum安装numactl
    yum install -y numactl
else
    echo "numactl is already installed"
fi
output=\$(numactl -H)
first_line=\$(echo "\$output" | head -n 1)
nodes=\$(echo "\$first_line" | awk '{print \$2}')
export AS_NUMA_NUM="\$nodes"
export CUDA_VISIBLE_DEVICES=6,7
export BFC_ALLOCATOR=ON
export BFC_ALLOW_GROWTH=OFF
export ALLSPARK_USE_TORCH_SAMPLE=1
export ALLSPARK_TESTCASE_PATH=/root/mnt/nas/HIE-Allspark-Testcase/
if [ "$cuda_version" == "12.4" ]; then
    export AS_CUDA_SM="'80;90a'"
else
    export AS_CUDA_SM="'80'"
fi
source /root/.bashrc
git config --global --add safe.directory /root/workspace/HIE-AllSpark
cd /root/workspace/HIE-AllSpark
echo "conan setup option (empty if default): \$CONAN_SETUP"
if [ "\$CONAN_SETUP" == "pull" ]; then
    /root/pull_conan.sh
fi
##########################################################################
# uncomment if want to use anaconda mirror
##########################################################################
mkdir /root/.pip

pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip3 config set install.trusted-host mirrors.aliyun.com

set -e

conda activate py310

(rm -rf python/dist/* python/build build)

(wget "https://hci-wlcb.oss-cn-wulanchabu.aliyuncs.com/wumi/dashinfer/replace_github_url.patch" && git apply replace_github_url.patch)

(pip3 uninstall -y dashinfer)
(echo "start build cpp package" && AS_PLATFORM="cuda" ./build.sh)

# run cpp test first.
if [ "$cuda_version" == "12.4" ]; then
    export BFC_MEM_RATIO=0.8
    # filter out some test cannot pass.
    (cd build && ./bin/cpp_kernel_test --gtest_filter=-MHAPrefillTest.flashv2_half_*:MHAPrefillTest.flashv2_bf16_*:MHAPrefillTest.xformer_bf16_*)
    (cd build && ./bin/cpp_operator_test)
    (cd build && ./bin/cpp_model_test --gtest_filter=-AsModelCUDA.M6_7B*)
    # (cd build && ./bin/cpp_model_test --gtest_filter=-AsModelCUDA.M6_7B_CacheDefault_Interrupted)
    (cd build && ./bin/cpp_interface_test)

    # export BFC_MEM_RATIO=0.2 # need little memory to trigger interruptions
    # (cd build && ./bin/cpp_model_test --gtest_filter=AsModelCUDA.M6_7B_CacheDefault_Interrupted)
fi

mkdir -p python/build
# python test and cpp test
ln -s /root/workspace/HIE-AllSpark/build /root/workspace/HIE-AllSpark/python/build/temp.linux-x86_64-cpython-310

(pip3 install -r python/requirements_dev.txt)
(echo "start build python package" && cd python && AS_PLATFORM="cuda" python3 setup.py bdist_wheel)

(pip3 install python/dist/*.whl)

# run python test after python build.
if [ "$cuda_version" == "12.4" ]; then
    export BFC_MEM_RATIO=0.8
    (cd tests; pytest -s python/x86)
    (cd tests; pytest -s python/gpu --ignore=python/gpu/test_02_m6_7b_rich.py)
fi

(rm -rf python/dist/* python/build build)
EOF

chmod +x run_unit_test.sh

docker run --privileged --net=host --shm-size=8g --gpus all       \
       -v ~/.ssh:/root/.ssh -v `pwd`:/root/workspace/HIE-AllSpark \
       -v ~/.ossutilconfig:/root/.ossutilconfig                   \
       -v `pwd`/run_unit_test.sh:/root/run_unit_test.sh           \
       -v $NAS_PATH/modelscope_cache:/root/.cache/                \
       -v `pwd`/pull_conan.sh:/root/pull_conan.sh                 \
       -v ~/ci_ccache_dir:/root/ccache_dir                        \
       -e CCACHE_DIR=/root/ccache_dir/                            \
       -v $NAS_PATH:/root/mnt/nas:ro                              \
       -e CONAN_SETUP=$CONAN_SETUP                                \
       -e AS_CUDA_VERSION="$cuda_version"                         \
       -e AS_NCCL_VERSION="$2"                                    \
       --rm "$3"                                                  \
       /root/run_unit_test.sh
}

echo "########################################"
echo "# Executing build for cuda-114"
echo "########################################"
build_cuda_package "11.4" "2.11.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev:${CUDA1104_DOCKER_VER}"

echo "########################################"
echo "# Executing build for cuda-124"
echo "########################################"
build_cuda_package "12.4" "2.11.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev-cu124:${CUDA1204_DOCKER_VER}"
