#!/bin/bash

cat > .build_ppu.sh << EOF
#!/bin/bash
set -x

source /root/.bashrc
git config --global --add safe.directory /root/workspace/HIE-AllSpark
cd /root/workspace/HIE-AllSpark

export AS_RPM_PACKAGE=ON
export AS_BUILD_HIEDNN=ON
export AS_SYSTEM_NV_LIB=ON
export AS_NV_STATIC_LIB=OFF
export AS_CUDA_VERSION="12.3"
export AS_SLS_LOG=OFF
export ENABLE_PPU_AS_BACKEND=ON 

pip install ninja

rm -rf build
rm -rf python/dist/*
rm -rf python/build/

set -e

(wget "https://hci-wlcb.oss-cn-wulanchabu.aliyuncs.com/wumi/dashinfer/replace_github_url.patch" && git apply replace_github_url.patch)

echo "start build ppu python package"
(cd python && python3 setup.py bdist_wheel) && \
(rm -rf python/build)

EOF

chmod +x .build_ppu.sh
rm -rf build

PPU11_DOCKER_VER=1.3

docker run --net=host --shm-size=8g --gpus all -v ~/.ssh:/root/.ssh   \
       -v ~/.ossutilconfig:/root/.ossutilconfig                       \
       -v `pwd`:/root/workspace/HIE-AllSpark                          \
       -v `pwd`/.build_ppu.sh:/root/build_ppu.sh                      \
       --rm reg.docker.alibaba-inc.com/hci/hie-allspark-ppu-dev:${PPU11_DOCKER_VER} \
       /root/build_ppu.sh 
