#!/bin/bash

pwd
if [ $# -eq 0 ]; then
       echo "no version number is given, eg 1.0.0"
       exit 255
fi

CONAN_SETUP="pull"
CONAN_DATA_URL="http://ait-public.oss-cn-hangzhou-zmf.aliyuncs.com/hci_team/public/HIE-Allspark/cicd/cuda/conan_data.tar.gz"

cat > pull_conan.sh << EOF
#!/bin/bash
rm -rf /root/.conan/data
wget -P /root/.conan/ $CONAN_DATA_URL
tar -xzf /root/.conan/conan_data.tar.gz -C /root/.conan
EOF


chmod +x pull_conan.sh

export CUDA1201_DOCKER_VER=2.9
export CUDA1204_DOCKER_VER=2.10

as_release_version=$1

echo ${as_release_version}
build_cuda_package(){
  local cuda_version="$1"
cat  << EOF  > .release-package_cuda.sh
#!/bin/bash
set -x
source /root/.bashrc
git config --global --add safe.directory /root/workspace/HIE-AllSpark
cd /root/workspace/HIE-AllSpark
export AS_PLATFORM="x86"
export AS_RELEASE_VERSION=${as_release_version}
export AS_RPM_PACKAGE=ON
export PLAT=manylinux2014_x86_64
if [ "$cuda_version" == "12.4" ]; then
    export AS_CUDA_SM="'80;86;89;90a'"
else
    export AS_CUDA_SM="'70;75;80;86;89;90a'"
fi

echo "conan setup option (empty if default): \$CONAN_SETUP"
if [ "\$CONAN_SETUP" == "pull" ]; then
    /root/pull_conan.sh
fi


pip install auditwheel

rm -rf build
rm -rf python/dist/*
rm -rf python/build/
rm -rf python/dist/*
rm -rf release_cuda_"$cuda_version"/
mkdir -p release_cuda_"$cuda_version"/
mkdir -p release_cuda_"$cuda_version"/python_cuda
mkdir -p release_cuda_"$cuda_version"/cpp-cuda/

(wget "https://hci-wlcb.oss-cn-wulanchabu.aliyuncs.com/wumi/dashinfer/replace_github_url.patch" && git apply replace_github_url.patch)

echo "start build cuda" && AS_RPM_PACKAGE=ON AS_PLATFORM="cuda" ./build.sh  &&    \
cp build/*.rpm  release_cuda_"$cuda_version"/cpp-cuda/ && rm -rf build  &&       \
cd python && AS_PLATFORM="cuda" python3 setup.py bdist_wheel && for whl in dist/*.whl;\
do auditwheel repair --plat $PLAT -w ../release_cuda_"$cuda_version"/python_cuda/ $whl; done  && \
rm -f dist/* &&    cd ../ &&             \
rm -rf python/build

EOF

chmod +x .release-package_cuda.sh

docker run --net=host --shm-size=8g --gpus all -v ~/.ssh:/root/.ssh   \
       -e AS_CUDA_VERSION="$cuda_version" \
       -e AS_NCCL_VERSION="$2" \
       -v `pwd`:/root/workspace/HIE-AllSpark                          \
       -v ~/.ossutilconfig:/root/.ossutilconfig                       \
       -v /tmp/ccache_dir:/root/ccache_dir                                  \
       -e CCACHE_DIR=/root/ccache_dir/                                      \
       -v `pwd`/pull_conan.sh:/root/pull_conan.sh                      \
       -v `pwd`/.release-package_cuda.sh:/root/release-package_cuda.sh \
       -e CONAN_SETUP=$CONAN_SETUP                                     \
       --rm "$3"     \
       /root/release-package_cuda.sh 
}


build_x86_package(){
cat << EOF > .release-package_x86.sh 
#!/bin/bash
set -x
source /root/.bashrc
git config --global --add safe.directory /root/workspace/HIE-AllSpark
cd /root/workspace/HIE-AllSpark
export AS_PLATFORM="x86"
export PLAT=manylinux2014_x86_64
export AS_RELEASE_VERSION=${as_release_version}
export AS_RPM_PACKAGE=ON
export PLAT=manylinux2014_x86_64

rm -rf build
rm -rf python/dist/*
rm -rf python/build/
rm -rf python/dist/*
rm -rf release_x86/
mkdir -p release_x86/
mkdir -p release_x86/python_x86
mkdir -p release_x86/cpp-x86/

echo "start build x86"


# install patchelf for audiowheel
yum install patchelf -y

conda run --no-capture-output --name py38     \
 AS_RPM_PACKAGE=ON ./build.sh \
 && cp -r build/*.rpm release_x86/cpp-x86/ && rm -rf build &&       \
 (cd python && AS_PLATFORM="x86" python3 setup.py bdist_wheel && for whl in dist/*.whl; \
do auditwheel repair --plat $PLAT -w ../release_cuda_"$cuda_version"/python_cuda/ $whl; done  && \
rm -f dist/* ) &&  cd .. &&  (rm -rf python/build)

EOF

chmod +x .release-package_x86.sh
docker run --net=host --shm-size=8g --gpus all -v ~/.ssh:/root/.ssh   \
       -v `pwd`:/root/workspace/HIE-AllSpark                          \
       -v ~/.ossutilconfig:/root/.ossutilconfig                       \
       -v `pwd`/.release-package_x86.sh:/root/release-package_x86.sh \
       --rm reg.docker.alibaba-inc.com/hci/hie-allspark-dev:${CUDA1201_DOCKER_VER}       \
       /root/release-package_x86.sh  
}

if [[ $2 == "x86" ]]; then
  build_x86_package
  echo "Executing build for x86"
elif [[ $2 == "cuda" ]]; then
  build_cuda_package "12.1" "2.11.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev-cu121:${CUDA1201_DOCKER_VER}"
  echo "Executing build for cuda-121"
  build_cuda_package "12.4" "2.23.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev-cu124:${CUDA1204_DOCKER_VER}"
  echo "Executing build for cuda-124"
else
  # defalut cuda
  build_cuda_package "12.1" "2.11.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev-cu121:${CUDA1201_DOCKER_VER}"
  # echo "Executing build for cuda-121"
  build_cuda_package "12.4" "2.23.4" "reg.docker.alibaba-inc.com/hci/hie-allspark-dev-cu124:${CUDA1204_DOCKER_VER}"
  echo "Executing build for cuda-124"
fi
