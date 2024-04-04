# 安装

## Requirements

OS: Linux

Python: 3.8, 3.10

测试过的编译器版本:

- gcc: 7.3.1, 11.4.0
- armclang: 22.1

多NUMA推理，需要安装依赖`numactl`、`openmpi`，例如：

- for Ubuntu: `apt-get install numactl libopenmpi-dev`
- for CentOS: `yum install numactl openmpi-devel openssh-clients -y`

> 对于多NUMA节点的CPU，即使只跑单NUMA，也建议安装以上依赖，否则可能会出现跨节点内存访问，无法保证性能达到最佳。

> 若package的使用过程中出现未提及的依赖问题，推荐使用预构建的docker镜像或参考`<path_to_dashinfer>/scripts/docker`下的dockerfile构建docker开发环境。

## 安装python包

install requirements:

```shell
conda install -y pytorch-cpu -c pytorch # install cpu-only pytorch
pip3 install huggingface_hub # for huggingface user
# pip3 install modelscope # for modelscope user
pip3 install tabulate gradio # requirements for running examples
pip3 install sentencepiece accelerate transformers_stream_generator tiktoken # model requirements
```

install DashInfer python package:

- install from pip: `pip install dashinfer`
- install local package: `pip install dashinfer-<dashinfer-version>-xxx.whl`
- uninstall: `pip uninstall dashinfer -y`

## 安装C++包

下载对应版本的C++ package到本地后，执行以下命令进行安装。

预编译的C++包:

- x86, ubuntu: [link](TODO)
- x86, centos: [link](TODO)
- arm, alinux: [link](TODO)

for Ubuntu:

- install: `dpkg -i DashInfer-<dashinfer-version>-ubuntu.deb`
- uninstall: `dpkg -r dashinfer`

for CentOS:

- install: `rpm -i DashInfer-<dashinfer-version>-centos.rpm`
- uninstall (x86): `rpm -e dashinfer-<dashinfer-version>-1.x86_64`
- uninstall (arm): `rpm -e dashinfer-<dashinfer-version>-1.aarch64`

# 从源码安装

## 用Docker配置开发环境

推荐使用预构建的docker image或参考`<path_to_dashinfer>/scripts/docker`下的dockerfile构建docker开发环境。

拉取预构建的docker image:

- x86, ubuntu:

```shell
# python 3.8
docker pull registry-1.docker.io/dashinfer/dev-ubuntu-22.04-x86:v1

# python 3.10
docker pull registry-1.docker.io/dashinfer/dev-ubuntu-22.04-x86:v1_py310
```

- x86, centos:

```shell
# python 3.8
docker pull registry-1.docker.io/dashinfer/dev-centos7-x86:v1

# python 3.10
docker pull registry-1.docker.io/dashinfer/dev-centos7-x86:v1_py310
```

- arm, alinux:

```shell
# python 3.8
docker pull registry-1.docker.io/dashinfer/dev-alinux-arm:v1

# python 3.10
docker pull registry-1.docker.io/dashinfer/dev-alinux-arm:v1_py310
```

或者从dockerfile构建docker image：

```shell
# if use podman
docker build -f <path_to_dockerfile> --format=docker 

# if use docker
docker build -f <path_to_dockerfile> .
```

开发docker默认使用Python 3.8，如果需要构建不同python版本的docker image，可以增加此参数`--build-arg PY_VER=3.10`，例如python 3.10：

```shell
# if use podman
docker build -f <path_to_dockerfile> --build-arg PY_VER=3.10 --format=docker 

# if use docker
docker build -f <path_to_dockerfile> --build-arg PY_VER=3.10 .
```

从docker image创建container：

```shell
docker run -d --name="dashinfer-dev-${USER}" \
  --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE \
  -v $(pwd):/root/workspace/DashInfer \
  -w /root/workspace \
  -it <docker_image_tag>
```

> 注：在创建container时，添加`--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host`参数是因为numactl、openmpi等组件需要相应的权限来运行。如仅需要使用单NUMA API，可以不赋予该权限。

运行container：

```shell
docker exec -it "dashinfer-dev-${USER}" /bin/bash
```

## 第三方依赖

DashInfer使用conan管理第三方依赖。

在首次进行编译时，可能需要较长时间下载第三方依赖包。

对于官方提供的docker image，我们提供编译好的第三方依赖。下载对应的压缩包，解压缩后放到`~/.conan`目录下，可以缩短第三方依赖的编译时间。

预编译conan package:

- x86, ubuntu: [link](TODO)
- x86, centos: [link](TODO)
- arm, alinux: [link](TODO)

## 编译C++包

在DashInfer仓库根目录执行以下命令：

- x86 CPU

```shell
AS_PLATFORM="x86" AS_BUILD_PACKAGE=ON AS_RELEASE_VERSION="1.0.0" ./build.sh
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_BUILD_PACKAGE=ON AS_RELEASE_VERSION="1.0.0" ./build.sh
```

> 说明：
> - AS_BUILD_PACKAGE选项：编译Linux软件安装包，对于Ubuntu操作系统，编译.deb安装包，对于CentOS，编译.rpm安装包，编译得到的.deb/.rpm安装包，位于`<path_to_dashinfer>/build`目录下；
> - AS_RELEASE_VERSION：指定安装包的版本号。

## 编译python包

在`<path_to_dashinfer>/python`下执行以下命令：

- x86 CPU

```shell
AS_PLATFORM="x86" AS_RELEASE_VERSION="1.0.0" python3 setup.py bdist_wheel
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_RELEASE_VERSION="1.0.0" python3 setup.py bdist_wheel
```

编译得到的.whl安装包位于`<path_to_dashinfer>/python/dist`目录。
