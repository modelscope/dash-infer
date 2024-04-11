# Installation

## Requirements

OS: Linux

Python: 3.8, 3.10

Tested compiler version:

- gcc: 7.3.1, 11.4.0
- armclang: 22.1

For multi-NUMA inference, `numactl`, `openmpi` are required:

- for Ubuntu: `apt-get install numactl libopenmpi-dev`
- for CentOS: `yum install numactl openmpi-devel openssh-clients -y`

> For CPUs with multiple NUMA nodes, it is recommended to install the above dependencies even if you only want to run the program on one NUMA node, otherwise remote memory accesses may occur and performance is not guaranteed to be optimal.

> If there are unmentioned dependency issues during the use of the package, it is recommended to use a pre-built docker image or refer to the dockerfile under `<path_to_dashinfer>/scripts/docker` to build the development environment.

## Install Python Package

install requirements:

```shell
conda install -y pytorch-cpu -c pytorch # install cpu-only pytorch

pip install -r examples/python/requirements.txt
```

install DashInfer python package:

- install from pip: `pip install dashinfer`
- install local package: `pip install dashinfer-<dashinfer-version>-xxx.whl`
- uninstall: `pip uninstall dashinfer -y`

## Install C++ Package

Download corresponding C++ package, and execute following command to install:

Pre-built C++ Packages:

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

# Build from Source

## Docker Environment

It is recommended to use a pre-built docker image or refer to the dockerfile under `<path_to_dashinfer>/scripts/docker` to build your own docker environment.

Pull official docker image:

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

Or build docker image from dockerfile:

```shell
# if use podman
docker build -f <path_to_dockerfile> --format=docker 

# if use docker
docker build -f <path_to_dockerfile> .
```

The default python version is 3.8. If you want to build docker images with different python version, you can specify python version by `--build-arg PY_VER=3.10`. For example, python 3.10:

```shell
# if use podman
docker build -f <path_to_dockerfile> --build-arg PY_VER=3.10 --format=docker 

# if use docker
docker build -f <path_to_dockerfile> --build-arg PY_VER=3.10 .
```

Create a container:

```shell
docker run -d --name="dashinfer-dev-${USER}" \
  --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE \
  -v $(pwd):/root/workspace/DashInfer \
  -w /root/workspace \
  -it <docker_image_tag>
```

> When creating a container, `--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host` arguments are required, because components such as numactl and openmpi need the appropriate permissions to run. If you only need to use the single NUMA API, you may not grant this permission.

Run the container:

```shell
docker exec -it "dashinfer-dev-${USER}" /bin/bash
```

## Clone the Repository

```shell
git clone git@github.com:modelscope/dash-infer.git
git lfs pull
```

## Third-party Dependencies

DashInfer uses conan to manage third-party dependencies.

During the initial compilation, downloading third-party dependency packages may take a considerable amount of time.

For the official docker image, we provide a conan package archive. If some of the packages are not accessible on your device, please download the corresponding zip archive, unzip it and put it in the `~/.conan` directory.

Conan package archive:

- x86, ubuntu: [link](TODO)
- x86, centos: [link](TODO)
- arm, alinux: [link](TODO)

## Build C++ Package

Execute the following command under DashInfer root path:

- x86 CPU

```shell
AS_PLATFORM="x86" AS_RELEASE_VERSION="1.0.0" AS_BUILD_PACKAGE=ON AS_CXX11_ABI=ON ./build.sh
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_RELEASE_VERSION="1.0.0" AS_BUILD_PACKAGE=ON AS_CXX11_ABI=ON ./build.sh
```

> Note:
> - AS_RELEASE_VERSION: Specifies the version number of the installation package.
> - AS_BUILD_PACKAGE option: Compile Linux software installation packages. For Ubuntu, it compiles .deb packages; for CentOS, it compiles .rpm packages. The compiled .deb/.rpm packages is located in the `<path_to_dashinfer>/build`.
> - AS_CXX11_ABI: Enable or disable CXX11 ABI.

## Build Python Package

Execute the following command under `<path_to_dashinfer>/python`:

- x86 CPU

```shell
AS_PLATFORM="x86" AS_RELEASE_VERSION="1.0.0" AS_CXX11_ABI="ON" python3 setup.py bdist_wheel
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_RELEASE_VERSION="1.0.0" AS_CXX11_ABI="ON" python3 setup.py bdist_wheel
```

The compiled .whl installer is located in the `<path_to_dashinfer>/python/dist`.
