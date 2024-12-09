# Installation Guide

## System Requirements

- OS: Linux

- Python:
  - Python 3.8, 3.9, 3.10, 3.11
  - PyTorch: any PyTorch version, CPU or GPU.

- Tested compiler version:
  - gcc: 7.3.1, 11.4.0
  - arm compiler: 22.1, 24.04

- CUDA
  - CUDA sdk version >= 11.4
  - cuBLAS: CUDA sdk provided

- CPU
For multi-NUMA inference, `numactl`, `openmpi` are required:
  - For Ubuntu: `apt-get install numactl libopenmpi-dev`
  - For CentOS: `yum install numactl openmpi-devel openssh-clients -y`



## Install Python Package

Install python package by following command:

<!---
- install from pip: `pip install dashinfer`
-->
- Install local package: `pip install dashinfer-allspark-<version>-xxx.whl`
- Uninstall: `pip uninstall dashinfer-allspark -y`

## Install C++ Pacakge

for Ubuntu:

- Install: `dpkg -i DashInfer-<version>-ubuntu.deb`
- Uninstall: `dpkg -r DashInfer`

for CentOS:

- Install: `rpm -i DashInfer-<version>-centos.rpm`

