FROM docker.io/centos:8

ARG PY_VER=3.8

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN dnf -y group install "Development Tools"
RUN yum install tar wget numactl openssl-devel curl-devel python3 -y --nogpgcheck

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash \
    && yum install git-lfs -y

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-aarch64.sh
RUN bash Miniconda3-py38_23.3.1-0-Linux-aarch64.sh -b \
    && rm -f Miniconda3-py38_23.3.1-0-Linux-aarch64.sh

ENV PATH=/root/miniconda3/bin${PATH:+:${PATH}}

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.21.6/cmake-3.21.6-linux-aarch64.sh \
    && bash ./cmake-3.21.6-linux-aarch64.sh --skip-license --prefix=/usr \
    && rm -f cmake-3.21.6-linux-aarch64.sh

RUN curl -LO https://developer.arm.com/-/media/Files/downloads/hpc/arm-compiler-for-linux/24-04/arm-compiler-for-linux_24.04_RHEL-8_aarch64.tar \
    && tar -xvf arm-compiler-for-linux_24.04_RHEL-8_aarch64.tar \
    && sh arm-compiler-for-linux_24.04_RHEL-8/arm-compiler-for-linux_24.04_RHEL-8.sh -a \
    && rm -rf arm-compiler-for-linux_24.04_RHEL-8_aarch64.tar arm-compiler-for-linux_24.04_RHEL-8

RUN echo "export ARM_COMPILER_ROOT=/opt/arm/arm-linux-compiler-24.04_RHEL-8/" >> /root/.bashrc && source /root/.bashrc
ENV PATH=/opt/arm/arm-linux-compiler-24.04_RHEL-8/bin${PATH:+:${PATH}}
ENV LIBRARY_PATH=/opt/arm/gcc-13.2.0_RHEL-8/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV LD_LIBRARY_PATH=/opt/arm/gcc-13.2.0_RHEL-8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

##########################################################################
# uncomment if want to use anaconda mirror
##########################################################################
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --remove channels defaults && \
    conda config --set show_channel_urls yes

RUN conda clean -i && conda config --show channels && conda create -y --name ds_py python==${PY_VER} && conda update -n base conda
SHELL ["conda", "run", "-n", "ds_py", "/bin/bash", "-c"]
RUN echo "source activate ds_py" >> /root/.bashrc && source /root/.bashrc

# build tools
RUN conda install -y pybind11

##########################################################################
# uncomment if want to use pip mirror
##########################################################################
RUN mkdir -p /root/.pip && \
echo $'[global] \n\
index-url = https://mirrors.aliyun.com/pypi/simple/ \n' > /root/.pip/pip.conf

RUN pip3 install --upgrade pip && pip3 install -U setuptools

# engine requirements
RUN pip3 install torch modelscope transformers==4.41.0 protobuf==3.18.3 conan==1.60.0 pytest tokenizers scons wheel pandas tabulate

RUN yum install -y libtool flex

RUN wget "ftp://ftp.gnu.org/gnu/automake/automake-1.15.1.tar.gz" && \
    tar -xvf automake-1.15.1.tar.gz && \
    cd automake-1.15.1 && ./configure --prefix=/usr/ && make -j && make install && \
    cd .. && rm -rf automake-1.15.1.tar.gz automake-1.15.1

RUN curl -LO https://github.com/NixOS/patchelf/archive/refs/tags/0.14.5.tar.gz && \
    tar -xzf 0.14.5.tar.gz && \
    cd patchelf-0.14.5 && \
    ./bootstrap.sh && \
    ./configure && \
    make install && \
    cd .. && rm -rf patchelf-0.14.5 0.14.5.tar.gz
RUN pip3 install auditwheel==6.1.0

RUN wget "https://xxxxxx/conan_allspark_source_arm_20241119.tar" && \
    tar -xvf conan_allspark_source_arm_20241119.tar && \
    mv conan_allspark_source_arm_20241119 /root/.conan && \
    rm -rf conan_allspark_source_arm_20241119.tar

WORKDIR /root/
