FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG PY_VER=3.10

# replace apt mirror with aliyun
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libssl-dev \
    libcurl4-openssl-dev \
    wget \
    curl \
    python3-dev \
    bzip2 \
    numactl \
    gcc \
    patchelf \
    git \
    git-lfs \
    zsh \
    build-essential \
    autoconf \
    automake \
    libtool \
    ca-certificates \
    python3 \
    python3-pip \
    unzip \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*
    
# install miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -p /miniconda -b \
    && rm -f Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --set show_channel_urls yes

RUN mkdir -p /root/.pip && printf '\
[global]\n\
index-url=http://mirrors.aliyun.com/pypi/simple/\n\
use-mirrors=true\n\
mirrors=http://mirrors.aliyun.com/pypi/simple/\n\
trusted-host=mirrors.aliyun.com' > /root/.pip/pip.conf && cat /root/.pip/pip.conf

RUN conda update --all -y && pip3 install --upgrade pip &&  pip3 install pyopenssl --upgrade
RUN conda clean -i && conda config --show channels && conda create  -y --name py310 python==3.10 && conda update -n base  conda
RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 && conda env list


SHELL ["conda", "run", "-n", "py310", "/bin/bash", "-c"]
RUN conda init
RUN pip3 install conan==1.60.0
RUN conda config --set ssl_verify false
RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
    && bash ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr
RUN curl https://gosspublic.alicdn.com/ossutil/install.sh |  bash
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia


RUN pip3 install pybind11-global
RUN pip3 install jsonlines GitPython editdistance sacrebleu nltk rouge-score
RUN echo "source activate py310" >> /root/.bashrc && source /root/.bashrc


## preinstall some conda package
COPY ./requirements_dev.txt /root/requirements_dev.txt
COPY ./conanfile.txt /root/conanfile.txt
RUN pip3 install -r /root/requirements_dev.txt
# 这里安装非常慢以至于卡住
RUN pip3 install pybind11-global jsonlines GitPython editdistance sacrebleu nltk rouge-score

RUN echo -e "if [ -f /usr/share/bash-completion/bash_completion ]; then \n    . /usr/share/bash-completion/bash_completion \nfi\n" >> /root/.bashrc && \
    source /root/.bashrc
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN pip3 install auditwheel==6.1.0


WORKDIR /root/
