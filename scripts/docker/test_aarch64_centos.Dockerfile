FROM docker.io/centos:8

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN yum install numactl curl-devel wget python3-devel gcc epel-release -y --nogpgcheck

ARG PY_VER=3.8

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-aarch64.sh \
    && bash Miniconda3-py38_23.3.1-0-Linux-aarch64.sh -b \
    && rm -f Miniconda3-py38_23.3.1-0-Linux-aarch64.sh

ENV PATH=/root/miniconda3/bin${PATH:+:${PATH}}

##########################################################################
# uncomment if want to use anaconda mirror
##########################################################################
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --remove channels defaults && \
    conda config --set show_channel_urls yes

RUN conda clean -i && conda config --show channels && conda create -y --name test_py python==${PY_VER} && conda update -n base conda
SHELL ["conda", "run", "-n", "test_py", "/bin/bash", "-c"]
RUN echo "source activate test_py" >> /root/.bashrc && source /root/.bashrc

##########################################################################
# uncomment if want to use pip mirror
##########################################################################
# RUN mkdir -p /root/.pip && \
# echo $'[global] \n\
# index-url = https://mirrors.aliyun.com/pypi/simple/ \n' > /root/.pip/pip.conf

WORKDIR /root/
