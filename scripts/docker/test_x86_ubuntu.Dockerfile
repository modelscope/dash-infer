FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install numactl libopenmpi-dev curl -y

ARG PY_VER=3.8

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -p /miniconda -b \
    && rm -f Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

##########################################################################
# uncomment if want to use anaconda mirror
##########################################################################
# RUN echo -e "\
# channels:\n\
#   - defaults\n\
# show_channel_urls: true\n\
# default_channels:\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
# custom_channels:\n\
#   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
# " > /root/.condarc

RUN conda clean -i && conda config --show channels && conda create -y --name test_py python==${PY_VER} && conda update -n base conda
SHELL ["conda", "run", "-n", "test_py", "/bin/bash", "-c"]
RUN echo "source activate test_py" >> /root/.bashrc && source /root/.bashrc

##########################################################################
# uncomment if want to use pip mirror
##########################################################################
# RUN mkdir -p /root/.pip/
# RUN echo -e "[global]\ntrusted-host=mirrors.aliyun.com\nindex-url = http://mirrors.aliyun.com/pypi/simple\n\n[install]\nuse-wheel=yes" > /root/.pip/pip.conf

WORKDIR /root/
