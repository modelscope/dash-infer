FROM ubuntu:22.04

RUN apt-get update \
  && apt-get install -y curl \
  && rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && chmod +x Miniconda3-latest-Linux-x86_64.sh \
  && ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
  && rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/miniconda/bin:${PATH}

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

RUN conda create -n py38env python==3.8 \
    && conda init bash \
    && echo "source activate py38env" >> ~/.bashrc

SHELL ["conda", "run", "-n", "py38env", "/bin/bash", "-c"]

RUN conda env list

RUN conda install pytorch cpuonly -c pytorch

RUN pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  dashinfer \
  pandas \
  tabulate \
  transformers \
  py-cpuinfo \
  einops \
  transformers_stream_generator
