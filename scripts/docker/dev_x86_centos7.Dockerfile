FROM centos:7

RUN rpm --rebuilddb && yum install openssl-devel curl-devel wget python3-devel -y --nogpgcheck
RUN yum install -y bzip2 numactl rpm-build epel-release

RUN yum install centos-release-scl -y --nogpgcheck
RUN yum install devtoolset-7 -y --nogpgcheck
RUN echo "source /opt/rh/devtoolset-7/enable" >> /root/.bashrc && source /root/.bashrc
ARG PY_VER=3.8

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash \
    && yum install git-lfs -y 

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -p /miniconda -b \
    && rm -f Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

# uncomment if want to use anaconda mirror
##########################################################################
# RUN echo -e "\                                                         #
# channels:\n\                                                           #
#   - defaults\n\                                                        #
# show_channel_urls: true\n\                                             #
# default_channels:\n\                                                   #
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\         #
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\            #
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\        #
# custom_channels:\n\                                                    #
#   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\  #
#   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\        #
#   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\     #
#   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\        #
#   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\      #
#   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\  #
#   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\    #
#   deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\ #
# " > /root/.condarc                                                     #
##########################################################################

RUN conda clean -i -y && conda config --show channels && conda create -y --name ds_py python==${PY_VER} && conda update -n base conda
RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 && conda env list
SHELL ["conda", "run", "-n", "ds_py", "/bin/bash", "-c"]
RUN echo "source activate ds_py" >> /root/.bashrc && source /root/.bashrc

RUN mkdir -p /root/.pip/

# uncomment if want to use pip mirror
##########################################################################
# RUN echo -e "[global]\ntrusted-host=mirrors.aliyun.com\nindex-url = http://mirrors.aliyun.com/pypi/simple\n\n[install]\nuse-wheel=yes" > /root/.pip/pip.conf
##########################################################################

# engine requirements
RUN conda install -y pytorch-cpu -c pytorch
RUN pip3 install transformers==4.38.0 protobuf==3.18.0 conan==1.60.0 pytest tokenizers scons wheel

# demo requirements
RUN pip3 install modelscope tabulate gradio

# model requirements
RUN pip3 install sentencepiece accelerate transformers_stream_generator tiktoken

# build tools
RUN conda install -y pybind11

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh  && bash ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr

WORKDIR /root/
