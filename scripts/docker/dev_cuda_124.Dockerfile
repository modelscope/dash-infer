FROM nvidia/cuda:12.4.0-devel-centos7
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
RUN echo "sslverify=false" >> /etc/yum.conf

RUN yum install centos-release-scl -y --nogpgcheck
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN rpm --rebuilddb && yum install openssl-devel curl-devel wget python3-devel -y --nogpgcheck
RUN yum install -y bzip2 numactl
RUN pip3 install --upgrade pip
RUN yum install devtoolset-7 devtoolset-10 -y --nogpgcheck
RUN echo "source /opt/rh/devtoolset-10/enable" >> /root/.bashrc && source /root/.bashrc
ARG PY_VER=3.10

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -p /miniconda -b \
    && rm -f Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

##########################################################################
# uncomment if want to use anaconda mirror
##########################################################################
RUN echo -e "\
channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
" > /root/.condarc

RUN conda update --all -y && pip3 install --upgrade pip &&  pip3 install pyopenssl --upgrade
RUN conda clean -i -y && conda config --show channels && conda create -y --name ds_py python==${PY_VER} && conda update -n base conda
# RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 && conda env list
RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 -i https://mirrors.aliyun.com/pypi/simple && conda env list

SHELL ["conda", "run", "-n", "ds_py", "/bin/bash", "-c"]
RUN conda init
RUN echo "source activate ds_py" >> /root/.bashrc && source /root/.bashrc

##########################################################################
# uncomment if want to use pip mirror
##########################################################################
RUN mkdir -p /root/.pip && \
echo $'[global] \n\
index-url=http://mirrors.aliyun.com/pypi/simple/ \n\
use-mirrors=true \n\
mirrors=http://mirrors.aliyun.com/pypi/simple/ \n\
trusted-host=mirrors.aliyun.com \n' > /root/.pip/pip.conf

RUN pip3 install conan==1.60.0
RUN conda config --set ssl_verify false

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
    && bash ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr
RUN pip3 install pytest
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

RUN yum install -y epel-release && yum install -y dnf
RUN dnf makecache &&  dnf -y install ccache
RUN pip3 install jsonlines GitPython editdistance sacrebleu nltk rouge-score

## preinstall some conda package
COPY ./requirements_dev.txt /root/requirements_dev.txt
RUN pip3 install -r /root/requirements_dev.txt
RUN pip3 install pybind11-global jsonlines GitPython editdistance sacrebleu nltk rouge-score

RUN yum install -y git-lfs rpm-build patchelf
RUN yum install -y bash-completion tig

RUN yum install -y build-essential autoconf automake libtool ca-certificates

RUN yum install -y libtool flex
RUN wget "ftp://ftp.gnu.org/gnu/automake/automake-1.15.1.tar.gz" && \
    tar -xvf automake-1.15.1.tar.gz && \
    cd automake-1.15.1 && ./configure --prefix=/usr/ && make -j && make install && \
    cd .. && rm -rf automake-1.15.1.tar.gz automake-1.15.1

# git version required by github actions
RUN yum install -y gettext
RUN source /root/.bashrc && \
    wget "https://github.com/git/git/archive/refs/tags/v2.47.0.tar.gz" && \
    tar -xvf v2.47.0.tar.gz && cd git-2.47.0 && \
    make configure && ./configure --prefix=/usr && \
    make -j && make install &&\
    cd .. && rm -rf v2.47.0.tar.gz git-2.47.0

RUN curl -LO https://github.com/NixOS/patchelf/archive/refs/tags/0.14.5.tar.gz && \
    tar -xzf 0.14.5.tar.gz && \
    cd patchelf-0.14.5 && \
    ./bootstrap.sh && \
    ./configure && \
    source /opt/rh/devtoolset-10/enable && make install && \
    cd .. && rm -rf patchelf-0.14.5 0.14.5.tar.gz
RUN pip3 install auditwheel==6.1.0

RUN wget "https://xxxxxx/conan_allspark_source_cuda124_20241203_verbose.tar" && \
    tar -xvf conan_allspark_source_cuda124_20241203_verbose.tar && \
    mv conan_allspark_source_cuda124_20241203_verbose /root/.conan && \
    rm -rf conan_allspark_source_cuda124_20241203_verbose.tar

WORKDIR /root/
