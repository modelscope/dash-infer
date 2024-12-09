FROM centos:7

RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
RUN echo "sslverify=false" >> /etc/yum.conf

RUN rpm --rebuilddb && yum install openssl-devel curl-devel wget python3-devel -y --nogpgcheck
RUN yum install -y bzip2 numactl rpm-build epel-release

RUN yum install centos-release-scl -y --nogpgcheck
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN yum install devtoolset-7 -y --nogpgcheck
RUN echo "source /opt/rh/devtoolset-7/enable" >> /root/.bashrc && source /root/.bashrc
ARG PY_VER=3.8

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash \
    && yum install git-lfs -y

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
    && bash ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr

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

RUN conda clean -i -y && conda config --show channels && conda create -y --name ds_py python==${PY_VER} && conda update -n base conda
# RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 && conda env list
RUN conda run python --version && pip3 install --upgrade pip pyOpenSSL==22.0.0 -i https://mirrors.aliyun.com/pypi/simple && conda env list
SHELL ["conda", "run", "-n", "ds_py", "/bin/bash", "-c"]
RUN echo "source activate ds_py" >> /root/.bashrc && source /root/.bashrc

# build tools
RUN conda install -y pybind11

##########################################################################
# uncomment if want to use pip mirror
##########################################################################
RUN mkdir -p /root/.pip/
RUN echo -e "[global]\ntrusted-host=mirrors.aliyun.com\nindex-url = http://mirrors.aliyun.com/pypi/simple\n\n[install]\nuse-wheel=yes" > /root/.pip/pip.conf

# engine requirements
RUN conda install -y pytorch-cpu -c pytorch
RUN pip3 install modelscope transformers==4.41.0 protobuf==3.18.3 conan==1.60.0 pytest tokenizers scons wheel pandas tabulate

RUN yum install -y libtool flex

# RUN wget "ftp://ftp.gnu.org/gnu/automake/automake-1.15.1.tar.gz" && \
RUN wget "https://test-bucket-duplicate.oss-cn-hangzhou.aliyuncs.com/wumi/downloads/automake-1.15.1.tar.gz" && \
    tar -xvf automake-1.15.1.tar.gz && \
    cd automake-1.15.1 && ./configure --prefix=/usr/ && make -j && make install && \
    cd .. && rm -rf automake-1.15.1.tar.gz automake-1.15.1

RUN wget "https://test-bucket-duplicate.oss-cn-hangzhou.aliyuncs.com/wumi/docker/conan_allspark_source_x86_20241119.tar" && \
    tar -xvf conan_allspark_source_x86_20241119.tar && \
    mv conan_allspark_source_x86_20241119 /root/.conan && \
    rm -rf conan_allspark_source_x86_20241119.tar

WORKDIR /root/
