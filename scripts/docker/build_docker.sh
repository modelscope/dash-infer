set -x

# yum install -y yum-utils
# yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
# yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
# systemctl start docker

###################
# docker build -f dev_x86_ubuntu.Dockerfile --build-arg PY_VER=3.8  --format=docker -t dashinfer/dev-ubuntu-22.04-x86:v1
# docker build -f dev_x86_ubuntu.Dockerfile --build-arg PY_VER=3.10 --format=docker -t dashinfer/dev-ubuntu-22.04-x86:v1_py310

# docker build -f dev_x86_centos7.Dockerfile --build-arg PY_VER=3.8  --format=docker -t dashinfer/dev-centos7-x86:v2
# docker build -f dev_x86_centos7.Dockerfile --build-arg PY_VER=3.10 --format=docker -t dashinfer/dev-centos7-x86:v2_py310

# docker build -f dev_arm_alinux.Dockerfile --build-arg PY_VER=3.8  --format=docker -t dashinfer/dev-alinux-arm:v1
# docker build -f dev_arm_alinux.Dockerfile --build-arg PY_VER=3.10 --format=docker -t dashinfer/dev-alinux-arm:v1_py310

# docker build -f dev_arm_centos8.Dockerfile --build-arg PY_VER=3.8  --format=docker -t dashinfer/dev-centos8-arm:v2
# docker build -f dev_arm_centos8.Dockerfile --build-arg PY_VER=3.10 --format=docker -t dashinfer/dev-centos8-arm:v2_py310
###################

# docker build -f release_x86_manylinux2.Dockerfile --build-arg PY_VER=3.8 --format=docker -t localhost/dashinfer/dev-manylinux-x86:v2
# docker build -f test_x86_ubuntu.Dockerfile --build-arg PY_VER=3.8 --format=docker -t localhost/dashinfer/test-ubuntu-x86:v1

# docker build -f release_aarch64_manylinux2.Dockerfile --build-arg PY_VER=3.8 --format=docker -t localhost/dashinfer/dev-manylinux-arm:v2
# docker build -f test_aarch64_centos.Dockerfile --build-arg PY_VER=3.8 --format=docker -t localhost/dashinfer/test-centos-arm:v1

# docker push dashinfer/dev-ubuntu-22.04-x86:v1
# docker push dashinfer/dev-ubuntu-22.04-x86:v1_py310
# docker push dashinfer/dev-centos7-x86:v2
# docker push dashinfer/dev-centos7-x86:v2_py310
# docker push dashinfer/dev-manylinux-x86:v2
# docker push dashinfer/dev-alinux-arm:v1
# docker push dashinfer/dev-alinux-arm:v1_py310
# docker push dashinfer/dev-centos8-arm:v2
# docker push dashinfer/dev-centos8-arm:v2_py310
# docker push dashinfer/dev-manylinux-arm:v2

# docker pull registry-1.docker.io/dashinfer/dev-ubuntu-22.04-x86:v1
# docker pull registry-1.docker.io/dashinfer/dev-ubuntu-22.04-x86:v1_py310
# docker pull registry-1.docker.io/dashinfer/dev-centos7-x86:v2
# docker pull registry-1.docker.io/dashinfer/dev-centos7-x86:v2_py310
# docker pull registry-1.docker.io/dashinfer/dev-manylinux-x86:v2
# docker pull registry-1.docker.io/dashinfer/dev-alinux-arm:v1
# docker pull registry-1.docker.io/dashinfer/dev-alinux-arm:v1_py310

set +x
