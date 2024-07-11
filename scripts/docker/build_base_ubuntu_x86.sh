set -x

dashinfer_version=v1.2.1
docker build -f base_ubuntu_x86.Dockerfile  . -t dashinfer/base_ubuntu_x86:${dashinfer_version}

set +x