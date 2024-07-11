set -x

cp ../../examples/python/4_fastchat/dashinfer_worker.py ./

dashinfer_version=v1.2.1
docker build -f fschat_ubuntu_x86.Dockerfile  . -t dashinfer/fschat_ubuntu_x86:${dashinfer_version}

rm ./dashinfer_worker.py
set +x