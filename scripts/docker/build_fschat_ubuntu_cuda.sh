set -x

cp ../../examples/python/4_fastchat/cuda/allspark_worker.py ./

dashinfer_version=v2.0
docker build -f fschat_ubuntu_cuda.Dockerfile  . -t dashinfer/fschat_ubuntu_cuda:${dashinfer_version}

rm ./allspark_worker.py
set +x
