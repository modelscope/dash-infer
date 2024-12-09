cp  ../../python/requirements_dev.txt  .

docker_ver=v1_py310
docker build -f dev_cuda_124.Dockerfile . --build-arg PY_VER=3.10 -t dashinfer/dev-cu124:${docker_ver}
