#!/bin/bash

dashinfer_version="1.0.3"

# git clean -dxf -f

docker run --name="dashinfer-build-manylinux-wumi" \
  --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE \
  -v /home/zhenglaiwen.zlw/workspace/dashinfer_models:/root/dashinfer_models \
  -v /home/zhenglaiwen.zlw/workspace/HIE-AllSpark:/root/workspace/DashInfer \
  -v /home/zhenglaiwen.zlw/workspace/tmp/modelscope:/root/.cache/modelscope \
  -e AS_RELEASE_VERSION=${dashinfer_version} \
  -w /root/workspace/DashInfer/scripts/release \
  -it localhost/dashinfer/dev-manylinux-arm:v1 \
  /bin/bash python_manylinux_build.sh 2>&1 | tee whl_build_log.txt

docker run --name="dashinfer-test-manylinux-wumi" \
  --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE \
  -v /home/zhenglaiwen.zlw/workspace/dashinfer_models:/root/dashinfer_models \
  -v /home/zhenglaiwen.zlw/workspace/HIE-AllSpark:/root/workspace/DashInfer \
  -v /home/zhenglaiwen.zlw/workspace/tmp/modelscope:/root/.cache/modelscope \
  -e AS_RELEASE_VERSION=${dashinfer_version} \
  -w /root/workspace/DashInfer/scripts/release \
  -it localhost/dashinfer/test-centos-arm:v1 \
  /bin/bash python_manylinux_test.sh 2>&1 | tee whl_test_log.txt

# docker rm -f dashinfer-build-manylinux-wumi
# docker rm -f dashinfer-test-manylinux-wumi
