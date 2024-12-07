#!/bin/bash

pwd

cat > run_unit_test.sh << "EOF"
#!/bin/bash
set -x

source /root/.bashrc
export ALLSPARK_TESTCASE_PATH=/root/workspace/
git config --global --add safe.directory /root/workspace/HIE-AllSpark
cd /root/workspace/HIE-AllSpark
conda run --no-capture-output --name py38

set -e

(wget "https://hci-wlcb.oss-cn-wulanchabu.aliyuncs.com/wumi/dashinfer/replace_github_url.patch" && git apply replace_github_url.patch)

(echo "start build arm" && rm -rf build && AS_PLATFORM="armclang" ./build.sh )
(cd build && ./bin/cpp_operator_test)
(cd build && ./bin/cpp_model_test)
(cd build && ./bin/cpp_interface_test)

(cd python && rm -rf build && AS_PLATFORM="arm" ENABLE_MULTINUMA="ON" python3 setup.py bdist_wheel)

# (pip3 uninstall -y pyhie-allspark pyhie)
# (pip3 install python/dist/pyhie_allspark-1.0.0*.whl)
# (cd tests/python/arm; python3 -m unittest discover .)

(rm -rf build)
(rm -rf python/build)
EOF

chmod +x run_unit_test.sh
# testcase_path=/mnt/zhenglaiwen.zlw/tmp/as_unittest/testcase
testcase_path=/data/zhenglaiwen.zlw/test_allspark/test_unit/testcase

docker run --network=host --privileged=true                            \
       -v ~/.ssh:/root/.ssh                                            \
       -v ~/.ossutilconfig:/root/.ossutilconfig                        \
       -v `pwd`:/root/workspace/HIE-AllSpark                           \
       -v `pwd`/run_unit_test.sh:/root/run_unit_test.sh                \
       -v ${testcase_path}:/root/workspace/testcase                    \
       --rm                                                            \
       reg.docker.alibaba-inc.com/hci/dev-dashinfer-centos8-arm:v2     \
       /root/run_unit_test.sh
