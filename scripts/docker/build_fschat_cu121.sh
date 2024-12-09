allspark_version=3.0.2

cp ../../examples/api_server/fschat/allspark_worker.py ./

docker build \
    -f fschat-hie-allspark-cuda.Dockerfile \
    -t fschat-hie-allspark-cuda:${allspark_version} \
    .
docker tag fschat-hie-allspark-cuda:${allspark_version} reg.docker.alibaba-inc.com/hci/fschat-hie-allspark-cuda:${allspark_version}
docker login --username= reg.docker.alibaba-inc.com
docker push reg.docker.alibaba-inc.com/hci/fschat-hie-allspark-cuda:${allspark_version}

rm ./allspark_worker.py