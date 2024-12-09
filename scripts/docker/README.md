# Build a Developer docker

notes: it will run a conan install to make most of conan depends cached in docker image.

``` bash
cp ../../conan/conanfile.txt  .
cp  ../../python/requirements_dev.txt  .
docker_ver=2.9
docker build  -f dev_cuda_114.Dockerfile  . -t hie-allspark-dev:${docker_ver}

# build with proxy.
docker build  \ 
   --build-arg http_proxy=http://11.169.82.19:80 \ 
   --build-arg https_proxy=http://11.169.82.19:80 \ 
   --build-arg no_proxy="localhost,127.0.0.1,.aliyun.com,.alibaba-inc.com" \ 
  -f dev_cuda_114.Dockerfile  . -t hie-allspark-dev:${docker_ver}
```

# Updaete 

``` bash
docker login --username= reg.docker.alibaba-inc.com

docker tag hie-allspark-dev:${docker_ver} reg.docker.alibaba-inc.com/hci/hie-allspark-dev:${docker_ver}
docker push reg.docker.alibaba-inc.com/hci/hie-allspark-dev:${docker_ver}
```

