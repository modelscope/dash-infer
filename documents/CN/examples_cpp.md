# 准备工作

1. 下载qwen.tiktoken文件

```shell
wget -O qwen_v1.tiktoken "https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?Revision=master&FilePath=qwen.tiktoken"
```

2. 安装DashInfer C++ package

for Ubuntu:

- install: `dpkg -i DashInfer-<dashinfer-version>-ubuntu.deb`
- uninstall: `dpkg -r dashinfer`

for CentOS:

- install: `rpm -i DashInfer-<dashinfer-version>-centos.rpm`
- uninstall (x86): `rpm -e dashinfer-<dashinfer-version>-1.x86_64`
- uninstall (arm): `rpm -e dashinfer-<dashinfer-version>-1.aarch64`

3. 模型转换

使用pyhton脚本进行模型转换，以得到DashInfer格式的模型，参考[Python示例文档](examples_python.md)。

# 从源码编译示例代码

```shell
cd <path_to_dashinfer>/examples/cpp
mkdir build
cd build
cmake .. # for gcc compiler
# cmake -DCMAKE_C_COMPILER=armclang -DCMAKE_CXX_COMPILER=armclang++ .. # for armclang compiler
make -j
```

编译过程中会从github下载第三方依赖，若因网络问题下载失败，请重试。

# 示例介绍

## 0_basic

Basic example提供了DashInfer C++接口的调用示例。其主要功能是加载DashInfer模型，并使用DashInfer进行多batch并行推理，并流式地获取推理结果。

Basic example编译可以得到两个可执行文件：example_qwen_single_numa、example_qwen_multi_numa。

### example_qwen_single_numa

功能：采用单NUMA节点推理，可以进行简单的终端交互。

运行方法：

```shell
./example_qwen_single_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

例如：

```shell
./example_qwen_single_numa -m ../../python/outputs/Qwen-7B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

### example_qwen_multi_numa

功能：采用多NUMA节点推理，可以进行简单的终端交互。

运行方法：

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./example_qwen_multi_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

例如：

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./example_qwen_multi_numa -m ../../python/outputs/Qwen-7B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

参数说明：

- AS_NUMA_NUM：用于推理的NUMA节点数量，可从Linux命令`lscpu`获取CPU的NUMA信息；
- AS_DAEMON_PATH：DashInfer binary 的安装路径。

## 1_apiserver

API server example演示了如何基于[restbed](https://github.com/Corvusoft/restbed)和DashInfer C++ API部署一个简单的流式（[generate_stream](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream)）和非流式（[generate](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream)）Text Generation Inference (TGI) server。

### 运行Server

#### Single-NUMA Server

运行方法：

```shell
./tgi_server -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

例如：

```shell
./tgi_server -m ../../python/outputs/Qwen-7B-Chat_cpu_single_float32 -t ../qwen_v1.tiktoken -c 32
```

#### Multi-NUMA Server

运行方法：

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./tgi_server_multi_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

例如：

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./tgi_server_multi_numa -m ../../python/outputs/Qwen-7B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

### 运行Client

进入到`1_apiserver`目录：

```shell
cd <path_to_dashinfer>/examples/cpp/1_apiserver
```

与流式生成Server交互：

```shell
curl -X POST -H "Content-Type: application/json" -d @example_request.json -N http://127.0.0.1:1984/generate_stream
```

与非流式生成Server交互：

```shell
curl -X POST -H "Content-Type: application/json" -d @example_request.json -N http://127.0.0.1:1984/generate
```

这条命令的作用是向`http://127.0.0.1:1984/generate_stream`（或`http://127.0.0.1:1984/generate`）发送一个POST请求。请求的内容为JSON格式。请求的主体数据来自文件example_request.json。

# Tokenizer

`<path_to_dashinfer>/examples/cpp/tokenizer`目录下为C++实现的tokenizer源码，由[斑马网络](https://www.ebanma.com/)贡献，用于encode、decode Qwen model的输入输出。
