# Prepare

1. Download qwen.tiktoken

```shell
wget -O qwen_v1.tiktoken "https://modelscope.cn/api/v1/models/qwen/Qwen-1_8B-Chat/repo?Revision=master&FilePath=qwen.tiktoken"
```

2. Install DashInfer C++ package

for Ubuntu:

- install: `dpkg -i DashInfer-<dashinfer-version>-ubuntu.deb`
- uninstall: `dpkg -r dashinfer`

for CentOS:

- install: `rpm -i DashInfer-<dashinfer-version>-centos.rpm`
- uninstall (x86): `rpm -e dashinfer-<dashinfer-version>-1.x86_64`
- uninstall (arm): `rpm -e dashinfer-<dashinfer-version>-1.aarch64`

3. Model Conversion

Run a python script (e.g. `basic_example_qwen_v10.py`) first to get the DashInfer format model. [Documentation for Python Examples](examples_python.md)

# Build Example Code

```shell
cd <path_to_dashinfer>/examples/cpp
mkdir build
cd build
cmake .. # for gcc compiler
# cmake -DCMAKE_C_COMPILER=armclang -DCMAKE_CXX_COMPILER=armclang++ .. # for armclang compiler
make -j
```

During the compilation, third-party dependencies will be downloaded from github. If the download fails due to network problems, please retry.

# Examples

## 0_basic

Basic example provides a sample call to the DashInfer C++ interface. Its main function involves loading a DashInfer format model, employing DashInfer for parallel multi-batch inference, and streaming responses from the LLM.

Basic example can be compiled to get two executable files: example_qwen_single_numa, example_qwen_multi_numa.

### example_qwen_single_numa

Function: Use a single NUMA node for inferece, allowing simple terminal interaction.

Run:

```shell
./example_qwen_single_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

To be more specific:

```shell
./example_qwen_single_numa -m ../../python/outputs/Qwen-1_8B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

### example_qwen_multi_numa

Function: Use multiple NUMA node for inference, allowing simple terminal interaction.

Run:

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./example_qwen_multi_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

To be more specific:

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./example_qwen_multi_numa -m ../../python/outputs/Qwen-1_8B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

Parameter description:

- AS_NUMA_NUM: the number of NUMA nodes used for inference, you can get the NUMA information of CPU from Linux command `lscpu`;
- AS_DAEMON_PATH: installation path of DashInfer binary.

## 1_apiserver

The API server example demonstrates how to deploy a simple streaming ([generate_stream](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream)) and non-streaming ([generate](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream)) Text Generation Inference (TGI) server based on [restbed](https://github.com/Corvusoft/restbed) and the DashInfer C++ API.

### Run the Server

#### Single-NUMA Server

Run:

```shell
./tgi_server -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

To be more specific:

```shell
./tgi_server -m ../../python/outputs/Qwen-1_8B-Chat_cpu_single_float32 -t ../qwen_v1.tiktoken -c 32
```

#### Single-NUMA Server

Run:

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./tgi_server_multi_numa -m <path_to_dashinfer_model> -t <path_to_qwen_v1.tiktoken> -c 32
```

To be more specific:

```shell
AS_NUMA_NUM=2 AS_DAEMON_PATH=/usr/bin ./tgi_server_multi_numa -m ../../python/outputs/Qwen-1_8B-Chat_cpu_multi_float32 -t ../qwen_v1.tiktoken -c 32
```

### Run the Client

Go to the `1_apiserver` directory:

```shell
cd <path_to_dashinfer>/examples/cpp/1_apiserver
```

Interacts with the streaming server:

```shell
curl -X POST -H "Content-Type: application/json" -d @example_request.json -N http://127.0.0.1:1984/generate_stream
```

Interacts with the non-streaming server:

```shell
curl -X POST -H "Content-Type: application/json" -d @example_request.json -N http://127.0.0.1:1984/generate
```

The effect of this command is to send a POST request to `http://127.0.0.1:1984/generate_stream` (or `http://127.0.0.1:1984/generate`). The content of the request is in JSON format. The body data of the request comes from the file example_request.json.

# Tokenizer

The source code in the `<path_to_dashinfer>/examples/cpp/tokenizer` folder is the tokenizer code implemented in C++, contributed by [banma network](https://www.ebanma.com/), for encode, decode Qwen model's input and output tokens.
