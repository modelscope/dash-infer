<div align="center">

[![PyPI](https://img.shields.io/pypi/v/dashinfer)](https://pypi.org/project/dashinfer/)
<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/modelscope/dash-infer/blob/main/README_CN.md">中文</a>
    </p>
</h4>


</div>

# Introduction

Written in C++ runtime, DashInfer aims to deliver production-level implementations highly optimized for various hardware architectures, including x86 and ARMv9. It supports both Continuous Batching and NUMA-Aware capabilities for CPU, and can fully utilize the capabilities of modern server-grade CPUs to host LLMs up to 14B in size.

## Main Features

- **Lightweight Architecture**: DashInfer requires minimal third-party dependencies and uses static linking for dependency libraries. By providing C++ and Python interfaces, DashInfer can be easily integrated into your existing system.
- **High-Precision**: DashInfer has been rigorously tested for accuracy and is able to provide inference accuracy consistent with PyTorch and other GPU engines (e.g., vLLM).
- **Standard LLM Inference Techniques**: DashInfer employs standard LLM inference techniques, such as:
  - Continuous Batching allows for the immediate insertion of new requests and supports streaming outputs.
  - Request-based asynchronous interfaces offer individual control over generation parameters and request statuses for each request.
- **Support for Mainstream Open-Source LLMs**: DashInfer supports mainstream open-source LLMs, including Qwen, LLaMA, ChatGLM, etc., and supports loading models in the Huggingface format.
- **Post Training Quantization (PTQ)**: Using DashInfer's InstantQuant (IQ), weight-only quantization acceleration can be achieved without fine-tuning, improving deployment efficiency. Accuracy evaluation shows that IQ has no impact on model accuracy. The current version supports weight-only 8-bit quantization on ARM CPUs.
- **Optimized Computation Kernels**: With OneDNN and self-developed assembly kernels, DashInfer is able to maximize the performance of the hardware on both ARM and x86.
- **NUMA-Aware Design**: DashInfer supports tensor parallel inference across multiple NUMA nodes, fully leveraging the computing power of server CPUs. With numactl and a multi-process architecture, the NUMA affinity of threads is accurately controlled to maximize the performance of multi-node CPUs and avoid the performance degradation caused by cross-NUMA access. For more information on NUMA, see: [Optimizing Applications for NUMA - Intel](https://www.intel.com/content/dam/develop/external/us/en/documents/3-5-memmgt-optimizing-applications-for-numa-184398.pdf), [What is NUMA?](https://www.kernel.org/doc/html/v5.0/vm/numa.html).
- **Context Length**: The current version supports up to 32k context length, with plans to extend to longer context lengths in the future.
- **Multi-Language API Interfaces**: Both C++ and Python interfaces are supported. It is possible to extend C++ interface to Java, Rust and other programming languages, via standard cross-language interfaces.
- **Operating System Support**: DashInfer supports mainstream Linux server operating systems like Centos7 and Ubuntu22.04, and provides corresponding Docker images.

## DashInfer Demo

[DashInfer Demo on ModelScope](https://modelscope.cn/studios/modelscope/DashInfer-Demo/summary)

Demo information:

- Model: Qwen1.5-7B-Chat
- Engine: DashInfer
- Hardware: x86, Emerald Rapids, 96 vCPU @ 3.2GHz, 16GBx24 DDR
- Aliyun instance: [ecs.g8i.24xlarge](https://www.alibabacloud.com/help/en/ecs/user-guide/overview-of-instance-families#g8i)

## Documentation

- [Installation](documents/EN/installation.md)
- [C++ Examples](documents/EN/examples_cpp.md)
- [Python Examples](documents/EN/examples_python.md)
- [Performance](documents/EN/performance.md)

# Supported Hardware and Data Types

## Hardware

- **x86 CPUs**: Hardware support for AVX2 instruction set is required. For Intel's 5th generation Xeon processors (Emerald Rapids), 4th generation Xeon processors (Sapphire Rapids), corresponding to Aliyun's 8th generation ECS instances (e.g., g8i), AMX instructions are used to accelerate caculation.
- **ARMv9 CPU**: Hardware support for SVE instruction set is required. DashInfer supports ARMv9 architecture processors such as Yitian710, corresponding to Aliyun's 8th generation ECS instances (e.g. g8y), and adopts SVE instruction to accelerate caculation.

## Data Types

- **x86 CPU**: FP32, BF16
- **ARM Yitian710 CPU**: FP32, BF16, InstantQuant

### InstantQuant

InstantQuant is a weight-only quantization technique.

On the Yitian710 CPU (ARMv9), DashInfer supports weight-only quantization.

To perform weight-only quantization, the `do_dynamic_quantize_convert` and `quantization_config` fields of the model configuration file need to be modified. Refer to the [documentation](documents/EN/examples_python.md) for a detailed description of the parameters.

The weight-only quantization, which will find the maximum and minimum values of weight within the range of GroupSize and map the weight values to the uint8 range, is calculated as follows:

$$ scale = \frac {x_{fp32_{max}} - x_{fp32_{min}}} {255 - 0} $$
$$ zeropoint = 0 - \frac {x_{fp32_{min}}} {scale} $$
$$ x_{u8} = x_{fp32} / scale + zeropoint $$

During inference, the quantized weight is recovered as bfloat16 for matrix multiplication.

# Supported Models

| Architecture | Models | DashInfer model_type | HuggingFace Models | ModelScope Models | DashInfer Models |
|:------------:|:------:|:--------------------:|:------------------:|:-----------------:|:----------------:|
| QWenLMHeadModel | Qwen | Qwen_v10 | [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc. | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. | / |
| Qwen2ForCausalLM | Qwen1.5 | Qwen_v15 | [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),<br>[Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),<br>[Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat),<br>[Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat),<br>[Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), etc. | [qwen/Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary),<br>[qwen/Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary),<br>[qwen/Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary),<br>[qwen/Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary),<br>[qwen/Qwen1.5-14B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary), etc. | / |
| ChatGLMModel | ChatGLM | ChatGLM_v2 | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b),<br>[THUDM/chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k) | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary),<br>[ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary) | / |
| ChatGLMModel | ChatGLM | ChatGLM_v3 | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b),<br>[THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary),<br>[ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary) | / |
| ChatGLMModel | ChatGLM | ChatGLM_v4 | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/summary) | [dash-infer/glm-4-9b-chat-DI](https://modelscope.cn/models/dash-infer/glm-4-9b-chat-DI/summary) |
| LlamaForCausalLM | LLaMA-2 | LLaMA_v2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [modelscope/Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary),<br>[modelscope/Llama-2-13b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary) | / |

# Software Architecture

## Workflow

![Workflow and Dependency](documents/resources/image/workflow-deps.jpg?row=true)

1. **Model Loading and Serialization**: This procedure involves loading model weights, setting up transformation parameters, and quantization settings. Based on this information, the model is serialized and converted into the DashInfer format (.dimodel, .ditensors). This functionality is accessible exclusively through a Python interface and relies on the PyTorch and transformers libraries to access the weights. The version requirements for PyTorch and transformers may vary from model to model. DashInfer itself does not impose any specific version constraints.

2. **Model Inference**: This step is responsible for executing the model inference using the serialized model with DashInfer, without depending on components like PyTorch. DashInfer employs [DLPack](https://github.com/dmlc/dlpack) format tensors to facilitate interaction with external frameworks, such as PyTorch. Tensors in DLPack format can be manually created or generated through tensor conversion functions provided by deep learning frameworks. Regarding the C++ interface, since most dependencies have been statically linked, it primarily relies on the OpenMP runtime library and C++ system libraries. We applied [control over symbol exports](https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/) to ensure that only DashInfer's API interface symbols are visible, thereby preventing version conflicts with existing libraries in the user's system, such as protobuf.

> Note:
> - .dimodel, .ditensors is a special model format defined by DashInfer kernel.
> - When utilizing the Python interface, you can combine the code from steps 1 and 2. However, due to the lack of functionality for loading Huggingface models at the C++ level, the C++ interface is limited to conducting inferences with models in the DashInfer format. Therefore, it's essential to serialize the model first using the Python interface before proceeding with the C++ interface.

## Single-NUMA Architecture

![Single-NUMA Arch](documents/resources/image/arch-single-numa.jpg?row=true)

In the model inference phase, an inference request can be initiated by passing in input tokens and generation parameters via `StartRequest`, and when the request is successful, the DashInfer engine will return an output queue `ResultQueue` and a control handle `RequestHandle`.

- The `ResultQueue` is used to get output tokens and the status of the generation. DashInfer will **asynchronously** put the generated token into the queue, and tokens in the queue can be fetched either in a blocking (`ResultQueue.Get()`) or non-blocking (`ResultQueue.GetNoWait()`) way.

- The `RequestHandle` is the handle used to manage the request. DashInfer `engine` provides Sync, Stop, and Release primitives for the request specified by the `RequestHandle`. The `SyncRequest` primitive, which returns at the end of generation (when the number of generated tokens reaches the limit, or when an EOS has been generated), is used to simulate the behavior of the synchronous interface.

In single-NUMA mode, DashInfer Runtime uses multi-threading and a thread pool for scheduling.

## Multi-NUMA Architecture

![Multi-NUMA Arch](documents/resources/image/arch-multi-numa.jpg?row=true)

Due to the inability of some Linux kernels to control CPU affinity at the thread level, running engine on multi-NUMA CPUs may result in remote memory node access, thereby causing a decline in performance. To enable precise control of a thread's CPU affinity, DashInfer multi-NUMA solution employs a multi-process client-server architecture to achieve tensor parallel model inference. On each NUMA node, an independent process runs the server, with each server handling a part of the tensor parallel inference, and the processes use OpenMPI to collaborate (e.g., via the allreduce operation). The client interacts with the servers via gRPC, providing a unique external interface to avoid the need to manage multiple processes when invoking the DashInfer interface.

In terms of API, multi-NUMA and single-NUMA inference need to use different header files and .so libraries (or call different python interfaces). Except for the header and the library, the rest of the interface is consistent and no code changes are required. For details, you can refer to the examples.

- Single-NUMA
    - header: allspark/allspark.h
    - .so library: liballspark_framework.so
    - python API: allspark.Engine()
- MultiNUMA
    - header: allspark/allspark_client.h
    - .so library: liballspark_client.so
    - python API: allspark.ClientEngine()

> Note: C++ liballspark_framework.so (called for single-NUMA inference) and liballspark_client.so (called for multi-NUMA inference) are mutually exclusive, you cannot link both libraries.

# Performance Test

Please refer to [documentation](documents/EN/performance.md) for detailed performance test results.

The results of this performance test can be reproduced with the scripts in `<path_to_dashinfer>/examples/python/1_performance`.

# Inference Accuracy

Tested model: [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

| Engine | DataType | MMLU | C-Eval | GSM8K | HumanEval |
|:------:|:--------:|:----:|:------:|:-----:|:---------:|
| transformers | BF16 | 55.8 | 59.7 | 50.3 | 37.2 |
| DashInfer | A16W8 | 55.78 | 61.10 | 51.25 | 37.19 |

- A16W8: The model weight is quantized to 8-bit and is recovered as bfloat16 for matrix multiplication during inference.
- The results of this accuracy evaluation can be reproduced with the scripts in `<path_to_dashinfer>/examples/python/2_evaluation`.

# Examples

In `<path_to_dashinfer>/examples` there are examples for C++ and Python interfaces, and please refer to the documentation in `<path_to_dashinfer>/documents/EN` to run the examples.

- [Basic Python Example](examples/python/0_basic/basic_example_qwen_v10_io.ipynb) [![Open In PAI-DSW](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/Open-in-DSW20px.svg)](https://gallery.pai-ml.com/#/import/https://github.com/modelscope/dash-infer/blob/main/examples/python/0_basic/basic_example_qwen_v10_io.ipynb)
- [Documentation for All Python Examples](documents/EN/examples_python.md)
- [Documentation for C++ Examples](documents/EN/examples_cpp.md)

# Third-party Dependencies

This subsection lists the third-party dependencies for the different stages of DashInfer.

> Note: These dependency packages are managed through conan and are automatically downloaded when compiling DashInfer.

## Code Compilation Phase

- [conan](https://conan.io/) (1.60.0): For managing C++ third-party dependencies.
- [cmake](https://cmake.org/) (3.18+): Build system.

## Model Conversion Phase

- [PyTorch](https://pytorch.org/) (CPU): For loading model files, no special version requirements.
- [transformers](https://github.com/huggingface/transformers): For loading model parameters and tokenizer.

## Model Inference Phase

- [protobuf](https://protobuf.dev/)(3.18): For parsing model files.
- [pybind11](https://github.com/pybind/pybind11)(2.8): For binding python interfaces.
- [onednn](https://github.com/oneapi-src/oneDNN), [mkl](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html): BLAS libraries, for accelerating GEMM calculations.
- [openmp](https://www.openmp.org/): A standard parallel programming library.
- [openmpi](https://www.open-mpi.org/): For implementing multi-NUMA service architecture.
- [grpc](https://grpc.io/): For implementing multi-NUMA service architecture.

# Future Plans

- [ ] Accelerate attention with Flash-Attention
- [x] Expand context length to over 32k
- [ ] Support 4-bit quantization
- [ ] Support quantized models fine-tuned with GPTQ
- [ ] Support MoE architecture

# License

The DashInfer source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
