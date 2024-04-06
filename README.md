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

DashInfer is a production-level Large Language Pre-trained Model (LLM) inference engine developed by Tongyi Laboratory, which is currently applied to the backend inference of Alibaba Tongyi-Qwen, Tongyi-Lingma, and DashScope Platform.

DashInfer is written in C++ Runtime, and provides both C++ and Python language interfaces, aiming to deliver a production-level implementation characterized by high inference performance, high model accuracy, high stability, and high portability with minimal third-party dependencies.

The open-source version of this project encompasses the CPU (x86, ARMv9) inference aspect of the engine, marking it as the first CPU LLM inference engine in the open-source community to support both **Continuous Batching** and **NUMA Aware** capabilities. DashInfer can fully utilize the performance of server CPUs to provide more hardware options for inferencing LLM models up to 14B. It has already been applied to some online API service on the DashScope Platform.

## Main Features

- **Lightweight Architecture**: DashInfer requires minimal third-party dependencies and uses static linking for dependency libraries. By providing C++ and Python interfaces, DashInfer can be easily integrated into your existed system.
- **High-Precision**: DashInfer has been rigorously tested for accuracy and is able to provide inference accuracy consistent with PyTorch, GPU engine (vLLM).
- **Standard LLM Inference Techniques**: It employs standard LLM inference techniques, such as:
  - Continuous Batching allows for the immediate insertion of new requests and supports streaming outputs.
  - Request-based asynchronous interfaces offer individual control over generation parameters and request statuses for each request.
- **Support for Mainstream LLM Open-Source Models**: DashInfer supports mainstream open-source LLM models, including Qwen, LLaMA, ChatGLM, etc., and supports reading models in the Huggingface format.
- **PTQ Quantization**: Using DashInfer's InstantQuant (IQ), weight-only quantization acceleration can be achieved without training fine-tuning, improving deployment efficiency. After accuracy evaluation, IQ has no impact on model accuracy. The current version supports weight-only 8-bit quantization on ARM CPUs.
- **Optimized Computation Kernels**: With OneDNN and self-developed assembly kernels, DashInfer is able to maximize the performance of the hardware on both ARM and x86.
- **NUMA Aware Design**: DashInfer supports tensor parallel inference across multiple NUMA nodes, fully leveraging the computational power of server CPUs. By numactl and a multi-process architecture, the NUMA affinity of threads is accurately controlled to fully utilize the performance of multi-node CPUs and avoid the performance degradation caused by cross-NUMA access. For more information on NUMA, see: [Optimizing Applications for NUMA - Intel](https://www.intel.com/content/dam/develop/external/us/en/documents/3-5-memmgt-optimizing-applications-for-numa-184398.pdf), [What is NUMA?](https://www.kernel.org/doc/html/v5.0/vm/numa.html).
- **Context Length**: The current version supports up to 11k context length, with plans to support even longer context lengths in the future.
- **Multi-Language API Interfaces**: Provide C++ and Python interfaces. You can directly use the C++ interface to Java, Rust and other programming languages.
- **Operating System Support**: Supports mainstream Linux server operating systems like Centos7 and Ubuntu22.04, and provides corresponding Docker images.

## Documents

- [installation.md](documents/EN/installation.md)
- [examples_cpp.md](documents/EN/examples_cpp.md)
- [examples_python.md](documents/EN/examples_python.md)
- [performance.md](documents/EN/performance.md)

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

To perform weight-only quantization, the `do_dynamic_quantize_convert` and `quantization_config` fields of the model configuration file need to be modified. Refer to the [documentation] (documents/EN/examples_python.md) for a detailed description of the parameters.

The weight-only quantization, which will find the maximum and minimum values of weight within the range of GroupSize and map the weight values to the uint8 value field range, is calculated as follows:

$$ scale = \frac {x_{fp32_{max}} - x_{fp32_{min}}} {255 - 0} $$
$$ zeropoint = 0 - \frac {x_{fp32_{min}}} {scale} $$
$$ x_{u8} = x_{fp32} / scale + zeropoint $$

During inference, the quantized weight is recovered as bfloat16 for matrix multiplication.

# Supported Models

| Architecture | Models | HuggingFace Models | ModelScope Models |
|:------------:|:------:|:------------------:|:-----------------:|
| QWenLMHeadModel | Qwen | [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc. | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| Qwen2ForCausalLM | Qwen1.5 | [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),<br>[Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),<br>[Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat),<br>[Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat),<br>[Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), etc. | [qwen/Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary),<br>[qwen/Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary),<br>[qwen/Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary),<br>[qwen/Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary),<br>[qwen/Qwen1.5-14B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary), etc. |
| ChatGLMModel | ChatGLM | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b),<br>[THUDM/chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k),<br>[THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b),<br>[THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary),<br>[ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary),<br>[ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary),<br>[ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary) |
| LlamaForCausalLM | LLaMA-2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [modelscope/Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary),<br>[modelscope/Llama-2-13b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary) |

# Software Architecture

## Workflow

![Workflow and Dependency](documents/resources/image/workflow-deps.jpg?row=true)

1. **Model Loading and Serialization**: This procedure involves reading model weights, setting up transformation parameters, and quantization settings. Based on this information, the model is serialized and converted into the DashInfer format. This functionality is accessible exclusively through a Python interface and relies on the PyTorch and transformers libraries to access the weights. The version requirements for PyTorch and transformers may vary from model to model. DashInfer itself does not impose any specific version constraints.

2. **Model Inference**: This step is responsible for executing the model inference using the serialized model with DashInfer, without depending on components like PyTorch. DashInfer employs [DLPack](https://github.com/dmlc/dlpack) format tensors to facilitate interaction with external frameworks, such as PyTorch. Tensors in DLPack format can be manually created or generated through conversion functions of tensors from deep learning frameworks. Regarding the C++ interface, since most dependencies have been statically compiled, it primarily relies on the OpenMP runtime library and C++ system libraries. We performed [link symbol handling](https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/) to ensure that only DashInfer's API interface symbols are visible, thereby preventing version conflicts with existing public libraries in the user's system, such as protobuf.

> Note: When utilizing the Python interface, you can combine the code from steps 1 and 2. However, due to the lack of functionality for loading Huggingface models at the C++ level, the C++ interface is limited to conducting inferences with models in the DashInfer format. Therefore, it's essential to serialize the model first using the Python interface before proceeding with the C++ interface.

## Single-NUMA Architecture

![Single-NUMA Arch](documents/resources/image/arch-single-numa.jpg?row=true)

In the model inference phase, an inference request can be initiated by passing in input tokens and generation parameters via `StartRequest`, and when the request is successful, the DashInfer engine will return an output queue `ResultQueue` and a control handle `RequestHandle`.

- The `ResultQueue` is used to get output tokens and the status of the generation, DashInfer will **asynchronously** put the generated token into the queue, and tokens in the queue can be fetched either in a blocking (`ResultQueue.Get()`) or non-blocking (`ResultQueue.GetNoWait()`) way.

- The `RequestHandle` is the handle used to manage the request. DashInfer `engine` implements Sync, Stop and Release operations on the specified request based on the incoming `RequestHandle`. The `SyncRequest` operation, which returns at the end of generation (when the number of generated tokens reaches the upper limit, or when a EOS has been generated), is used to simulate the behavior of the synchronization interface.

In single-NUMA mode, DashInfer Runtime uses a multi-threading and thread pooling structure for scheduling.

## Multi-NUMA Architecture

![Multi-NUMA Arch](documents/resources/image/arch-multi-numa.jpg?row=true)

Due to the inability of some Linux kernels to control CPU affinity at the thread level, running engine on multi-NUMA CPUs may result in remote memory node access, thereby causing a decline in performance. To enable precise control of a thread's CPU affinity, DashInfer multi-NUMA solution employs a multi-process client-server architecture to achieve tensor parallel model inference. On each NUMA node, an independent process runs the server, with each server handling a part of the tensor parallel inference, and the processes use OpenMPI to collaborate (e.g., allreduce operations). The client interacts with the server via gRPC, providing a unique external interface to avoid the need to manage multiple processes when calling the DashInfer interface.

In terms of API usage, multi-NUMA and single-NUMA inference need to reference different header files, .so libraries (or call different python interfaces). Except for the referencing phase, the rest of the interface is consistent and no code changes are required. For details, you can refer to the examples.

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

tested model: [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

| Engine | DataType | MMLU | C-Eval | GSM8K | HumanEval |
|:------:|:--------:|:----:|:------:|:-----:|:---------:|
| transformers | BF16 | 55.8 | 59.7 | 50.3 | 37.2 |
| DashInfer | A16W8 | 55.78 | 61.10 | 51.25 | 37.19 |

- A16W8: The model weight is quantized to 8-bit and is recovered as bfloat16 for matrix multiplication during inference.
- The results of this accuracy evaluation can be reproduced with the scripts in `<path_to_dashinfer>/examples/python/2_evaluation`.

# Examples

Under `<path_to_dashinfer>/examples` there are examples of calling C++, Python interface, please refer to the documents under the `<path_to_dashinfer>/documents/EN` to run the examples. ([Document for C++ Examples](documents/EN/examples_cpp.md)、[[Document for Python Examples](documents/EN/examples_python.md))

# Third-party Dependencies

This subsection lists the third-party dependencies for the different stages of DashInfer.

> Note: These dependency packages are managed through conan and are automatically downloaded when compiling DashInfer.

## Code Compilation Phase

- [conan](https://conan.io/) (1.60.0): For managing c++ third-party dependencies.
- [cmake](https://cmake.org/) (3.18+): Build system.

## Model Conversion Phase

- [PyTorch](https://pytorch.org/) (cpu): For reading model files, no special version requirements.
- [transformers](https://github.com/huggingface/transformers): For loading model parameters and tokenizer.

## Model Inference Phase

- [protobuf](https://protobuf.dev/)(3.18): For parsing model files.
- [pybind11](https://github.com/pybind/pybind11)(2.8): For binding python interfaces.
- [onednn](https://github.com/oneapi-src/oneDNN), [mkl](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html): BLAS libraries, for accelerating GEMM calculations.
- [openmp](https://www.openmp.org/): A standard parallel programming library.
- [openmpi](https://www.open-mpi.org/): For realizing multi-NUMA service architecture.
- [grpc](https://grpc.io/): For realizing multi-NUMA service architecture.

# Future Plans

- [ ] Accelerate attention calculations with Flash-Attention
- [ ] Expand context length to over 32k
- [ ] Support quantized models fine-tuned with the GPTQ algorithm
- [ ] Support MoE architecture

# License

The DashInfer source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
