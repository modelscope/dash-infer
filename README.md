<div align="center">

[![PyPI](https://img.shields.io/pypi/v/dashinfer)](https://pypi.org/project/dashinfer/)
[![Documentation Status](https://readthedocs.org/projects/dashinfer/badge/?version=latest)](https://dashinfer.readthedocs.io/en/latest/) 

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/modelscope/dash-infer/blob/main/README_CN.md">中文</a>
    </p>
</h4>


</div>

# Introduction

Written in C++ runtime, DashInfer aims to deliver production-level implementations highly optimized for various hardware architectures, including CUDA, x86 and ARMv9.

## Main Features
DashInfer is a highly optimized LLM inference engine with the following core features:

- **Lightweight Architecture**: DashInfer requires minimal third-party dependencies and uses static linking for almost all dependency libraries. By providing C++ and Python interfaces, DashInfer can be easily integrated into your existing system.

- **High Precision**: DashInfer has been rigorously tested to ensure accuracy, and is able to provide inference whose accuracy is consistent with PyTorch and other GPU engines (e.g., vLLM).

- **High Performance**: DashInfer employs optmized kernels to provide high-performance LLM serving, as well as lots of standard LLM inference techniques, including:

  - **Continuous Batching**: DashInfer allows for the immediate insertion of new requests and supports streaming outputs.

  - **Paged Attention**: Using our self-developed paged attention technique (which we call *SpanAttention*), we can achieve efficient acceleration of attention operator, combined with int8 and uint4 KV cache quantization, based on highly efficient GEMM and GEMV implementations.

  - **Prefix Cache**: DashInfer supports highly efficient Prefix Cache for prompts, which accelerates standard LLMs and MultiModal LMs (MMLMs) like Qwen-VL, using both GPU and CPU.

  - **Quantization Support**: Using DashInfer's *InstantQuant* (IQ), weight-only quantization acceleration can be achieved without fine-tuning, improving deployment efficiency. Accuracy evaluation shows that IQ has almost no impact on model accuracy, for detail, see :doc:`quant/weight_activate_quant`.

  - **Asynchronous Interface**: Request-based asynchronous interfaces offer individual control over generation parameters and request status of each request.

- Supported Models:

  - **Mainstream Open-Source LLMs**: DashInfer supports mainstream open-source LLMs including Qwen, LLaMA, ChatGLM, etc., and supports loading models in the Huggingface format.

  - **MultiModal LMs**: DashInfer supports MultiModal Language Models (MMLMs) including Qwen-VL, Qwen-AL, and Qwen2-VL.

- **OpenAI API Server**: DashInfer can easily serve with fastChat to achieve OpenAI-compatible API server.

- **Multi-Programming-Language API**: Both C++ and Python interfaces are provided. It is possible to extend C++ interface to Java, Rust and other programming languages, via standard cross-language interfaces.



## Documentation
- [Release Note](https://dashinfer.readthedocs.io/en/latest/#release-note)
- [User Manual](https://dashinfer.readthedocs.io/en/latest/)
- [Installation](docs/EN/installation.md)
- [C++ Examples](docs/EN/examples_cpp.md)
- [Python Examples](docs/EN/examples_python.md)
- [Performance](docs/EN/performance.md)

# Supported Hardware and Data Types

## Hardware
- **CUDA GPUs**: Support CUDA Version from 11.4 - 12.4, and supports various CUDA compute architectures like SM70 - SM90a (T4, 3090, 4090, V100, A100, A10, L20, H20, H100)
- **x86 CPUs**: Hardware support for AVX2 instruction set is required. For Intel's 5th generation Xeon processors (Emerald Rapids), 4th generation Xeon processors (Sapphire Rapids), corresponding to Aliyun's 8th generation ECS instances (e.g., g8i), AMX instructions are used to accelerate caculation.
- **ARMv9 CPU**: Hardware support for SVE instruction set is required. DashInfer supports ARMv9 architecture processors such as Yitian710, corresponding to Aliyun's 8th generation ECS instances (e.g. g8y), and adopts SVE instruction to accelerate caculation.

## Data Types
- **CUDA GPUs**: FP16, BF16, FP32, Int8(InstantQuant), Int4(InstantQuant)
- **x86 CPU**: FP32, BF16
- **ARM Yitian710 CPU**: FP32, BF16, InstantQuant

### Quantization
DashInfer provides various many quantization technology for LLM weight, such as, int{8,4} weight only quantization, int8 activate quantization, and many customized fused kernel to provide best performance on specified device.

To put it simply, models fine-tuned with GPTQ will provide better accuracy, but our InstantQuant (IQ) technique,
which does not require fine-tuning, can offer a faster deployment experience.
Detailed explanations of IQ quantization can be found at the end of this article.

In terms of supported quantization algorithms, AllSpark supports models fine-tuned with GPTQ and dynamic quantization
using the IQ quantization technique in two ways:

- **IntantQuant(IQ)**: AllSpark provides the InstantQuant (IQ) dynamic quantization technique, which does not require fine-tuning and can offer a faster deployment experience. Detailed explanations of IQ quantization can be found at the end of this article.
- **GPTQ**: Models fine-tuned with GPTQ will provide better accuracy, but it requires a fine-tuning step.

The quantization strategies introduced here can be broadly divided into two categories:

- **Weight Only Quantization**: This quantization technique only quantizes and compresses the weights,
  such as storing weights in int8 format, but uses bf16/fp16 for computations. It only reduces memory access requirements, 
  without improving computational performance compared to BF16.
- **Activation Quantization**: This quantization technique not only stores weights in int8 format but also performs low-precision quantized computations (such as int8) during the calculation phase. (Since Nvidia GPUs only have int8 Tensor Cores that can easily maintain precision, this quantization technique can reduce memory access requirements and improve computational performance, making it a more ideal quantization approach. In terms of accuracy, it may have a slight decrease compared to Weight Only quantization, so business data accuracy testing is required.


In terms of quantization granularity, there are two types:

- **Per-Channel**: AllSpark's quantization techniques at least adopt the Per-Channel (also known as Per-Token) quantization granularity, and some also provide Sub-Channel quantization granularity. Generally speaking, Per-Channel quantization can meet most accuracy requirements due to its simple implementation and optimal performance. Only when the accuracy of Per-Channel quantization is insufficient should the Sub-Channel quantization strategy be considered.
- **Sub-Channel**: Compared to Per-Channel quantization, Sub-Channel refers to dividing a channel into N groups, and calculating quantization parameters within each group. This quantization granularity typically provides better accuracy, but due to increased implementation complexity, it comes with many limitations. For example, performance may be slightly slower than Per-Channel quantization, and Activation quantization is difficult to implement Sub-Channel quantization due to computational formula constraints (AllSpark's Activation quantization is all Per-Channel).

# Supported Models

DashInfer support two kind of model load method:
1. HF format: directly load model from Hugging Face, which provides most convenient method, the model can be downloaded from huggingface or modelscope.  
2. DashInfer format:  serialized model file by DashInfer, which provided less python dependency and can be loaded by c++ library.

| Architecture |     Models      |                                                                                                                                                                                 HuggingFace Models                                                                                                                                                                                  | ModelScope Models |
|:------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| QWenLMHeadModel |      Qwen       |                                                                                    [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc.                                                                                    | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| Qwen2ForCausalLM | Qwen1.5-Qwen2.5 | [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),<br>[Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),<br>[Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat),<br>[Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat),<br>[Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), etc. | [qwen/Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary),<br>[qwen/Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary),<br>[qwen/Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary),<br>[qwen/Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary),<br>[qwen/Qwen1.5-14B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary), etc. |
| Qwen2VLForConditionalGeneration |     QwenVL      |                                                                                    [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc.                                                                                    | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| ChatGLMModel |     ChatGLM     |                                                                                                                                                          [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)                                                                                                                                                          | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/summary) |
| LlamaForCausalLM |     LLaMA-2     |                                                                                                  [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)                                                                                                  | [modelscope/Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary),<br>[modelscope/Llama-2-13b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary) |
| LlamaForCausalLM |     LLaMA-3     |                                                                                                                                          [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                                                                                                                                          | [modelscope/Meta-Llama-3-8B-Instruct](https://modelscope.cn/models/modelscope/Meta-Llama-3-8B-Instruct/summary) |
| BaichuanForCausalLM |    Baichuan2    |                                                                                               [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), <br>[baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)                                                                                                | [baichuan-inc/Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat), <br>[baichuan-inc/Baichuan2-13B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat) |

# Software Architecture

## Workflow

![Workflow and Dependency](docs/resources/image/workflow-deps.jpg?row=true)

1. **Model Loading**: This procedure involves loading model weights, setting up transformation parameters, and quantization settings. Based on this information, the model is serialized and converted into the DashInfer format (.dimodel, .ditensors, or .asparams, .asmodel). This functionality is accessible exclusively through a Python interface and relies on the PyTorch and transformers libraries to access the weights. The version requirements for PyTorch and transformers may vary from model to model. DashInfer itself does not impose any specific version constraints.

2. **Model Inference**: This step is responsible for executing the model inference using the serialized model with DashInfer, without depending on components like PyTorch. DashInfer employs [DLPack](https://github.com/dmlc/dlpack) format tensors to facilitate interaction with external frameworks, such as PyTorch. Tensors in DLPack format can be manually created or generated through tensor conversion functions provided by deep learning frameworks. Regarding the C++ interface, since most dependencies have been statically linked, it primarily relies on the OpenMP runtime library and C++ system libraries. We applied [control over symbol exports](https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/) to ensure that only DashInfer's API interface symbols are visible, thereby preventing version conflicts with existing libraries in the user's system, such as protobuf.

> Note:
> - After 2.0 version, user rarely needs to care about the model type, which will detected by DashInfer Runtime automatically. 
> - ~~.dimodel, .ditensors is a special model format defined by DashInfer kernel.~~
> - When utilizing the Python interface, you can combine the code from steps 1 and 2. However, due to the lack of functionality for loading Huggingface models at the C++ level, the C++ interface is limited to conducting inferences with models in the DashInfer format. Therefore, it's essential to serialize the model first using the Python interface before proceeding with the C++ interface.

## GPU and Single-NUMA Architecture

![Single-NUMA Arch](docs/resources/image/arch-single-numa.jpg?row=true)

GPU and Single NUMA CPU Inference share same interface and architecture, in the model inference phase, an inference request can be initiated by passing in input tokens and generation parameters via `StartRequest`, and when the request is successful, the DashInfer engine will return an output queue `ResultQueue` and a control handle `RequestHandle`.

- The `ResultQueue` is used to get output tokens and the status of the generation. DashInfer will **asynchronously** put the generated token into the queue, and tokens in the queue can be fetched either in a blocking (`ResultQueue.Get()`) or non-blocking (`ResultQueue.GetNoWait()`) way.

- The `RequestHandle` is the handle used to manage the request. DashInfer `engine` provides Sync, Stop, and Release primitives for the request specified by the `RequestHandle`. The `SyncRequest` primitive, which returns at the end of generation (when the number of generated tokens reaches the limit, or when an EOS has been generated), is used to simulate the behavior of the synchronous interface.

In GPU and single-NUMA mode, DashInfer Runtime uses multi-threading and a thread pool for scheduling.

## Multi-NUMA Architecture

![Multi-NUMA Arch](docs/resources/image/arch-multi-numa.jpg?row=true)

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

Please refer to [documentation](docs/EN/performance.md) for detailed performance test results.

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
- [Documentation for All Python Examples](docs/EN/examples_python.md)
- [Documentation for C++ Examples](docs/EN/examples_cpp.md)

## Multi-Modal Model(VLMs)) Support

The VLM Support in [multimodal](multimodal/) folder, 
it's a toolkit to support Vision Language Models (VLMs) inference based on the DashInfer engine. It's compatible with the OpenAI Chat Completion API, supporting text and image/video inputs.


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

- [protobuf](https://protobuf.dev/)(3.18.3): For parsing model files.
- [pybind11](https://github.com/pybind/pybind11)(2.8): For binding python interfaces.
- [onednn](https://github.com/oneapi-src/oneDNN), [mkl](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html): BLAS libraries, for accelerating GEMM calculations.
- [openmp](https://www.openmp.org/): A standard parallel programming library.
- [openmpi](https://www.open-mpi.org/): For implementing multi-NUMA service architecture.
- [grpc](https://grpc.io/): For implementing multi-NUMA service architecture.

# Future Plans
- [x] GPU Support
- [x] Multi Modal Model support
- [x] Accelerate attention with Flash-Attention
- [x] Expand context length to over 32k
- [x] Support 4-bit quantization
- [x] Support quantized models fine-tuned with GPTQ
- [x] Support MoE architecture
- [x] Guided output: Json Mode
- [x] Prefix Cache: Support GPU Prefix Cache and CPU Swap 
- [ ] Quantization: Fp8 support on CUDA.
- [ ] LORA: Continues Batch LORA Optimization.

# License

The DashInfer source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
