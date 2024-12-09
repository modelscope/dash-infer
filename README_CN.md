<div align="center">

[![PyPI](https://img.shields.io/pypi/v/dashinfer)](https://pypi.org/project/dashinfer/)
[![Documentation Status](https://readthedocs.org/projects/dashinfer/badge/?version=latest)](https://dashinfer.readthedocs.io/en/latest/)

<h4 align="center">
    <p>
        <a href="https://github.com/modelscope/dash-infer/blob/main/README.md">English</a> |
        <b>中文</b>
    </p>
</h4>


</div>

# 简介

DashInfer采用C++ Runtime 编写，提供C++ 和 Python语言接口。 DashInfer 具有生产级别的高性能表现， 支持多种CUDA架构， CPU架构。 DashInfer支持多种主流LLM推理技术连续批处理（Continuous Batching），权重量化， KV-Cache量化， Page Attention（自研SpanAttention Kernel），Guided Output， Prefix Caching。

## DashInfer的主要特征

## 主要特性
DashInfer 是一个高度优化的 LLM 推理引擎，具有以下核心特性：
- **轻量级架构**: DashInfer 需要最少的第三方依赖，并使用静态链接几乎所有的依赖库。通过提供 C++ 和 Python 接口，DashInfer 可以轻松集成到您现有的系统中。
- **高精度**: DashInfer 经过严格测试以确保准确性，能够提供与 PyTorch 和其他 GPU 引擎（例如 vLLM）一致的推理精度。
- **高性能**: DashInfer 采用优化的内核提供高性能 LLM 服务，同时支持许多标准 LLM 推理技术，包括：
  - **连续批处理**: DashInfer 允许即时插入新请求，并支持流式输出。
  - **分页注意力机制**: 使用我们自研的分页注意力机制（我们称之为 *SpanAttention*），结合基于高效 GEMM 和 GEMV 实现的 int8 和 uint4 KV 缓存量化，能够实现注意力运算符的高效加速。
  - **前缀缓存**: DashInfer 支持高效的前缀缓存，用于加速标准 LLMs 和多模态 LMs（如 Qwen-VL），支持 GPU 和 CPU。
  - **量化支持**: 使用 DashInfer 的 *InstantQuant*（IQ），无需微调即可实现权重量化加速，提高部署效率。准确率评估显示，IQ 几乎不影响模型准确率，详细信息请参见：:doc:`quant/weight_activate_quant`。
  - **异步接口**: 基于请求的异步接口提供对每个请求生成参数和请求状态的独立控制。
- 支持的模型：
  - **主流开源 LLMs**: DashInfer 支持主流开源 LLMs，包括 Qwen、LLaMA、ChatGLM 等，且支持加载 Huggingface 格式的模型。
  - **多模态大模型(VLMs)**: DashInfer 支持多模态语言模型（VLMs），包括 Qwen-VL、Qwen-AL 和 Qwen2-VL。
- **OpenAI API 服务器**: DashInfer 可以轻松与 fastChat 配合使用，实现兼容 OpenAI 的 API 服务器。
- **多编程语言 API**: 提供 C++ 和 Python 接口。通过标准的跨语言接口，可以将 C++ 接口扩展到 Java、Rust 等编程语言。

## 文档
- [Release Note](https://dashinfer.readthedocs.io/en/latest/#release-note)
- [User Manual](https://dashinfer.readthedocs.io/en/latest/)
- [安装](docs/CN/installation.md)
- [C++示例](docs/CN/examples_cpp.md)
- [Python示例](docs/CN/examples_python.md)
- [性能测试](docs/EN/performance.md)
- [使用魔搭notebook部署](docs/CN/modelscope_notebook.md)

# 硬件支持和数据类型

## 硬件支持
- **CUDA GPU**：支持 CUDA 版本从 11.4 到 12.4，并支持多种 CUDA 计算架构，例如 SM70 - SM90a（T4、3090、4090、V100、A100、A10、L20、H20、H100）
- **x86 CPU**：要求硬件至少需要支持AVX2指令集。对于第五代至强（Xeon）处理器（Emerald Rapids）、第四代至强（Xeon）处理器（Sapphire Rapids）等（对应于阿里云第8代ECS实例，如g8i），采用AMX矩阵指令加速计算。
- **ARMv9 CPU**：要求硬件支持SVE指令集。支持如倚天（Yitian）710等ARMv9架构处理器（对应于阿里云第8代ECS实例，如g8y），采用SVE向量指令加速计算。

## 数据类型
- **CUDA GPUs**: FP16, BF16, FP32, Int8(InstantQuant), Int4(InstantQuant)
- **x86 CPU**：支持FP32、BF16。
- **ARM Yitian710 CPU**：FP32、BF16、InstantQuant。

### InstantQuant
DashInfer 为 LLM 权重提供了多种量化技术，例如 int{8,4} 仅权重量化、int8 激活量化，还有许多定制的融合内核，以在指定设备上提供最佳性能。简而言之，使用 GPTQ 微调的模型将提供更好的准确性，而我们无需微调的 InstantQuant (IQ) 技术可提供更快的部署体验。IQ 量化的详细解释可以在本文末尾找到。

在支持的量化算法方面，AllSpark 通过两种方式支持使用 GPTQ 微调的模型和使用 IQ 量化技术的动态量化：
- **InstantQuant (IQ)**: AllSpark 提供了 InstantQuant (IQ) 动态量化技术，无需微调即可提供更快的部署体验。IQ 量化的详细解释可以在本文末尾找到。
- **GPTQ**: 使用 GPTQ 微调的模型将提供更好的准确性，但它需要一个微调步骤。

这里介绍的量化策略大致可以分为两类：
- **仅权重量化**: 这种量化技术仅对权重进行量化和压缩，例如以 int8 格式存储权重，但在计算时仍旧使用 bf16/fp16。它只是减少了内存访问需求，相比 BF16 并没有提高计算性能。
- **激活量化**: 这种量化技术不仅以 int8 格式存储权重，还在计算阶段执行低精度量化计算（如 int8）。由于 Nvidia GPU 只有 int8 Tensor Core 容易保持精度，这种量化技术既能减少内存访问需求，又能提高计算性能，使其成为理想的量化方法。在准确性方面，它相比仅权重量化可能会有轻微下降，因此需要业务数据的准确性测试。

在量化粒度方面，有两种类型：
- **每通道量化**: AllSpark 的量化技术至少采用了每通道（也称为每 Token）量化粒度，有些还提供了子通道量化粒度。一般而言，每通道量化由于实现简单且性能最佳，通常能满足大多数准确性需求。只有当每通道量化的准确性不足时，才应考虑子通道量化策略。
- **子通道量化**: 与每通道量化相比，子通道量化是指将一个通道划分为 N 组，并在每组内计算量化参数。这种量化粒度通常能提供更好的准确性，但由于实现复杂度增加，带来了许多限制。例如，性能可能比每通道量化稍慢，并且由于计算公式限制，激活量化难以实现子通道量化（AllSpark 的激活量化都是每通道量化）。

# 模型支持
DashInfer 支持两种模型加载方式：
1. **HF 格式**：直接从 Hugging Face 加载模型，这是最方便的方法，模型可以从 Hugging Face 或 ModelScope 下载。
2. **DashInfer 格式**：由 DashInfer 序列化的模型文件，依赖更少的 Python 组件，可以通过 C++ 库加载。

| Architecture |     Models      |                                                                                                                                                                                 HuggingFace Models                                                                                                                                                                                  | ModelScope Models |
|:------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| QWenLMHeadModel |      Qwen       |                                                                                    [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc.                                                                                    | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| Qwen2ForCausalLM | Qwen1.5-Qwen2.5 | [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),<br>[Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),<br>[Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat),<br>[Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat),<br>[Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), etc. | [qwen/Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary),<br>[qwen/Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary),<br>[qwen/Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary),<br>[qwen/Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary),<br>[qwen/Qwen1.5-14B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary), etc. |
| Qwen2VLForConditionalGeneration |     QwenVL      |                                                                                    [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc.                                                                                    | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| ChatGLMModel |     ChatGLM     |                                                                                                                                                          [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)                                                                                                                                                          | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/summary) |
| LlamaForCausalLM |     LLaMA-2     |                                                                                                  [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)                                                                                                  | [modelscope/Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary),<br>[modelscope/Llama-2-13b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary) |
| LlamaForCausalLM |     LLaMA-3     |                                                                                                                                          [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                                                                                                                                          | [modelscope/Meta-Llama-3-8B-Instruct](https://modelscope.cn/models/modelscope/Meta-Llama-3-8B-Instruct/summary) |
| BaichuanForCausalLM |    Baichuan2    |                                                                                               [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), <br>[baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)                                                                                                | [baichuan-inc/Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat), <br>[baichuan-inc/Baichuan2-13B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat) |

# 软件框架

## 推理流程

![Workflow and Dependency](documents/resources/image/workflow-deps.jpg?row=true)

1. **模型加载**：该过程包括加载模型权重、设置转换参数和量化设置。基于这些信息，模型会被序列化并转换成 DashInfer 格式（.dimodel, .ditensors 或 .asparams, .asmodel） 。此功能仅通过 Python 接口访问，并依赖于 PyTorch 和 transformers 库来访问权重。PyTorch 和 transformers 的版本要求可能因模型而异。DashInfer 本身没有具体的版本限制。
2. **模型推理**：此步骤负责使用 DashInfer 执行序列化模型的推理，而不依赖于 PyTorch 等组件。DashInfer 采用 [DLPack](https://github.com/dmlc/dlpack) 格式的张量，以便与外部框架（如 PyTorch）进行交互。DLPack 格式的张量可以手动创建，也可以通过深度学习框架提供的张量转换函数生成。对于 C++ 接口，由于大多数依赖项已经被静态链接，它主要依赖于 OpenMP 运行时库和 C++ 系统库。我们应用了 [控制符号导出](https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/) 技术，以确保只有 DashInfer 的 API 接口符号是可见的，从而防止与用户系统中的现有库（如 protobuf）发生版本冲突。

> 注意：
> - 版本 2.0 之后，用户很少需要关心模型类型(在1.0中），它会被 DashInfer Runtime 自动检测。
> - ~~.dimodel, .ditensors 是 DashInfer 内核定义的一种特殊模型格式。~~
> - 使用 Python 接口时，可以将步骤 1 和步骤 2 的代码结合起来。然而，由于在 C++ 层面缺乏加载 Huggingface 模型的功能，C++ 接口仅限于使用 DashInfer 格式的模型进行推理。因此，必须先使用 Python 接口序列化模型，然后再进行 C++ 接口的推理。
## GPU 和 CPU 单NUMA架构图

![Single-NUMA Arch](docs/resources/image/arch-single-numa.jpg?row=true)

GPU 和单 NUMA CPU 推理共享相同的接口和架构。在模型推理阶段，可以通过 `StartRequest` 传入输入标记和生成参数来启动推理请求，当请求成功时，DashInfer 引擎将返回一个输出队列 `ResultQueue` 和一个控制句柄 `RequestHandle`。

- `ResultQueue`用来获取输出token以及生成的状态，推理引擎会**异步**地把生成的token放到该队列中，可以阻塞（`ResultQueue.Get()`）或非阻塞（`ResultQueue.GetNoWait()`）地获取队列中的token。

- `RequestHandle`是用来管理请求的句柄，DashInfer `engine`根据传入的`RequestHandle`实现对指定request的同步（Sync）、停止（Stop）和释放（Release）操作。其中`SyncRequest`操作，会在生成结束（生成的token数达到上限，或产生结束符）后返回，用来模拟同步接口的行为。

在GPU 和 单NUMA的模式下，DashInfer Runtime采用多线程和线程池的结构做调度。

## 多NUMA架构图

![Multi-NUMA Arch](docs/resources/image/arch-multi-numa.jpg?row=true)

由于部分Linux内核无法在线程级别控制CPU亲和性，在多NUMA的CPU上采用单进程推理可能会出现跨NUMA访问内存访问，从而导致性能下降。为了能够精确地控制程序的CPU亲和性，DashInfer的多NUMA方案采用了多进程的client-server架构，实现tensor parallel的模型推理。在每个NUMA节点上，都有一个独立的进程运行DashInfer server，每个server负责一部分的tensor parallel推理，进程间使用OpenMPI进行协同（例如allreduce操作）。DashInfer client通过gRPC与server交互，提供唯一的对外接口，避免在调用DashInfer接口时，需要对多进程进行管理。

在API使用上，多NUMA和单NUMA的推理需要引用不同的头文件、.so库（或调用不同的python接口）。除了引用阶段外，其余接口一致，无需修改代码。具体可以参考examples中的示例。

- 单NUMA
  - 头文件：allspark/allspark.h
  - .so库：liballspark_framework.so
  - python接口：allspark.Engine()
- 多NUMA
  - 头文件：allspark/allspark_client.h
  - .so库：liballspark_client.so
  - python接口：allspark.ClientEngine()

> 注意：C++的liballspark_framework.so（单NUMA推理时调用）和liballspark_client.so（多NUMA推理时调用）是互斥的，不能同时链接两个库。

# 性能测试

详细的性能测试结果请参考[文档](docs/EN/performance.md)。

该性能测试结果可用`<path_to_dashinfer>/examples/python/1_performance`中的脚本复现。

# 精度测试

测试模型：[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

| Engine | DataType | MMLU | C-Eval | GSM8K | HumanEval |
|:------:|:--------:|:----:|:------:|:-----:|:---------:|
| transformers | BF16 | 55.8 | 59.7 | 50.3 | 37.2 |
| DashInfer | A16W8 | 55.78 | 61.10 | 51.25 | 37.19 |

- A16W8：指weight采用8-bit量化，在推理过程中恢复为bfloat16进行矩阵乘法计算；
- 该精度评测结果，可用`<path_to_dashinfer>/examples/python/2_evaluation`中的脚本复现。

# 示例代码

在`<path_to_dashinfer>/examples`下提供了C++、python接口的调用示例，请参考`<path_to_dashinfer>/documents/CN`目录下的文档运行示例。

- [基础Python示例](examples/python/0_basic/basic_example_qwen_v10_io.ipynb) [![Open In PAI-DSW](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/Open-in-DSW20px.svg)](https://gallery.pai-ml.com/#/import/https://github.com/modelscope/dash-infer/blob/main/examples/python/0_basic/basic_example_qwen_v10_io.ipynb)
- [所有Python示例文档](docs/CN/examples_python.md)
- [C++示例文档](docs/CN/examples_cpp.md)

# 依赖库

本小节列出了DashInfer不同阶段的第三方依赖。

> 注：这些依赖包通过conan管理，在编译DashInfer时自动下载。

## 代码编译阶段

- [conan](https://conan.io/) (1.60.0): For managing C++ third-party dependencies.
- [cmake](https://cmake.org/) (3.18+): Build system.

## 模型转换阶段

- [PyTorch](https://pytorch.org/) (CPU): For reading model files, no special version requirements.
- [transformers](https://github.com/huggingface/transformers): For loading model parameters and tokenizer.

## 模型推理阶段

- [protobuf](https://protobuf.dev/)(3.18.3): For parsing model files.
- [pybind11](https://github.com/pybind/pybind11)(2.8): For binding python interfaces.
- [onednn](https://github.com/oneapi-src/oneDNN), [mkl](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html): BLAS libraries, for accelerating GEMM calculations.
- [openmp](https://www.openmp.org/): A standard parallel programming library.
- [openmpi](https://www.open-mpi.org/): For implementing multi-NUMA service architecture.
- [grpc](https://grpc.io/): For implementing multi-NUMA service architecture.

# 未来规划

- [x] GPU 支持
- [x] 多模态模型支持
- [x] 使用 Flash-Attention 加速注意力机制
- [x] 将上下文长度扩展到超过 32k
- [x] 支持 4 位量化
- [x] 支持使用 GPTQ 微调的量化模型
- [x] 支持 MoE 架构
- [x] 引导输出：Json 模式
- [x] 前缀缓存：支持 GPU 前缀缓存和 CPU 交换
- [ ] 量化：CUDA 上的 Fp8 支持
- [ ] LORA：持续批量 LORA 优化


# License

DashInfer源代码采用Apache 2.0协议授权，您可在该仓库根目录找到协议全文。
