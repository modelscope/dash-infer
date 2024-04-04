# 简介

DashInfer是由通义实验室开发的生产级别的大语言预训练模型（LLM）推理引擎，目前已应用于阿里巴巴通义千问、通义灵码、灵积平台的后端推理。

DashInfer采用C++ Runtime编写，提供C++和Python语言接口，旨在提供生产级别的高性能、高模型精度、高稳定性、高可移植性，同时仅需要最小程度的第三方依赖的实现。

本项目开源版本是该引擎的CPU（x86、ARMv9）推理部分的实现，是开源社区**首个同时支持Continuous Batching 和 NUMA Aware 的 CPU LLM 推理引擎**。DashInfer可以充分发挥服务器CPU的性能，为推理14B以下的LLM模型提供更多的硬件选择。目前已在灵积平台上应用于部分LLM模型的线上API服务中。

## DashInfer的主要特征

- **轻量级架构**：仅需要最小程度的第三方依赖，并采用静态链接的方式引用依赖库。提供C++和Python接口，让DashInfer可以轻松集成到您的系统和其他编程语言中。
- **提供高精度实现**：DashInfer经过严格的精度测试，能够提供与PyTorch、GPU引擎（vLLM）一致的推理精度。
- **行业标准LLM推理技术**：采用行业标准的LLM推理技术，例如：
  - 连续批处理（Continuous Batching），能够进行即时插入新请求，支持流式输出；
  - 基于请求的异步接口允许对每个请求的生成参数、请求状态等进行单独控制。
- **支持主流LLM开源模型**：支持主流的开源LLM模型，包括Qwen、LLaMA、ChatGLM等，支持Huggingface格式的模型读取。
- **PTQ量化**：使用DashInfer的InstantQuant（IQ），无需训练微调即可实现weight-only量化加速，提高部署效率。经过精度测试，IQ对模型精度不会产生影响。目前版本支持ARM CPU上的weight-only 8-bit量化。
- **优化的计算Kernel**：结合OneDNN和自研汇编kernel，DashInfer能够在ARM和x86上发挥硬件的最大性能。
- **NUMA Aware Design**：支持多NUMA的tensor并行推理，充分发挥服务器级CPU的算力。通过numactl和多进程架构，精准控制计算线程的NUMA亲和性，充分利用多节点CPU的性能，并且避免跨NUMA访存带来性能下降问题。关于多NUMA的性能指导可以参考：[Optimizing Applications for NUMA - Intel](https://www.intel.com/content/dam/develop/external/us/en/documents/3-5-memmgt-optimizing-applications-for-numa-184398.pdf), [What is NUMA?](https://www.kernel.org/doc/html/v5.0/vm/numa.html)。
- **Context Length**：目前版本支持11k的Context Length，未来还会继续支持更长Context Length。
- **提供多语言API接口**：提供C++和Python接口，能够直接使用C++接口对接到Java、Rust等其他编程语言。
- **操作系统支持**：支持Centos7、Ubuntu22.04等主流Linux服务器操作系统，并提供对应的Docker。

## 文档

[installation.md](documents/CN/installation.md)
[examples_cpp.md](documents/CN/examples_cpp.md)
[examples_python.md](documents/CN/examples_python.md)
[performance.md](documents/EN/performance.md)

# 硬件支持和数据类型

## 硬件支持

- **x86 CPU**：要求硬件至少需要支持AVX2指令集。对于第五代至强处理器（Emerald Rapids）、第四代至强处理器（Sapphire Rapids）等（对应于阿里云第8代ECS实例（如g8i）），采用AMX矩阵指令加速计算。
- **ARMv9 CPU**：要求硬件支持SVE指令集。支持如倚天710等ARMv9架构处理器（对应于阿里云第8代ECS实例（如g8y）），采用SVE向量指令加速计算。

## 数据类型

- **x86 CPU**：支持FP32、BF16。
- **ARM Yitian710 CPU**：FP32、BF16、InstantQuant。

### InstantQuant

InstantQuant是一种weight-only量化技术。

在Yitian710 CPU（ARMv9）上，DashInfer支持weight-only量化。

要进行weight-only量化，需要修改模型配置文件的`do_dynamic_quantize_convert`和`quantization_config`字段，参数的详细说明参考[文档](documents/CN/examples_python.md)。

weight-only量化，会在GroupSize的范围内求取weight的最大、最小值，并将weight数值映射到uint8的值域范围，计算公式如下：

$$ scale = \frac {x_{fp32_{max}} - x_{fp32_{min}}} {255 - 0} $$
$$ zeropoint = 0 - \frac {x_{fp32_{min}}} {scale} $$
$$ x_{u8} = x_{fp32} / scale + zeropoint $$

推理过程中，量化的weight会被恢复成bfloat16进行矩阵乘法计算。

# 模型支持

| Architecture | Models | HuggingFace Models | ModelScope Models |
|:------------:|:------:|:------------------:|:-----------------:|
| QWenLMHeadModel | Qwen | [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat),<br>[Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat),<br>[Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat), etc. | [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary),<br>[qwen/Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary),<br>[qwen/Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), etc. |
| Qwen2ForCausalLM | Qwen1.5 | [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),<br>[Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),<br>[Qwen/Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat),<br>[Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat),<br>[Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat), etc. | [qwen/Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary),<br>[qwen/Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary),<br>[qwen/Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary),<br>[qwen/Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary),<br>[qwen/Qwen1.5-14B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary), etc. |
| ChatGLMModel | ChatGLM | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b),<br>[THUDM/chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k),<br>[THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b),<br>[THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary),<br>[ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary),<br>[ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary),<br>[ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary) |
| LlamaForCausalLM | LLaMA-2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),<br>[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [modelscope/Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary),<br>[modelscope/Llama-2-13b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary) |

# 软件框架

## 推理流程

![Workflow and Dependency](documents/resources/image/workflow-deps.jpg?row=true)

1. **模型加载与序列化**：此过程负责读取模型权重、配置模型转换参数及量化参数，并根据这些信息对模型进行序列话，并生成DashInfer格式的模型。此功能仅提供Python接口，并依赖于PyTorch和transformers库来访问权重。不同模型对PyTorch和transformers的版本要求可能有所不同，DashInfer本身并没有特殊的版本要求。

2. **模型推理**：此步骤负责执行模型推理，使用DashInfer推理序列化后的模型使用序列化的模型，不依赖PyTorch等组件。DashInfer采用[DLPack](https://github.com/dmlc/dlpack)格式的tensor来实现与外部框架（如PyTorch）的交互。DLPack格式的tensor，可以通过手动创建或由深度学习框架的tensor转换函数产生。对于C++接口，由于已经将几乎所有依赖静态编译，仅对openmp运行时库以及C++系统库的有依赖。我们进行了[链接符号处理](https://anadoxin.org/blog/control-over-symbol-exports-in-gcc.html/)，以确保只有DashInfer的API接口符号可见，避免与客户系统中已有的公共库（如protobuf等）发生版本冲突。

> 注意：使用Python接口时，可以将步骤1和2的代码放在一起。由于缺少C++层面加载Huggingface模型的功能，C++接口只能进行DashInfer格式的模型推理，因此在使用C++接口前，必须先用Python接口先对模型进行序列化。

## 单NUMA架构图

![Single-NUMA Arch](documents/resources/image/arch-single-numa.jpg?row=true)

在模型推理阶段，可以通过`StartRequest`传入请求输入token和生成参数发起推理请求，当请求成功后，DashInfer engine会返回一个输出队列`ResultQueue`和控制句柄`RequestHandle`。

- `ResultQueue`用来获取输出token以及生成的状态，推理引擎会**异步**地把生成的token放到该队列中，可以阻塞（`ResultQueue.Get()`）或非阻塞（`ResultQueue.GetNoWait()`）地获取队列中的token。

- `RequestHandle`是用来管理请求的句柄，DashInfer `engine`根据传入的`RequestHandle`实现对指定request的同步（Sync）、停止（Stop）和释放（Release）操作。其中`SyncRequest`操作，会在生成结束（生成的token数达到上限，或产生结束符）后返回，用来模拟同步接口的行为。

在单NUMA的模式下，DashInfer Runtime采用多线程和线程池的结构做调度。

## 多NUMA架构图

![Multi-NUMA Arch](documents/resources/image/arch-multi-numa.jpg?row=true)

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

详细的性能测试结果请参考[文档](documents/EN/performance.md)。

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

在`<path_to_dashinfer>/examples`下提供了C++、python接口的调用示例，请参考`<path_to_dashinfer>/documents/CN`目录下的文档运行示例。（[C++示例文档](documents/CN/examples_cpp.md)、[Python示例文档](documents/CN/examples_python.md)）

# 依赖库

本小节列出了DashInfer不同阶段的第三方依赖。

> 注：这些依赖包通过conan管理，在编译DashInfer时自动下载。

## 代码编译阶段

- [conan](https://conan.io/) (1.60.0): For managing c++ third-party dependencies.
- [cmake](https://cmake.org/) (3.18+): Build system.

## 模型转换阶段

- [PyTorch](https://pytorch.org/) (cpu): For reading model files, no special version requirements.
- [transformers](https://github.com/huggingface/transformers): For loading model parameters and tokenizer.

## 模型推理阶段

- [protobuf](https://protobuf.dev/)(3.18): For parsing model files.
- [pybind11](https://github.com/pybind/pybind11)(2.8): For binding python interfaces.
- [onednn](https://github.com/oneapi-src/oneDNN), [mkl](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html): BLAS libraries, for accelerating GEMM calculations.
- [openmp](https://www.openmp.org/): A standard parallel programming library.
- [openmpi](https://www.open-mpi.org/): For realizing multi-NUMA service architecture.
- [grpc](https://grpc.io/): For realizing multi-NUMA service architecture.

# 未来规划

- [ ] 首包加速：加入CPU实现的Flash-Attention等Attention加速技术；
- [ ] Context Length：扩展到32k以上；
- [ ] QAT量化支持：支持GPTQ算法量化微调过的模型；
- [ ] MoE：支持MoE模型和架构。

# License

DashInfer源代码采用Apache 2.0协议授权，您可在该仓库根目录找到协议全文。
