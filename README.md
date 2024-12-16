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

# Examples

In `<path_to_dashinfer>/examples` there are examples for C++ and Python interfaces, and please refer to the documentation in `<path_to_dashinfer>/documents/EN` to run the examples.

- [Basic Python Example](examples/python/0_basic/basic_example_qwen_v10_io.ipynb) [![Open In PAI-DSW](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/Open-in-DSW20px.svg)](https://gallery.pai-ml.com/#/import/https://github.com/modelscope/dash-infer/blob/main/examples/python/0_basic/basic_example_qwen_v10_io.ipynb)
- [Documentation for All Python Examples](docs/EN/examples_python.md)
- [Documentation for C++ Examples](docs/EN/examples_cpp.md)

## Multi-Modal Model(VLMs) Support

The VLM Support in [multimodal](multimodal/) folder, it's a toolkit to support Vision Language Models (VLMs) inference based on the DashInfer engine. It's compatible with the OpenAI Chat Completion API, supporting text and image/video inputs.

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
