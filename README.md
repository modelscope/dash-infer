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


## News
- [2024/12] 🔥 DashInfer: Announcing the release of v2.0, now with enhanced GPU (CUDA) support! This version includes features like prefix caching (with GPU & CPU swapping), guided decoding, optimized attention for GQA, a lockless reactor engine, and newly added support for the VLM model (Qwen-VL) and MoE Models. For more details, please refer to the [release notes](https://dashinfer.readthedocs.io/en/latest/index.html#v2-0-0).

- [2024/06] DashInfer:  v1.0 release with x86 & ARMv9 CPU and CPU flash attention support.

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
- **CUDA GPUs**: FP16, BF16, FP8, FP32, Int8(InstantQuant), Int4(InstantQuant)
- **x86 CPU**: FP32, BF16
- **ARM Yitian710 CPU**: FP32, BF16, Int8(InstantQuant)

### Quantization
DashInfer provides various many quantization technology for LLM weight, such as, int{8,4} weight only quantization, int8 activate quantization, and many customized fused kernel to provide best performance on specified device.

To put it simply, models fine-tuned with GPTQ will provide better accuracy, but our InstantQuant (IQ) technique,
which does not require fine-tuning, can offer a faster deployment experience.
Detailed explanations of IQ quantization can be found at the end of this article.

In terms of supported quantization algorithms, DashInfer supports models fine-tuned with GPTQ and dynamic quantization
using the IQ quantization technique in two ways:

- **IntantQuant(IQ)**: DashInfer provides the InstantQuant (IQ) dynamic quantization technique, which does not require fine-tuning and can offer a faster deployment experience. Detailed explanations of IQ quantization can be found at the end of this article.
- **GPTQ**: Models fine-tuned with GPTQ will provide better accuracy, but it requires a fine-tuning step.

The quantization strategies introduced here can be broadly divided into two categories:

- **Weight Only Quantization**: This quantization technique only quantizes and compresses the weights,
  such as storing weights in int8 format, but uses bf16/fp16 for computations. It only reduces memory access requirements, 
  without improving computational performance compared to BF16.
- **Activation Quantization**: This quantization technique not only stores weights in int8 format but also performs low-precision quantized computations (such as int8) during the calculation phase. (Since Nvidia GPUs only have int8 Tensor Cores that can easily maintain precision, this quantization technique can reduce memory access requirements and improve computational performance, making it a more ideal quantization approach. In terms of accuracy, it may have a slight decrease compared to Weight Only quantization, so business data accuracy testing is required.


In terms of quantization granularity, there are two types:

- **Per-Channel**: DashInfer's quantization techniques at least adopt the Per-Channel (also known as Per-Token) quantization granularity, and some also provide Sub-Channel quantization granularity. Generally speaking, Per-Channel quantization can meet most accuracy requirements due to its simple implementation and optimal performance. Only when the accuracy of Per-Channel quantization is insufficient should the Sub-Channel quantization strategy be considered.
- **Sub-Channel**: Compared to Per-Channel quantization, Sub-Channel refers to dividing a channel into N groups, and calculating quantization parameters within each group. This quantization granularity typically provides better accuracy, but due to increased implementation complexity, it comes with many limitations. For example, performance may be slightly slower than Per-Channel quantization, and Activation quantization is difficult to implement Sub-Channel quantization due to computational formula constraints (DashInfer's Activation quantization is all Per-Channel).

# Documentation and Example Code

## Documentation

For the detailed user manual, please refer to the documentation: [Documentation Link](https://dashinfer.readthedocs.io/en/latest/).

### Quick Start:

1. Using API [Python Quick Start](https://dashinfer.readthedocs.io/en/latest/get_started/quick_start_api_py_en.html)
2. LLM OpenAI Server [Quick Start Guide for OpenAI API Server](https://dashinfer.readthedocs.io/en/latest/get_started/quick_start_api_server_en.html)
3. VLM OpenAI Server [VLM Support](https://dashinfer.readthedocs.io/en/latest/vlm/vlm_offline_inference_en.html)

### Feature Introduction:

1. [Prefix Cache](https://dashinfer.readthedocs.io/en/latest/llm/prefix_caching.html)
2. [Guided Decoding](https://dashinfer.readthedocs.io/en/latest/llm/guided_decoding.html)
3. [Engine Config](https://dashinfer.readthedocs.io/en/latest/llm/runtime_config.html)

### Development:

1. [Development Guide](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#)
2. [Build From Source](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#build-from-source-code)
3. [OP Profiling](https://dashinfer.readthedocs.io/en/latest/devel/source_code_build_en.html#profiling)
4. [Environment Variable](https://dashinfer.readthedocs.io/en/latest/get_started/env_var_options_en.html)

## Code Examples

In `<path_to_dashinfer>/examples` there are examples for C++ and Python interfaces, and please refer to the documentation in `<path_to_dashinfer>/documents/EN` to run the examples.



- [Base GPU Python Example](examples/python/0_basic/cuda/demo_dashinfer_2_0_gpu_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modelscope/dash-infer/blob/main/examples/python/0_basic/cuda/demo_dashinfer_2_0_gpu_example.ipynb)
- [Documentation for All Python Examples](docs/EN/examples_python.md)
- [Documentation for C++ Examples](docs/EN/examples_cpp.md)

## Multi-Modal Model(VLMs) Support

The VLM Support in [multimodal](multimodal/) folder, it's a toolkit to support Vision Language Models (VLMs) inference based on the DashInfer engine. It's compatible with the OpenAI Chat Completion API, supporting text and image/video inputs.

## Performance

We have conducted several benchmarks to compare the performance of mainstream LLM inference engines.

### Multi-Modal Model (VLMs)

We compared the performance of Qwen-VL with vllm across various model sizes:

![img_1.png](docs/resources/image/dashinfer-benchmark-vl.png)

Benchmarks were conducted using an A100-80Gx1 for 2B and 7B sizes, and an A100-80Gx4 for the 72B model. For more details, please refer to the [benchmark documentation](https://github.com/modelscope/dash-infer/blob/main/multimodal/tests/README.md).

### Prefix Cache

We evaluated the performance of the prefix cache at different cache hit rates:

![dahsinfer-benchmark-prefix-cache.png](docs/resources/image/dahsinfer-benchmark-prefix-cache.png)

The chart above shows the reduction in TTFT (Time to First Token) with varying PrefixCache hit rates in DashInfer.

![dashinfer-prefix-effect.png](docs/resources/image/dashinfer-prefix-effect.png)

**Test Setup:**  
- **Model:** Qwen2-72B-Instruct  
- **GPU:** 4x A100  
- **Runs:** 20  
- **Batch Size:** 1  
- **Input Tokens:** 4000  
- **Output Tokens:** 1  

### Guided Decoding (JSON Mode)

We compared the guided output (in JSON format) between different engines using the same request with a customized JSON schema (Context Length: 45, Generated Length: 63):

![dashinfer-benchmark-json-mode.png](docs/resources/image/dashinfer-benchmark-json-mode.png)

# Subprojects

1. [HIE-DNN](https://github.com/modelscope/dash-infer/tree/main/HIE-DNN): an operator library for high-performance inference of deep neural network (DNN).
2. [SpanAttention](https://github.com/modelscope/dash-infer/tree/main/span-attention): a high-performance decode-phase attention with paged KV cache for LLM inference on CUDA-enabled devices.

# Citation

The high-performance implementation of DashInfer MoE operator is introduced in [this paper](https://arxiv.org/abs/2501.16103), and DashInfer employs the efficient top-k operator [*RadiK*](https://arxiv.org/abs/2501.14336).
If you find them useful, please feel free to cite these papers:

```bibtex
@misc{dashinfermoe2025,
  title = {Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference}, 
  author = {Yinghan Li and Yifei Li and Jiejing Zhang and Bujiao Chen and Xiaotong Chen and Lian Duan and Yejun Jin and Zheng Li and Xuanyu Liu and Haoyu Wang and Wente Wang and Yajie Wang and Jiacheng Yang and Peiyang Zhang and Laiwen Zheng and Wenyuan Yu},
  year = {2025},
  eprint = {2501.16103},
  archivePrefix = {arXiv},
  primaryClass = {cs.DC},
  url = {https://arxiv.org/abs/2501.16103}
}

@inproceedings{radik2024,
  title = {RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection},
  author = {Li, Yifei and Zhou, Bole and Zhang, Jiejing and Wei, Xuechao and Li, Yinghan and Chen, Yingda},
  booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
  year = {2024}
}
```

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
- [x] Quantization: Fp8 A8W8 Activation quantization support on CUDA.
- [x] LORA: Continues Batch LORA Optimization.
- [x] Parallel Context phase and Generation phase within engine.
- [x] More effective MoE Operator on GPU.
- [ ] Porting to AMD(ROCm) Platform.

# License

The DashInfer source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
