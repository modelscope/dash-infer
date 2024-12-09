======================================
Welcome to DashInfer documentation
======================================

DashInfer is a highly optimized LLM inference engine with the following core features:

- **Lightweight Architecture**: DashInfer requires minimal third-party dependencies and uses static linking for almost all dependency libraries. By providing C++ and Python interfaces, DashInfer can be easily integrated into your existing system.

- **High Precision**: DashInfer has been rigorously tested to ensure accuracy, and is able to provide inference whose accuracy is consistent with PyTorch and other GPU engines (e.g., vLLM).

- **High Performance**: DashInfer employs optimized kernels to provide high-performance LLM serving, as well as lots of standard LLM inference techniques, including:

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

=============
Release Note
=============

.. include:: release_note.rst

==================
Table of Contents
==================

.. _get_started:
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   get_started/install_en.md

   get_started/quick_start_api_py_en.md

   get_started/quick_start_api_server_en.md

   get_started/env_var_options_en

.. _supported_models:
.. toctree::
   :maxdepth: 1
   :caption: Models

   supported_models_en

.. _llm_deployment:
.. toctree::
   :maxdepth: 1
   :caption: LLM Deployment

   llm/llm_offline_inference_en

   llm/runtime_config

   llm/guided_decoding

   llm/prefix_caching

.. _vlm_deployment:
.. toctree::
   :maxdepth: 1
   :caption: MultiModal LM (MMLM) Deployment

   vlm/vlm_offline_inference_en

.. _developer_guide:
.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   devel/source_code_build_en.rst

.. _quant_support:
.. toctree::
   :maxdepth: 2
   :caption: Quantization

   quant/weight_activate_quant
   quant/kv_cache_quant

.. _sub_proj:
.. toctree::
   :maxdepth: 2
   :caption: Subprojects

   sub_proj/intro
   sub_proj/hiednn.md
   sub_proj/spanattn.md

.. The following sections are not ready yet

.. Benchmark
.. ===========

..    profile/profile_latency_throughput
..    profile/profile_op
..    eval/evaluation_llm


.. Advanced Guide
.. ==============

..    adv/content_length_extention
..    adv/json_mode_output


.. API Reference
.. =============

..    api/py_api_ref

.. _faq:
.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq_en
