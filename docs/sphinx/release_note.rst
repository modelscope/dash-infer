Version Numbering Rules:
========================

- The version numbering follows the major.minor.patch convention.
- A major upgrade primarily indicates changes in interfaces or API upgrades.
- A minor upgrade represents the introduction of significant new features.
- A patch upgrade represents bug fixes, and generally, the latest patch version can be used directly.

v2.0.0
------
Release Date: 2024/11/13

- Model Support:

   - Added prefix cache support for Qwen-VL and Qwen2-VL models.
   - Added support for 2:4 sparse models using CUDA Sparse Tensor Core.
   - Added support for the Qwen2Moe model, including the functionality for the A8W8 MoE model.
   - Multimodal models: added support for the Qwen2-VL model.

- New Features:

   - Implemented structured JSON mode output for easier model output parsing.
   - Prefix Cache: added support for CPU memory-backed prefix cache.

- Performance Optimizations:

   - Optimized SpanAttention implementation with efficient GEMM-based batch attention kernels, achieving 3~8x GQA speedup and 1.6~2x end-to-end speedup on A100, depending on batch size and context length.
   - Introduced HIE-DNN MoE Batch GEMM implementation, improving performance of MoE models.
   - Added BF16 Fused Skinny GEMM kernel support for Ampere architectures.
   - Optimized context priority scheduling and chunked prefill strategies.
   - Introduced large-batch-but-short-length Softmax kernel, improving performance for corresponding scenarios.
   - Optimized the performance of the ResultQueue for multiple concurrent operations, and added the GetTimeout interface to support blocking interfaces and return after a timeout.
   - Optimized the logits kernel, improving performance for large batch sizes. With batch sizes up to 400~500, the engine runtime overhead is less than 1%.
   - GPU Memory Usage: optimized GPU memory usage for large batch sizes.
   - FlashAttention: upgraded to FlashAttention 2.6.1 and cutlass 3.5.0.
   - Resolved crashes for MoE models with large batch sizes.
   - Prefill Cache (Prefix Cache): added Prefill Cache functionality for Qwen-VL and Qwen2-VL models.
   - Added support for A8W8 quantization for Qwen1.5 and Qwen2 models.

- Python:
  - Removed the dependency on protobuf for Python.

- Bug Fixes:

   - Optimized memory allocation strategy for warmup requests, fixing potential out-of-memory issues.
   - Fixed context loss bug in rare cases.
   - Improved model loading environment logging, adding model loading speed logs.
   - Fixed infrequent 0-token output in A8W8 and A16W8 quantized models.

- Interface Improvements:

   - Added dynamic quantization support, currently supporting A8W8, A16W8, and A16W4 quantization modes.
   - Added support for stopping requests in the pending request queue.
   - Optimized initial request processing flow and simplified logic.

- Misc
  - Upgraded the default compiler to gcc10.
  - Deprecated support for CUDA 12.1, and upgraded to CUDA 12.4 for better support on new hardware including H20.
  - Deprecated support for the bad words list in the generation parameters.
  - Enabled dynamic linking to NVIDIA-related libraries by default.
