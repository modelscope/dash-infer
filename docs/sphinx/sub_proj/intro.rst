Introduction to Subprojects
---------------------------

DashInfer implements a variety of highly optimized CUDA kernels, among which some are provided as self-contained subprojects.
The two subprojects are:

- **HIE-DNN**: HIE-DNN is an operator library for high-performance inference of deep neural network (DNN), mostly complying with ONNX open format.
- **SpanAttention**: SpanAttention is a high-performance decode-phase attention implementation with paged KV cache for LLM inference on CUDA-enabled devices.

For detailed information about the subprojects, please refer to the following links:

- :doc:`hiednn`
- :doc:`spanattn`
