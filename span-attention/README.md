# SpanAttention

SpanAttention implements a high-performance decode-phase attention with paged KV cache for LLM inference on CUDA-enabled devices.

SpanAttention supports general group-query attention (GQA), which also covers the cases of multi-head attention (MHA) and multi-query attention (MQA).
For input and output, SpanAttention supports the commonly used data types including FP32 (float), FP16 (half), and BF16 (bfloat16).
For KV cache, SpanAttention supports optional 8-bit and 4-bit integer quantization.

To achieve high performance, SpanAttention supports batch execution on multiple queries.
It also leverages Tensor Cores to further accelerate the GQA computation.

## Table of contents

- [SpanAttention](#spanattention)
  - [Table of contents](#table-of-contents)
  - [Usage](#usage)
    - [Basic example](#basic-example)
    - [Asynchronous kernel launch and performance considerations](#asynchronous-kernel-launch-and-performance-considerations)
    - [KV cache quantization](#kv-cache-quantization)
    - [KV cache spans](#kv-cache-spans)
      - [Span size](#span-size)
      - [Span pointer array](#span-pointer-array)
  - [Installation and integration](#installation-and-integration)
    - [Requirements](#requirements)
    - [Build and install](#build-and-install)
    - [Integrate with CMake](#integrate-with-cmake)
  - [License](#license)


## Usage

For API documentation, please refer to the [interface header](include/spanattn/span_attn.h).

### Basic example

The following example gives a basic use case of SpanAttention for a single kernel launch.
The supported input and output data types include `span::DataType::FP32`, `span::DataType::FP16`, and `span::DataType::BF16`.
For supported KV cache quantization modes, see [KV cache quantization](#kv-cache-quantization).

```cpp
#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>
#include <spanattn/span_attn.h>

// check status code
#define SA_CHECK(expr)                                              \
  do {                                                              \
    span::SaStatus __error_code = (expr);                           \
    if (__error_code != span::SaStatus::SUCCESS) {                  \
      throw std::runtime_error(span::GetErrorString(__error_code)); \
    }                                                               \
  } while (0)

// suppose input/output data type is BF16
// suppose KV cache is not quantized
span::DataType dtype = span::DataType::BF16;
span::QuantMode qmode = span::QuantMode::NONE;

// first, create a handle
span::SpanAttnHandle_t handle{nullptr};
SA_CHECK(span::CreateHandle(&handle, dtype, qmode, ...));
if (handle == nullptr) {
  throw std::runtime_error("Failed to create SpanAttn handle");
}

// next, alloc workspace for span attention kernel
void* device_workspace{nullptr};
void* host_workspace{nullptr};
size_t device_ws_size{0};
size_t host_ws_size{0};
SA_CHECK(span::GetDeviceWorkspaceSize(&device_ws_size, handle));
SA_CHECK(span::GetHostWorkspaceSize(&host_ws_size, handle));
cudaMalloc(&device_workspace, device_ws_size);
host_workspace = std::malloc(host_ws_size);

// then, run the kernel
SA_CHECK(span::Run(...));

// finally, destroy the handle and free workspace
SA_CHECK(span::DestroyHandle(handle));
cudaFree(device_workspace);
std::free(host_workspace);
```

### Asynchronous kernel launch and performance considerations

In real-world LLM inference, SpanAttention kernel is expected to be launched asynchronously, just like other CUDA kernels.
In this case, synchronous workspace allocation is not preferred.

Thus, we recommend allocating workspace at the initialization phase, using a handle created with maximum possible values of the parameters in your use case.
Then in the execution loop, reuse the workspace and launch kernels fully asynchronously.

### KV cache quantization

SpanAttention supports quantization of KV cache.
For now, 8-bit integer (`span::QuantMode::I8`) and 4-bit unsigned integer (`span::QuantMode::U4`) quantization modes are supported.
If `span::QuantMode::NONE` is used, KV cache is not quantized and will be of the same type as input and output.

### KV cache spans

SpanAttention assumes paged KV cache, which is useful for LLM inference with long context.
A KV cache page is called a *span* in SpanAttention.
For now, SpanAttention does not provide APIs for span management.
So users are expected to manage spans manually.

On the one hand, users need to manage the spans with the span size calculated as described below.
On the other hand, users need to pass the spans as arguments to SpanAttention API.

#### Span size

Let `spanLen` be the number of tokens in a span.
For a query of length `queryLen` (i.e., the number of tokens in the cache), the number of spans for this query is `ceil(queryLen / spanLen)`.
Let `FT` be the type of input and output (can be float, half, or bfloat16).
The size of a span in bytes is computed as follows:

- Not quantized: each span takes up `spanLen * #kvHeads * headSize * sizeof(FT)` bytes.
- Quantized: let `QT` be the type of quantized KV cache (can be 8-bit or 4-bit integer), and each span takes up `spanLen * #kvHeads * headSize * sizeof_bits(QT) / 8 + 2 * spanLen * #kvHeads * sizeof(float)` bytes.

Here, `#kvHeads` is the number of KV heads in GQA, and `headSize` is the size of each head.
For now, `headSize` is always 128, which is practically the default setting for LLM models.

Spans should be allocated in **device memory**.

#### Span pointer array

Given a batch of `N` queries, for each one of K cache and V cache, the spans should be passed to APIs as a 2D array of pointers, with the first dimension being the batch index, and the second dimension being the span index inside each query.

Suppose the longest query has `maxQueryLen` tokens, then the second dimension of the array should be **at least** `ceil(maxQueryLen / spanLen)`, which is passed as `nSpansPerRequest` to `span::Create`.
Thus the expected shape of the `kSpanArray` argument passed to `span::Run` is `[N, nSpansPerRequest]`.
So is the `vSpanArray` argument.

## Installation and integration

### Requirements

- CMake 3.21 or above
- C++ compiler with C++17 support
- CUDA 12.0 or above
- Ninja is recommended
- CUTLASS 3.5 or above if using external CUTLASS

### Build and install

First configure the project:

```sh
cd span-attention
cmake -S . -B build [-G <generator>] [options]
```

Tested generators are:

- `Unix Makefiles` -- default generator;
- `Ninja` -- recommended.

Options include:

- `-DSPANATTN_CUDA_ARCHS="arch1;arch2;..."` -- target CUDA architecture compute capabilities (SM versions), default is `"75;80;90a"`;
- `-DSPANATTN_ENABLE_TEST=ON|OFF` -- build tests, default is `ON`;
- `-DSPANATTN_ENABLE_FP16=ON|OFF` -- enable FP16 (half) support, default is `ON`;
- `-DSPANATTN_ENABLE_BF16=ON|OFF` -- enable BF16 (bfloat16) support, default is `ON`;
- `-DSPANATTN_STATIC_CUDART=ON|OFF` -- statically link CUDA runtime, default is `OFF`;
- `-DSPANATTN_EXTERNAL_CUTLASS=ON|OFF` -- use external CUTLASS, default is `OFF`;
- `-DCMAKE_BUILD_TYPE=type` -- build type, default `type` is `Release`, for debugging use `Debug`;
- `-DCMAKE_INSTALL_PREFIX=directory` -- install directory.

Then build and install:

```sh
cmake --build build [--target install]
```

When using Ninja, build is running in parallel by default.
Otherwise, you can specify the number of parallel jobs with `-j` option.
For detailed output, use `-v` option.

### Integrate with CMake

Instead of installing, you can integrate SpanAttention with your project using CMake ExternalProject.
An example is provided below.
This example assumes CUTLASS is already integrated with your project as `project_cutlass`, installed in path `CUTLASS_INSTALL`.

```cmake
# suppose your project specifies CUDA archs with CMAKE_CUDA_ARCHITECTURES
set(SPANATTN_CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
set(SPANATTN_EXTERNAL_CUTLASS ON)
set(SPANATTN_ENABLE_TEST OFF)

# set your own paths
set(SPANATTN_SOURCE_DIR /path/to/source/codes/of/span-attention)
set(SPANATTN_INSTALL /path/to/install/span-attention/in/your/project)

include(ExternalProject)
ExternalProject_Add(project_spanattn
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/span-attention
  SOURCE_DIR ${SPANATTN_SOURCE_DIR}
  DEPENDS project_cutlass
  CMAKE_GENERATOR "Ninja"
  BUILD_COMMAND ${CMAKE_COMMAND} --build . -v
  CMAKE_CACHE_ARGS
    -DSPANATTN_CUDA_ARCHS:STRING=${SPANATTN_CUDA_ARCHS}
  CMAKE_ARGS
    -DSPANATTN_EXTERNAL_CUTLASS=${SPANATTN_EXTERNAL_CUTLASS}
    -DSPANATTN_ENABLE_TEST=${SPANATTN_ENABLE_TEST}
    -DCMAKE_INSTALL_PREFIX=${SPANATTN_INSTALL}
    -DCUTLASS_INSTALL_PATH=${CUTLASS_INSTALL}
)

unset(SPANATTN_CUDA_ARCHS)
unset(SPANATTN_EXTERNAL_CUTLASS)
unset(SPANATTN_ENABLE_TEST)
```

Suppose you want to use SpanAttention as a static library in your project, you can import it as `spanattn::spanattn_static`:

```cmake
file(MAKE_DIRECTORY ${SPANATTN_INSTALL}/include)
add_library(spanattn::spanattn_static STATIC IMPORTED)
add_dependencies(spanattn::spanattn_static project_spanattn)
set_target_properties(spanattn::spanattn_static
  PROPERTIES
    IMPORTED_LOCATION ${SPANATTN_INSTALL}/lib/libspanattn.a
    INTERFACE_INCLUDE_DIRECTORIES ${SPANATTN_INSTALL}/include
)
```

## License

The SpanAttention source code is licensed under the Apache 2.0 license, and you can find the full text of the license in the root of the repository.
