// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace r2sk {

struct BFloat16GraphShape {
  uint32_t batch;
  uint32_t vocabulary;
  uint32_t k;
  uint32_t sort_limit = 1024;
  uint32_t block_cap = 1280;
};

enum class GraphRoute : uint32_t { exact, early_sort, refine, invalid_nan };

struct BFloat16GraphRow {
  GraphRoute route;
  uint32_t first_winners;
  uint32_t first_boundary;
  uint32_t passes;
};

// Resources are created before capture and kept alive for graph replay.
// Leaving these handles null selects the same schedule on one stream.
struct BFloat16GraphFork {
  cudaStream_t stream = nullptr;
  cudaEvent_t ready = nullptr;
  cudaEvent_t done = nullptr;
};

uint64_t bfloat16_graph_bytes(BFloat16GraphShape shape);

// Start a selection and optionally copy the batch-wide nonfinite flag into
// pinned host memory. With that option, synchronize the stream and decline
// on a nonzero flag before finishing. The same arena, shape and stream must
// be used for both calls.
void begin_bfloat16_selection(BFloat16GraphShape shape, const void* input,
                              void* arena, uint64_t arena_bytes,
                              cudaStream_t stream,
                              uint32_t* pinned_nonfinite = nullptr);
void finish_bfloat16_selection(BFloat16GraphShape shape, void* values,
                               uint32_t* indices, void* arena,
                               uint64_t arena_bytes, cudaStream_t stream,
                               BFloat16GraphFork fork = {},
                               BFloat16GraphRow* rows = nullptr);

// Dense BF16 [batch,vocabulary] in, descending BF16 [batch,k] and
// row-relative indices out. Signed zeros compare as equal numerically.
// Infinities are ordered numerically. NaN rows leave both outputs untouched
// and report invalid_nan. Callers omitting rows must ensure no NaNs.
// Buffers must be disjoint and survive until the graph or stream is done.
// The arena is borrowed, 256-byte aligned, and exclusive to an in-flight call.
void enqueue_bfloat16_graph(BFloat16GraphShape shape, const void* input,
                            void* values, uint32_t* indices, void* arena,
                            uint64_t arena_bytes, cudaStream_t stream,
                            BFloat16GraphFork fork = {},
                            BFloat16GraphRow* rows = nullptr);

}  // namespace r2sk
