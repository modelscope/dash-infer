// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include "core.hpp"
#include "batch_core.hpp"
#include "bfloat16.hpp"
#include <memory>

namespace r2sk {
void cuda_check(cudaError_t error, const char* operation);
// One workspace per concurrent call/stream. Reserve outside timed calls.
// N <= INT_MAX; finite FP32 input and disjoint output/workspace required.
// Indices are uint32 original offsets, not tie-break priorities.
class Workspace {
 public:
  // enable_winner_runs reserves a separate k-element sort scratch buffer,
  // nonblocking stream and dependency events for serial/overlap experiments.
  // enable_capture adds a 4096-counter histogram; captured values reuse the
  // existing candidate ping-pong buffers. Defaults preserve the original API.
  Workspace(uint32_t max_n, uint32_t max_k, uint32_t max_sort_limit,
            bool indexed, bool enable_winner_runs = false,
            bool enable_capture = false);
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  uint64_t device_bytes() const;
  uint64_t pinned_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend Stats topk(const float*, uint32_t, uint32_t, float*, uint32_t*,
                    Workspace&, Options, cudaStream_t);
};
// Host-controlled call: synchronizes for decisions, enqueues final sorting and
// output copy on stream. Caller must synchronize before consuming output or
// releasing/reusing buffers. No cudaMalloc/cudaFree during topk().
Stats topk(const float* device_input, uint32_t n, uint32_t k,
           float* device_output, uint32_t* device_indices, Workspace& workspace,
           Options options = {}, cudaStream_t stream = nullptr);

class BatchWorkspace {
 public:
  BatchWorkspace(uint32_t max_batch, uint32_t max_n, uint32_t max_k,
                 uint32_t max_sort_limit, bool indexed,
                 bool enable_winner_runs = false);
  ~BatchWorkspace();
  BatchWorkspace(const BatchWorkspace&) = delete;
  BatchWorkspace& operator=(const BatchWorkspace&) = delete;
  uint64_t device_bytes() const;
  uint64_t pinned_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend BatchStats topk_batch(const float*, const uint32_t*, uint32_t,
                               uint32_t, float*, uint32_t*, BatchWorkspace&,
                               Options, cudaStream_t);
};
// Concatenated input rows, HOST lengths[batch], common k; output is [batch,k].
// Indices are relative to each original row. Arbitrary row lengths/offsets are
// supported, including unaligned packed loads. Shifting samples input[0] of
// EACH row. enable_winner_runs reserves a separate per-row sort scratch and
// one shared nonblocking winner stream for serial/overlap controls.
// One workspace per in-flight call; synchronize
// stream before reuse/release. No device allocations inside topk_batch().
BatchStats topk_batch(const float* device_input, const uint32_t* host_lengths,
                      uint32_t batch, uint32_t k, float* device_output,
                      uint32_t* device_indices, BatchWorkspace& workspace,
                      Options options = {}, cudaStream_t stream = nullptr);

// Same batch controller, radix digits and primitives, with raw BF16 payloads
// in every global value buffer. Keys/comparisons expand only in registers or
// shared sort staging. No full-array conversion or final BF16 gather is used.
// IEEE BF16 device storage must be contiguous, finite, and aligned to 2 bytes.
// As with BatchWorkspace: one workspace per in-flight call; disjoint output;
// synchronize the caller stream before reuse/destruction. No per-call
// cudaMalloc.
class BFloat16BatchWorkspace {
 public:
  BFloat16BatchWorkspace(uint32_t max_batch, uint32_t max_n, uint32_t max_k,
                         uint32_t max_sort_limit, bool indexed,
                         bool enable_winner_runs = false);
  // Borrow device storage owned by the engine. It must be 256-byte aligned,
  // disjoint from inputs/outputs and remain live through caller-stream
  // completion. This object owns only pinned host metadata/events/streams.
  BFloat16BatchWorkspace(uint32_t max_batch, uint32_t max_n, uint32_t max_k,
                         uint32_t max_sort_limit, bool indexed,
                         bool enable_winner_runs, void* device_storage,
                         uint64_t storage_bytes);
  static uint64_t required_device_bytes(uint32_t max_batch, uint32_t max_n,
                                        uint32_t max_k, uint32_t max_sort_limit,
                                        bool indexed,
                                        bool enable_winner_runs = false);
  // Only after the previous call's caller stream has completed. This never
  // frees the old storage and cannot convert an owning workspace to borrowed.
  void bind_device_storage(void* device_storage, uint64_t storage_bytes);
  uint64_t owned_device_bytes() const;
  ~BFloat16BatchWorkspace();
  BFloat16BatchWorkspace(const BFloat16BatchWorkspace&) = delete;
  BFloat16BatchWorkspace& operator=(const BFloat16BatchWorkspace&) = delete;
  uint64_t device_bytes() const;
  uint64_t pinned_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend BatchStats topk_batch_bf16(const BFloat16*, const uint32_t*, uint32_t,
                                    uint32_t, BFloat16*, uint32_t*,
                                    BFloat16BatchWorkspace&, Options,
                                    cudaStream_t);
};
BatchStats topk_batch_bf16(const BFloat16* device_input,
                           const uint32_t* host_lengths, uint32_t batch,
                           uint32_t k, BFloat16* device_output,
                           uint32_t* device_indices,
                           BFloat16BatchWorkspace& workspace,
                           Options options = {}, cudaStream_t stream = nullptr);
}  // namespace r2sk
