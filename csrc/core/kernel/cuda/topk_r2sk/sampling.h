// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace allspark::cuda {

// One instance per GenerateOp. Calls are serialized by that operator.
// Device storage is borrowed from the engine's top-k/top-p scratch arena.
class R2skSampling {
 public:
  static constexpr int kMaxEagerRank = 256;

  explicit R2skSampling(int device);
  ~R2skSampling();

  size_t ReserveBytes(int max_batch, int vocabulary);
  void Wait();

  // Start the first histogram without touching the outputs. Synchronize
  // stream before FinishPrepared, which leaves outputs untouched on fallback.
  bool Prepare(const void* logits, void* values, int* indices, int batch,
               int vocabulary, int k, void* arena, size_t arena_bytes,
               cudaStream_t stream);
  bool FinishPrepared();

  // True means the output is enqueued on stream. False leaves the output
  // untouched and lets the caller invoke RadiK on that same stream.
  bool Try(const void* logits, void* values, int* indices, int batch,
           int vocabulary, int k, void* arena, size_t arena_bytes,
           cudaStream_t stream);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace allspark::cuda
