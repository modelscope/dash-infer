// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace allspark::cuda {

// Post-logits preparation and pre-token-history device buffers for one batch.
struct DecodeGraphArgs {
  const void* logits;
  void* scores;
  void* temp_probs;
  int* indices;
  int64_t* tokens;
  void** rng_states;
  int* counts;
  const float* top_p;
  const float* temperature;
  void* scratch;
  size_t scratch_bytes;
  size_t workspace_bytes;
  hiednnCudaHandle_t hiednn;
  cudaStream_t stream;
  int batch;
  int vocabulary;
  int k;
};

class R2skDecodeGraph {
 public:
  explicit R2skDecodeGraph(int device);
  ~R2skDecodeGraph();

  // Enqueue a finite-input check before the caller's existing synchronization.
  // Returns false for unsupported shapes, leaving the caller's buffers intact.
  bool Prepare(const DecodeGraphArgs& args);

  // Call only after synchronization of Prepare. Nonfinite input falls back
  // before any sampling outputs or RNG states are touched.
  bool LaunchPrepared();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace allspark::cuda
