// SPDX-License-Identifier: Apache-2.0
#include "sampling.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "r2sk/bfloat16_graph.hpp"
#include "r2sk/cuda.hpp"

namespace allspark::cuda {

struct R2skSampling::State {
  explicit State(int device_id)
      : device(device_id), trace(std::getenv("AS_TRACE_R2SK_EAGER") != nullptr) {}

  ~State() {
    if (!fork.stream && !complete && !nonfinite) return;
    int previous = device;
    const bool restore =
        cudaGetDevice(&previous) == cudaSuccess && previous != device;
    if (restore) cudaSetDevice(device);
    if (prepared.arena) cudaStreamSynchronize(prepared.stream);
    if (pending) cudaEventSynchronize(complete);
    if (complete) cudaEventDestroy(complete);
    if (fork.ready) cudaEventDestroy(fork.ready);
    if (fork.done) cudaEventDestroy(fork.done);
    if (fork.stream) cudaStreamDestroy(fork.stream);
    if (nonfinite) cudaFreeHost(nonfinite);
    if (restore) cudaSetDevice(previous);
  }

  void Initialize() {
    if (nonfinite) return;
    if (!fork.stream)
      r2sk::cuda_check(
          cudaStreamCreateWithFlags(&fork.stream, cudaStreamNonBlocking),
          "create eager R2S-K winner stream");
    if (!fork.ready)
      r2sk::cuda_check(
          cudaEventCreateWithFlags(&fork.ready, cudaEventDisableTiming),
          "create eager R2S-K ready event");
    if (!fork.done)
      r2sk::cuda_check(
          cudaEventCreateWithFlags(&fork.done, cudaEventDisableTiming),
          "create eager R2S-K join event");
    if (!complete)
      r2sk::cuda_check(
          cudaEventCreateWithFlags(&complete, cudaEventDisableTiming),
          "create eager R2S-K completion event");
    r2sk::cuda_check(
        cudaMallocHost(reinterpret_cast<void**>(&nonfinite), sizeof(*nonfinite)),
        "allocate eager R2S-K input status");
  }

  struct Selection {
    r2sk::BFloat16GraphShape shape;
    void* values;
    int* indices;
    void* arena;
    size_t bytes;
    cudaStream_t stream;
  };

  int device;
  bool trace;
  int batch = 0;
  int vocabulary = 0;
  size_t bytes = 0;
  uint32_t* nonfinite = nullptr;
  r2sk::BFloat16GraphFork fork;
  cudaEvent_t complete = nullptr;
  cudaStream_t last_stream = nullptr;
  bool pending = false;
  Selection prepared{};
};

R2skSampling::R2skSampling(int device) : state_(std::make_unique<State>(device)) {}
R2skSampling::~R2skSampling() = default;

size_t R2skSampling::ReserveBytes(int max_batch, int vocabulary) {
  auto& s = *state_;
  if (s.batch != max_batch || s.vocabulary != vocabulary) {
    Wait();
    s.bytes = max_batch < 1 || max_batch > 65535 || vocabulary <= 2
                  ? 0
                  : r2sk::bfloat16_graph_bytes({
                        static_cast<uint32_t>(max_batch),
                        static_cast<uint32_t>(vocabulary),
                        static_cast<uint32_t>(
                            std::min(kMaxEagerRank, vocabulary - 1))});
    s.batch = max_batch;
    s.vocabulary = vocabulary;
  }
  return s.bytes;
}

void R2skSampling::Wait() {
  auto& s = *state_;
  if (s.prepared.arena) {
    r2sk::cuda_check(cudaStreamSynchronize(s.prepared.stream),
                     "wait for eager R2S-K preparation");
    s.prepared = {};
  }
  if (s.pending) {
    r2sk::cuda_check(cudaEventSynchronize(s.complete), "wait for R2S-K scratch");
    s.pending = false;
  }
}

bool R2skSampling::Prepare(const void* logits, void* values, int* indices,
                           int batch, int vocabulary, int k, void* arena,
                           size_t arena_bytes, cudaStream_t stream) {
  auto& s = *state_;
  if (!s.bytes || batch < 1 || batch > s.batch || vocabulary != s.vocabulary ||
      k < 2 || k > kMaxEagerRank || k >= vocabulary || !logits || !values ||
      !indices || !arena || reinterpret_cast<uintptr_t>(arena) % 256 ||
      arena_bytes < s.bytes) {
    return false;
  }
  cudaStreamCaptureStatus capture;
  r2sk::cuda_check(cudaStreamIsCapturing(stream, &capture),
                   "query R2S-K stream capture status");
  if (capture != cudaStreamCaptureStatusNone) return false;

  if (s.prepared.arena || (s.pending && s.last_stream != stream)) Wait();
  s.Initialize();
  const r2sk::BFloat16GraphShape shape{
      static_cast<uint32_t>(batch), static_cast<uint32_t>(vocabulary),
      static_cast<uint32_t>(k)};
  // Retain the preparation binding until the stream has completed its flag copy.
  s.prepared = {shape, values, indices, arena, arena_bytes, stream};
  r2sk::begin_bfloat16_selection(shape, logits, arena, arena_bytes, stream,
                                 s.nonfinite);
  return true;
}

bool R2skSampling::FinishPrepared() {
  auto& s = *state_;
  const auto a = s.prepared;
  s.prepared = {};
  if (!a.arena || *s.nonfinite) return false;

  static_assert(sizeof(int) == sizeof(uint32_t));
  try {
    r2sk::finish_bfloat16_selection(a.shape, a.values,
                                    reinterpret_cast<uint32_t*>(a.indices),
                                    a.arena, a.bytes, a.stream, s.fork);
    r2sk::cuda_check(cudaEventRecord(s.complete, a.stream),
                     "record eager R2S-K completion");
    s.last_stream = a.stream;
    s.pending = true;
  } catch (...) {
    // A launch failure may leave work on either stream.
    cudaDeviceSynchronize();
    throw;
  }
  if (s.trace) {
    LOG(INFO) << "R2SK_DEVICE_EAGER batch=" << a.shape.batch
              << " k=" << a.shape.k;
  }
  return true;
}

bool R2skSampling::Try(const void* logits, void* values, int* indices, int batch,
                       int vocabulary, int k, void* arena, size_t arena_bytes,
                       cudaStream_t stream) {
  if (!Prepare(logits, values, indices, batch, vocabulary, k, arena, arena_bytes,
               stream)) {
    return false;
  }
  r2sk::cuda_check(cudaStreamSynchronize(stream), "read eager R2S-K input status");
  return FinishPrepared();
}

}  // namespace allspark::cuda
