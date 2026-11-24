// SPDX-License-Identifier: Apache-2.0
#include "sampling.h"

#include "r2sk/cuda.hpp"

#include <cstdint>
#include <vector>

namespace allspark::cuda {
namespace {
// Keep the shared arena reservation used by the decode graph.
constexpr int kArenaRankCapacity = 256;
constexpr int kSortLimit = 1024;
constexpr size_t kArenaAlignment = 256;
}  // namespace

struct R2skSampling::State {
  explicit State(int device) : device(device) {}

  ~State() {
    if (!workspace && !completion) return;
    int previous = device;
    const bool restore =
        cudaGetDevice(&previous) == cudaSuccess && previous != device;
    if (restore) cudaSetDevice(device);
    if (pending) cudaEventSynchronize(completion);
    workspace.reset();
    if (completion) cudaEventDestroy(completion);
    if (restore) cudaSetDevice(previous);
  }

  int device;
  int batch = 0;
  int vocabulary = 0;
  int rank_capacity = 0;
  size_t bytes = 0;
  void* base = nullptr;
  cudaEvent_t completion = nullptr;
  bool pending = false;
  std::vector<uint32_t> lengths;
  std::unique_ptr<r2sk::BFloat16BatchWorkspace> workspace;
};

R2skSampling::R2skSampling(int device) : state_(new State(device)) {}
R2skSampling::~R2skSampling() = default;

size_t R2skSampling::ReserveBytes(int max_batch, int vocabulary) {
  auto& state = *state_;
  if (max_batch < 1 || max_batch > 65535 || vocabulary <= 2) {
    state.rank_capacity = 0;
    return 0;
  }
  if (state.batch == max_batch && state.vocabulary == vocabulary) {
    state.rank_capacity = std::min(kArenaRankCapacity, vocabulary - 1);
    return state.bytes;
  }
  const int rank = std::min(kArenaRankCapacity, vocabulary - 1);
  const auto bytes = r2sk::BFloat16BatchWorkspace::required_device_bytes(
      max_batch, vocabulary, rank, kSortLimit, true, true);
  Wait();
  state.workspace.reset();
  state.base = nullptr;
  state.lengths.assign(max_batch, vocabulary);
  state.batch = max_batch;
  state.vocabulary = vocabulary;
  state.rank_capacity = rank;
  state.bytes = bytes;
  return bytes;
}

void R2skSampling::Wait() {
  auto& state = *state_;
  if (state.pending) {
    r2sk::cuda_check(cudaEventSynchronize(state.completion),
                     "wait for R2S-K scratch");
    state.pending = false;
  }
}

bool R2skSampling::Try(const void* logits, void* values, int* indices,
                       int batch, int vocabulary, int k, void* arena,
                       size_t arena_bytes, cudaStream_t stream) {
  auto& state = *state_;
  // Use the small-rank regime that improved across the GB300 test cases.
  if (batch < 1 || batch > state.batch || vocabulary != state.vocabulary ||
      k <= 1 || k > kMaxEagerRank || k > state.rank_capacity || !arena ||
      reinterpret_cast<uintptr_t>(arena) % kArenaAlignment != 0 ||
      arena_bytes < state.bytes) {
    return false;
  }
  cudaStreamCaptureStatus capture;
  r2sk::cuda_check(cudaStreamIsCapturing(stream, &capture),
                   "query R2S-K stream capture status");
  if (capture != cudaStreamCaptureStatusNone) return false;

  // Pinned descriptors can only be reused after their asynchronous uploads
  // complete. The prior completion event includes the winner-stream join.
  Wait();
  if (!state.completion) {
    r2sk::cuda_check(
        cudaEventCreateWithFlags(&state.completion, cudaEventDisableTiming),
        "create R2S-K completion event");
  }
  if (!state.workspace) {
    state.workspace = std::make_unique<r2sk::BFloat16BatchWorkspace>(
        state.batch, vocabulary, state.rank_capacity, kSortLimit, true, true,
        arena, arena_bytes);
  } else if (state.base != arena) {
    state.workspace->bind_device_storage(arena, arena_bytes);
  }
  state.base = arena;

  r2sk::Options options;
  options.primitives = r2sk::PrimitiveMode::radik;
  options.winner_sort = r2sk::WinnerSort::overlap;
  options.entry_sort = false;
  options.reject_nonfinite = true;
  static_assert(sizeof(int) == sizeof(uint32_t));
  try {
    r2sk::topk_batch_bf16(static_cast<const r2sk::BFloat16*>(logits),
                          state.lengths.data(), batch, k,
                          static_cast<r2sk::BFloat16*>(values),
                          reinterpret_cast<uint32_t*>(indices),
                          *state.workspace, options, stream);
    r2sk::cuda_check(cudaEventRecord(state.completion, stream),
                     "record R2S-K completion");
    state.pending = true;
  } catch (const r2sk::NonFiniteInput&) {
    // The entry histogram is synchronized before reporting this condition.
    // No gather, output, or winner-stream work has been queued.
    return false;
  } catch (...) {
    // A launch failure may leave queued work on either stream. Drain it
    // before allowing teardown or arena reuse on the error path.
    cudaDeviceSynchronize();
    throw;
  }
  return true;
}

}  // namespace allspark::cuda
