// SPDX-License-Identifier: Apache-2.0
#include <core/kernel/cuda/cuda_kernel.h>
#include <core/kernel/cuda/topk_r2sk/decode_graph.h>
#include <core/kernel/cuda/topk_r2sk/r2sk/bfloat16_graph.hpp>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Bf16 = hie::bfloat16;
constexpr int kBatch = 2;
constexpr int kVocab = 2053;
constexpr int kRank = 20;

void Check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void CheckHie(hiednnStatus_t result) {
  Require(result == HIEDNN_STATUS_SUCCESS, "HIE-DNN status");
}

struct Stream {
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  cudaStream_t value;
};

struct HieHandle {
  explicit HieHandle(cudaStream_t stream) {
    CheckHie(hiednnCreateCudaHandle(&value));
    CheckHie(hiednnSetCudaStream(value, stream));
  }
  ~HieHandle() { hiednnDestroyCudaHandle(value); }
  hiednnCudaHandle_t value;
};

template <class T>
struct Buffer {
  explicit Buffer(size_t count) {
    Check(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * count));
  }
  ~Buffer() { cudaFree(data); }
  T* data = nullptr;
};

template <class T>
void Upload(Buffer<T>& buffer, const T* data, size_t count,
            cudaStream_t stream) {
  Check(cudaMemcpyAsync(buffer.data, data, sizeof(T) * count,
                        cudaMemcpyHostToDevice, stream));
}

template <class T>
std::vector<T> Read(const Buffer<T>& buffer, size_t count) {
  std::vector<T> data(count);
  Check(cudaMemcpy(data.data(), buffer.data, count * sizeof(T),
                   cudaMemcpyDeviceToHost));
  return data;
}

struct Arm {
  explicit Arm(size_t bytes) : scratch(bytes) {}
  Buffer<uint16_t> scores{kBatch * kRank};
  Buffer<uint16_t> tmp{kBatch * kRank};
  Buffer<int> indices{kBatch * kRank}, counts{kBatch};
  Buffer<int64_t> tokens{kBatch};
  Buffer<curandState_t> state{kBatch};
  Buffer<void*> rng{kBatch};
  Buffer<unsigned char> scratch;
};

void Run(int device) {
  using namespace allspark::cuda;
  Stream queue;
  const auto stream = queue.value;
  HieHandle handle(stream);
  Buffer<uint16_t> logits(kBatch * kVocab);
  Buffer<float> p(kBatch), temperature(kBatch);
  size_t radik_bytes = 0, topp_bytes = 0;
  TopKRadixGetWorkspaceSize<Bf16>(&radik_bytes, kBatch, kVocab);
  TopPSoftmaxGetWorkspaceSize<Bf16>(&topp_bytes, kBatch, kRank, true);
  size_t bytes = std::max({radik_bytes, topp_bytes,
                           static_cast<size_t>(r2sk::bfloat16_graph_bytes(
                               {kBatch, kVocab, kRank}))});
  Arm eager(bytes), graph(bytes);
  Buffer<uint16_t> alternate_tmp(kBatch * kRank);
  R2skDecodeGraph selector(device);
  for (auto* a : {&eager, &graph}) {
    for (int row = 0; row < kBatch; ++row) {
      SampleKernelInitLauncher(a->state.data + row, 1234 + row, 1, stream);
    }
  }

  std::vector<uint16_t> input(kBatch * kVocab);
  const std::array<int, kBatch> initial_counts{kRank, kRank - 3};
  int graph_launches = 0;
  for (int step = 0; step < 20; ++step) {
    std::fill(input.begin(), input.end(), uint16_t{0x3f80});
    for (int j = 0; j < kRank + 7; ++j) {
      input[(step * 7 + j) % kVocab] = 0x4000 + j;
    }
    for (int j = 0; j < 24; ++j) {
      // Row one has nine higher-bin winners, fifteen distinct upper-boundary
      // values, and a large boundary bin. Its sorted selected values are
      // unique.
      input[kVocab + (step * 13 + j) % kVocab] =
          j < 9 ? 0x4000 + j : 0x3f81 + j - 9;
    }
    const bool invalid = step == 8 || step == 12;
    if (step == 8) input.back() = 0x7fc1u;
    if (step == 12) input.back() = 0xff80u;
    Upload(logits, input.data(), input.size(), stream);
    const std::array<float, kBatch> top_p{0.70f + (step % 3) * 0.1f, 0.85f};
    const std::array<float, kBatch> temp{0.65f, 0.85f + (step % 2) * 0.2f};
    Upload(p, top_p.data(), top_p.size(), stream);
    Upload(temperature, temp.data(), temp.size(), stream);
    for (auto* a : {&eager, &graph}) {
      // Pointer table stays in place across graph launches. Swap its contents
      // to model active requests changing slots in the packed decode batch.
      const int first = step >= 10 ? 1 : 0;
      const std::array<void*, kBatch> states{a->state.data + first,
                                             a->state.data + (1 - first)};
      Upload(a->rng, states.data(), states.size(), stream);
      Upload(a->counts, initial_counts.data(), initial_counts.size(), stream);
      Check(cudaMemsetAsync(a->tokens.data, 0xcc, kBatch * sizeof(int64_t),
                            stream));
      // Host arrays above may go out of scope after completion of these copies.
      Check(cudaStreamSynchronize(stream));
    }
    // Switch buffers to capture a second graph after eager work used the
    // handle.
    auto* temp_probs = step % 2 ? alternate_tmp.data : graph.tmp.data;
    Require(selector.Prepare(
                {logits.data, graph.scores.data, temp_probs, graph.indices.data,
                 graph.tokens.data, graph.rng.data, graph.counts.data, p.data,
                 temperature.data, graph.scratch.data, bytes, bytes,
                 handle.value, stream, kBatch, kVocab, kRank}),
            "valid graph shape failed preflight");
    Check(cudaStreamSynchronize(stream));
    Require(selector.LaunchPrepared() == !invalid, "preflight result mismatch");
    Check(cudaStreamSynchronize(stream));
    if (invalid) {
      Require(Read(graph.tokens, kBatch) == Read(eager.tokens, kBatch),
              "rejected graph changed its tokens");
      continue;
    }
    ++graph_launches;

    auto* scores = reinterpret_cast<Bf16*>(eager.scores.data);
    TopKRadixKernelLauncher(scores, eager.indices.data,
                            reinterpret_cast<const Bf16*>(logits.data),
                            eager.scratch.data, kBatch, kVocab, kRank, stream);
    TopPSoftmaxLauncher(
        eager.counts.data, scores, eager.indices.data,
        static_cast<const Bf16*>(scores), p.data, temperature.data,
        reinterpret_cast<Bf16*>(eager.tmp.data), eager.scratch.data, bytes,
        kBatch, kRank, true, handle.value, stream);
    SampleKernelLauncher(eager.tokens.data, eager.rng.data, scores,
                         eager.indices.data, kBatch, eager.counts.data, kRank,
                         stream, nullptr);
    Check(cudaStreamSynchronize(stream));

    const auto counts = Read(graph.counts, kBatch);
    Require(counts == Read(eager.counts, kBatch), "top-p counts differ");
    const auto ids = Read(graph.indices, kBatch * kRank);
    Require(ids == Read(eager.indices, kBatch * kRank),
            "top-k ranked IDs differ");
    const auto logits_graph = Read(graph.scores, kBatch * kRank);
    const auto logits_eager = Read(eager.scores, kBatch * kRank);
    for (int row = 0; row < kBatch; ++row) {
      for (int rank = 0; rank < counts[row]; ++rank) {
        const int j = row * kRank + rank;
        Require(logits_graph[j] == logits_eager[j],
                "top-p probabilities differ");
      }
    }
    Require(Read(graph.tokens, kBatch) == Read(eager.tokens, kBatch),
            "graph and eager sampled tokens differ");
  }
  Require(graph_launches == 18, "unexpected replay count");
  std::cout << "PASS decode sampling parity: 18 graph calls, two captured "
               "bindings, changing logits, top-p, "
               "temperatures and RNG pointers, exact probabilities and tokens, "
               "NaN and infinity fallback.\n";
}

}  // namespace

int main() {
  try {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      std::cout << "R2S-K decode graph tests require a visible CUDA device\n";
      return 77;
    }
    Check(status);
    int device = 0, major = 0;
    Check(cudaGetDevice(&device));
    Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                 device));
    if (major < 8) return 77;
    Run(device);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
