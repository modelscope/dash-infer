// SPDX-License-Identifier: Apache-2.0
#include "decode_graph.h"

#include "core/kernel/cuda/cuda_kernel.h"
#include "r2sk/bfloat16_graph.hpp"
#include "r2sk/cuda.hpp"

#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace allspark::cuda {
namespace {

constexpr uint32_t kThreads = 256;
constexpr int kMaxGraphs = 24;

auto Arguments(const DecodeGraphArgs& a) {
  return std::tie(a.logits, a.scores, a.temp_probs, a.indices, a.tokens,
                  a.rng_states, a.counts, a.top_p, a.temperature, a.scratch,
                  a.scratch_bytes, a.workspace_bytes, a.hiednn, a.stream,
                  a.batch, a.vocabulary, a.k);
}

r2sk::BFloat16GraphShape Shape(const DecodeGraphArgs& a) {
  return {static_cast<uint32_t>(a.batch), static_cast<uint32_t>(a.vocabulary),
          static_cast<uint32_t>(a.k)};
}

__global__ void FindNonfinite(const uint16_t* logits, uint64_t count,
                              uint32_t* nonfinite) {
  int bad = 0;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    bad |= (logits[i] & 0x7f80u) == 0x7f80u;
  }
  if (__syncthreads_or(bad) && threadIdx.x == 0) atomicOr(nonfinite, 1u);
}

void Enqueue(const DecodeGraphArgs& a, r2sk::BFloat16GraphFork fork) {
  using Bf16 = hie::bfloat16;
  static_assert(sizeof(Bf16) == 2 && sizeof(int) == sizeof(uint32_t));
  r2sk::enqueue_bfloat16_graph(Shape(a), a.logits, a.scores,
                               reinterpret_cast<uint32_t*>(a.indices),
                               a.scratch, a.scratch_bytes, a.stream, fork);
  auto* scores = static_cast<Bf16*>(a.scores);
  TopPSoftmaxLauncher(
      a.counts, scores, a.indices, static_cast<const Bf16*>(scores), a.top_p,
      a.temperature, static_cast<Bf16*>(a.temp_probs), a.scratch,
      a.workspace_bytes, a.batch, a.k, true, a.hiednn, a.stream);
  SampleKernelLauncher(a.tokens, a.rng_states, scores, a.indices, a.batch,
                       a.counts, a.k, a.stream, nullptr);
  r2sk::cuda_check(cudaGetLastError(), "enqueue graph sampler");
}

cudaGraphExec_t Capture(const DecodeGraphArgs& a,
                        r2sk::BFloat16GraphFork fork) {
  cudaGraph_t graph = nullptr;
  r2sk::cuda_check(
      cudaStreamBeginCapture(a.stream, cudaStreamCaptureModeThreadLocal),
      "begin decode sampling capture");
  try {
    Enqueue(a, fork);
  } catch (...) {
    cudaStreamEndCapture(a.stream, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  r2sk::cuda_check(cudaStreamEndCapture(a.stream, &graph),
                   "end decode sampling capture");
  cudaGraphExec_t executable = nullptr;
  const auto result = cudaGraphInstantiate(&executable, graph, 0);
  cudaGraphDestroy(graph);
  r2sk::cuda_check(result, "instantiate decode sampling graph");
  return executable;
}

}  // namespace

struct R2skDecodeGraph::State {
  explicit State(int device_id)
      : device(device_id),
        trace(std::getenv("AS_TRACE_R2SK_DECODE_GRAPH") != nullptr) {}

  ~State() {
    if (!fork.stream && !flag && !done) return;
    int previous = device;
    const bool restore =
        cudaGetDevice(&previous) == cudaSuccess && previous != device;
    if (restore) cudaSetDevice(device);
    if (prepared.logits) cudaStreamSynchronize(prepared.stream);
    if (pending) cudaEventSynchronize(done);
    for (auto& entry : graphs) cudaGraphExecDestroy(entry.exec);
    if (done) cudaEventDestroy(done);
    if (fork.ready) cudaEventDestroy(fork.ready);
    if (fork.done) cudaEventDestroy(fork.done);
    if (fork.stream) cudaStreamDestroy(fork.stream);
    if (flag) cudaFreeHost(flag);
    if (restore) cudaSetDevice(previous);
  }

  void Initialize() {
    if (flag) return;
    r2sk::cuda_check(
        cudaStreamCreateWithFlags(&fork.stream, cudaStreamNonBlocking),
        "create decode graph winner stream");
    r2sk::cuda_check(
        cudaEventCreateWithFlags(&fork.ready, cudaEventDisableTiming),
        "create decode graph ready event");
    r2sk::cuda_check(
        cudaEventCreateWithFlags(&fork.done, cudaEventDisableTiming),
        "create decode graph join event");
    r2sk::cuda_check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
                     "create decode graph completion event");
    r2sk::cuda_check(
        cudaMallocHost(reinterpret_cast<void**>(&flag), sizeof(*flag)),
        "allocate decode graph input status");
    graphs.reserve(kMaxGraphs);
  }

  struct Entry {
    DecodeGraphArgs args;
    cudaGraphExec_t exec;
  };

  int device;
  bool trace;
  uint32_t* flag = nullptr;
  cudaEvent_t done = nullptr;
  r2sk::BFloat16GraphFork fork;
  std::vector<Entry> graphs;
  DecodeGraphArgs prepared{};
  bool pending = false;
};

R2skDecodeGraph::R2skDecodeGraph(int device)
    : state_(std::make_unique<State>(device)) {}
R2skDecodeGraph::~R2skDecodeGraph() = default;

bool R2skDecodeGraph::Prepare(const DecodeGraphArgs& a) {
  auto& s = *state_;
  s.prepared = {};
  if (a.batch < 1 || a.batch > 65535 || a.k < 2 || a.k > 256 ||
      a.vocabulary <= a.k || !a.scratch ||
      reinterpret_cast<uintptr_t>(a.scratch) % 256 ||
      a.scratch_bytes < r2sk::bfloat16_graph_bytes(Shape(a))) {
    return false;
  }
  cudaStreamCaptureStatus status;
  r2sk::cuda_check(cudaStreamIsCapturing(a.stream, &status),
                   "query decode sampling stream");
  if (status != cudaStreamCaptureStatusNone) return false;
  s.Initialize();
  auto* flag = static_cast<uint32_t*>(a.scratch);
  const uint64_t count = uint64_t(a.batch) * a.vocabulary;
  const uint32_t blocks =
      std::min<uint64_t>((count + kThreads - 1) / kThreads, 256);
  r2sk::cuda_check(cudaMemsetAsync(flag, 0, sizeof(*flag), a.stream),
                   "reset decode graph input status");
  FindNonfinite<<<blocks, kThreads, 0, a.stream>>>(
      static_cast<const uint16_t*>(a.logits), count, flag);
  r2sk::cuda_check(cudaGetLastError(), "check decode graph logits");
  r2sk::cuda_check(cudaMemcpyAsync(s.flag, flag, sizeof(*flag),
                                   cudaMemcpyDeviceToHost, a.stream),
                   "read decode graph input status");
  s.prepared = a;
  return true;
}

bool R2skDecodeGraph::LaunchPrepared() {
  auto& s = *state_;
  const auto a = s.prepared;
  s.prepared = {};
  if (!a.logits || *s.flag) return false;

  auto entry = std::find_if(
      s.graphs.begin(), s.graphs.end(),
      [&](const auto& e) { return Arguments(e.args) == Arguments(a); });
  const bool cached = entry != s.graphs.end();
  if (!cached) {
    cudaGraphExec_t executable = Capture(a, s.fork);
    if (s.graphs.size() == kMaxGraphs) {
      // Prepare's synchronization also completed previous uses of this cache.
      cudaGraphExecDestroy(s.graphs.front().exec);
      s.graphs.erase(s.graphs.begin());
    }
    s.graphs.push_back({a, executable});
    entry = s.graphs.end() - 1;
  }

  r2sk::cuda_check(cudaGraphLaunch(entry->exec, a.stream),
                   "launch decode sampler graph");
  const auto result = cudaEventRecord(s.done, a.stream);
  if (result != cudaSuccess) cudaStreamSynchronize(a.stream);
  r2sk::cuda_check(result, "record decode sampler completion");
  s.pending = true;
  if (s.trace) {
    LOG(INFO) << "R2SK_DECODE_GRAPH cached=" << cached << " batch=" << a.batch
              << " k=" << a.k;
  }
  return true;
}

}  // namespace allspark::cuda
