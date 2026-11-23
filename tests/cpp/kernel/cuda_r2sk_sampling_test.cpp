// SPDX-License-Identifier: Apache-2.0
#include <core/kernel/cuda/topk_r2sk/sampling.h>
#include <core/kernel/cuda/topk_r2sk/r2sk/bfloat16_graph.hpp>
#include <core/kernel/cuda/cuda_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Stream {
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  cudaStream_t value;
};

template <class T>
struct DeviceBuffer {
  explicit DeviceBuffer(size_t count) {
    Check(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * count));
  }
  ~DeviceBuffer() { cudaFree(data); }
  T* data;
};

float Decode(uint16_t bf16) {
  const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void TestSampling(int device, cudaStream_t stream) {
  constexpr int n = 2053;  // Odd length exercises packed tails and row alignment.
  constexpr int max_batch = 3;
  constexpr int max_k = 100;
  std::vector<uint16_t> input(max_batch * n, 0x3f80);  // All-tied first row.
  for (int i = 0; i < n; ++i) {
    input[n + i] = static_cast<uint16_t>(0xbf80 + (i * 23) % 128);
    input[2 * n + i] = (i % 2 == 0) ? 0 : 0x8000;
  }
  // Sparse maxima in the negative row exercise an early sort.
  for (int i = 0; i < 31; ++i) input[n + 37 * i] = 0x3f00 + i;
  // Enough positive zeros remain to force refinement while the saved
  // positive winners in this row can be sorted on the second stream.
  for (int i = 0; i < 5; ++i) input[2 * n + i] = 0x4000 + i;

  allspark::cuda::R2skSampling selector(device);
  const size_t bytes = selector.ReserveBytes(max_batch, n);
  Require(bytes > 0, "missing R2S-K scratch requirement");
  DeviceBuffer<uint16_t> logits(input.size());
  DeviceBuffer<uint16_t> values(max_batch * max_k + 2);
  DeviceBuffer<int> indices(max_batch * max_k + 2);
  DeviceBuffer<unsigned char> first(bytes), second(bytes);
  Check(cudaMemcpy(logits.data, input.data(), input.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice));

  for (int batch : {3, 1, 2, 3}) {
    for (int k : {2, 20, 100}) {
      auto* scratch = k == 20 ? first.data : second.data;
      std::vector<uint16_t> selected(batch * k + 2);
      std::vector<int> ids(batch * k + 2);
      Check(cudaMemsetAsync(values.data, 0xcc, selected.size() * sizeof(uint16_t), stream));
      Check(cudaMemsetAsync(indices.data, 0xcc, ids.size() * sizeof(int), stream));
      Require(selector.Try(logits.data, values.data + 1, indices.data + 1,
                           batch, n, k, scratch, bytes, stream),
              "BF16 selection fell back unexpectedly");
      // The real top-p stage also reuses this storage on the caller stream.
      Check(cudaMemsetAsync(scratch, 0xa5, bytes, stream));
      Check(cudaMemcpyAsync(selected.data(), values.data,
                            selected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
      Check(cudaMemcpyAsync(ids.data(), indices.data, ids.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
      Check(cudaStreamSynchronize(stream));
      Require(selected.front() == 0xcccc && selected.back() == 0xcccc &&
                  static_cast<uint32_t>(ids.front()) == 0xcccccccc &&
                  static_cast<uint32_t>(ids.back()) == 0xcccccccc,
              "output guards were overwritten");
      for (int row = 0; row < batch; ++row) {
        std::vector<float> expected(n);
        std::vector<bool> used(n);
        for (int j = 0; j < n; ++j) expected[j] = Decode(input[row * n + j]);
        std::partial_sort(expected.begin(), expected.begin() + k,
                          expected.end(), std::greater<float>{});
        for (int j = 0; j < k; ++j) {
          const int out = 1 + row * k + j;
          const int id = ids[out];
          Require(id >= 0 && id < n && !used[id], "invalid or duplicate index");
          used[id] = true;
          Require(selected[out] == input[row * n + id], "key/index mismatch");
          Require(Decode(selected[out]) == expected[j], "incorrect sorted top-k");
        }
      }
    }
  }
  std::vector<uint16_t> saved(input.size());
  Check(cudaMemcpy(saved.data(), logits.data, saved.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost));
  Require(saved == input, "selection modified its input");

  for (uint16_t nonfinite : {uint16_t{0xff80}, uint16_t{0x7f80}, uint16_t{0x7fc0}}) {
    // Put the value in a different row and at the last element of a partial tile.
    Check(cudaMemcpy(logits.data + input.size() - 1, &nonfinite, sizeof(nonfinite),
                     cudaMemcpyHostToDevice));
    Check(cudaMemsetAsync(values.data, 0xcc, 3 * 20 * sizeof(uint16_t), stream));
    Check(cudaMemsetAsync(indices.data, 0xcc, 3 * 20 * sizeof(int), stream));
    Require(!selector.Try(logits.data, values.data, indices.data, 3, n, 20,
                          first.data, bytes, stream), "nonfinite input must fall back");
    uint16_t untouched[60];
    int untouched_ids[60];
    Check(cudaMemcpyAsync(untouched, values.data, sizeof(untouched),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaMemcpyAsync(untouched_ids, indices.data, sizeof(untouched_ids),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    for (int j = 0; j < 60; ++j) {
      Require(untouched[j] == 0xcccc &&
                  static_cast<uint32_t>(untouched_ids[j]) == 0xcccccccc,
              "fallback changed the output");
    }
  }
  Require(!selector.Try(logits.data, values.data, indices.data, 1, n, 512,
                        first.data, bytes, stream), "large k must fall back");
  Require(!selector.Try(logits.data, values.data, indices.data, 1, n, 20,
                        first.data, bytes - 1, stream), "short arena must fall back");
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const bool captured = selector.Try(logits.data, values.data, indices.data, 1,
                                     n, 20, first.data, bytes, stream);
  cudaGraph_t graph;
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphDestroy(graph));
  Require(!captured, "R2S-K must defer to the existing capture path");
}

// A matched complete-call replay against the RadiK compiled into this tree.
// The input is dense little-endian BF16 words from a pre-top-k capture.
void Replay(int device, cudaStream_t stream, const char* path, int batch,
            int n, int k) {
  Require(batch >= 1 && batch <= 65535 && k > 1 && k <= 256 && k < n,
          "unsupported replay shape");
  const size_t count = static_cast<size_t>(batch) * n;
  std::vector<uint16_t> input(count);
  std::ifstream file(path, std::ios::binary);
  Require(static_cast<bool>(file.read(reinterpret_cast<char*>(input.data()),
                                      count * sizeof(uint16_t))) &&
              file.peek() == std::ifstream::traits_type::eof(),
          "capture size must match batch * vocabulary * 2");
  Require(std::none_of(input.begin(), input.end(), [](uint16_t key) {
            return (key & 0x7f80u) == 0x7f80u;
          }), "replay requires finite logits");
  const size_t output_count = static_cast<size_t>(batch) * k;
  std::vector<uint16_t> reference(output_count), selected(output_count);
  std::vector<int> ids(output_count);
  std::vector<float> expected(output_count);
  for (int row = 0; row < batch; ++row) {
    const auto begin = input.begin() + static_cast<size_t>(row) * n;
    std::vector<float> oracle(n);
    std::transform(begin, begin + n, oracle.begin(), Decode);
    const auto result = expected.begin() + static_cast<size_t>(row) * k;
    std::partial_sort_copy(oracle.begin(), oracle.end(), result, result + k,
                           std::greater<float>{});
  }
  size_t native_bytes = 0;
  using Bf16 = hie::bfloat16;
  static_assert(sizeof(Bf16) == sizeof(uint16_t));
  allspark::cuda::TopKRadixGetWorkspaceSize<Bf16>(&native_bytes, batch, n);
  allspark::cuda::R2skSampling selector(device);
  const size_t r2sk_bytes = selector.ReserveBytes(batch, n);
  const size_t bytes = std::max(native_bytes, r2sk_bytes);
  DeviceBuffer<unsigned char> scratch(bytes);
  DeviceBuffer<uint16_t> logits(count), out(output_count);
  DeviceBuffer<int> indices(output_count);
  Check(cudaMemcpy(logits.data, input.data(), count * sizeof(uint16_t),
                   cudaMemcpyHostToDevice));
  auto invoke = [&](bool r2sk) {
    if (r2sk) {
      Require(selector.Try(logits.data, out.data, indices.data, batch, n, k,
                           scratch.data, bytes, stream), "replay fell back to RadiK");
    } else {
      allspark::cuda::TopKRadixKernelLauncher(
          reinterpret_cast<Bf16*>(out.data), indices.data,
          reinterpret_cast<const Bf16*>(logits.data), scratch.data,
          batch, n, k, stream);
    }
    Check(cudaStreamSynchronize(stream));
  };
  for (bool r2sk : {false, true}) {
    Check(cudaMemsetAsync(indices.data, 0xff, ids.size() * sizeof(int), stream));
    invoke(r2sk);
    auto& keys = r2sk ? selected : reference;
    Check(cudaMemcpy(keys.data(), out.data, keys.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(ids.data(), indices.data, ids.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    for (int row = 0; row < batch; ++row) {
      std::vector<bool> used(n);
      for (int j = 0; j < k; ++j) {
        const size_t pos = static_cast<size_t>(row) * k + j;
        const int id = ids[pos];
        Require(id >= 0 && id < n && !used[id], "invalid replay index");
        used[id] = true;
        Require(keys[pos] == input[static_cast<size_t>(row) * n + id] &&
                    Decode(keys[pos]) == expected[pos], "incorrect replay output");
      }
    }
  }
  for (int i = 0; i < 10; ++i) {
    invoke(false);
    invoke(true);
  }
  std::cout << "rep,method,batch,n,k,required_scratch_bytes,host_us\n";
  for (int rep = 0; rep < 101; ++rep) {
    for (int order = 0; order < 2; ++order) {
      const bool r2sk = (rep + order) % 2 != 0;
      const auto start = std::chrono::steady_clock::now();
      invoke(r2sk);
      const auto elapsed = std::chrono::steady_clock::now() - start;
      std::cout << rep << ',' << (r2sk ? "r2sk" : "radik") << ',' << batch << ','
                << n << ',' << k << ',' << (r2sk ? r2sk_bytes : native_bytes) << ','
                << std::chrono::duration<double, std::micro>(elapsed).count() << '\n';
    }
  }
}

struct Event {
  explicit Event(unsigned flags = cudaEventDisableTiming) {
    Check(cudaEventCreateWithFlags(&value, flags));
  }
  ~Event() { cudaEventDestroy(value); }
  cudaEvent_t value;
};

struct CapturedSelector {
  ~CapturedSelector() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
  }
  void Finish(cudaStream_t stream) {
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&exec, graph, 0));
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};

bool IsNan(uint16_t value) { return (value & 0x7fffu) > 0x7f80u; }

void CheckGraphOutput(const std::vector<uint16_t>& input,
                      const std::vector<uint16_t>& values,
                      const std::vector<uint32_t>& indices,
                      r2sk::BFloat16GraphShape shape, size_t guard = 0) {
  for (uint32_t row = 0; row < shape.batch; ++row) {
    const auto begin = input.begin() + size_t(row) * shape.vocabulary;
    const auto end = begin + shape.vocabulary;
    if (std::any_of(begin, end, IsNan)) {
      for (uint32_t j = 0; j < shape.k; ++j) {
        const size_t pos = size_t(row) * shape.k + j + guard;
        Require(values[pos] == 0xccccu && indices[pos] == 0xccccccccu,
                "NaN row modified output");
      }
      continue;
    }
    std::vector<float> expected(shape.vocabulary);
    std::transform(begin, end, expected.begin(), Decode);
    std::partial_sort(expected.begin(), expected.begin() + shape.k,
                      expected.end(), std::greater<float>{});
    std::vector<bool> used(shape.vocabulary);
    for (uint32_t j = 0; j < shape.k; ++j) {
      const size_t pos = size_t(row) * shape.k + j + guard;
      const uint32_t index = indices[pos];
      Require(index < shape.vocabulary && !used[index], "invalid graph index");
      used[index] = true;
      Require(begin[index] == values[pos] && Decode(values[pos]) == expected[j],
              "incorrect graph value or occurrence");
    }
  }
}

void TestGraph(cudaStream_t main) {
  using r2sk::GraphRoute;
  Stream winners;
  Event ready, done;
  const r2sk::BFloat16GraphFork fork{winners.value, ready.value, done.value};
  for (const r2sk::BFloat16GraphShape shape : {
           r2sk::BFloat16GraphShape{4, 2053, 20},
           r2sk::BFloat16GraphShape{1, 2, 2},
           r2sk::BFloat16GraphShape{3, 2099, 256},
       }) {
    const auto bytes = r2sk::bfloat16_graph_bytes(shape);
    DeviceBuffer<unsigned char> scratch(bytes);
    const size_t n = shape.vocabulary;
    const size_t output_count = size_t(shape.batch) * shape.k;
    std::vector<uint16_t> input(n * shape.batch), saved(input.size());
    std::vector<uint16_t> out(output_count + 2);
    std::vector<uint32_t> indices(output_count + 2);
    std::vector<r2sk::BFloat16GraphRow> reports(shape.batch);
    DeviceBuffer<uint16_t> logits(input.size()), values(out.size());
    DeviceBuffer<uint32_t> ids(indices.size());
    DeviceBuffer<r2sk::BFloat16GraphRow> device_reports(shape.batch);

    for (auto resources : {fork, r2sk::BFloat16GraphFork{}}) {
      static_assert(sizeof(uint16_t) == 2);
      CapturedSelector captured;
      Check(cudaStreamBeginCapture(main, cudaStreamCaptureModeGlobal));
      r2sk::enqueue_bfloat16_graph(shape, logits.data, values.data + 1, ids.data + 1,
          scratch.data, bytes, main, resources, device_reports.data);
      // The output consumer may reuse the arena immediately after the join.
      Check(cudaMemsetAsync(scratch.data, 0xa5, bytes, main));
      captured.Finish(main);
      for (uint32_t rep = 0; rep < 18; ++rep) {
        for (uint32_t row = 0; row < shape.batch; ++row) {
          auto* src = input.data() + row * n;
          uint32_t pattern = (row + rep) % 6;
          std::fill_n(src, n, uint16_t{0xbf80});
          if (pattern == 0 || pattern == 1) {
            const uint32_t high = std::min(shape.vocabulary,
                shape.k + (pattern == 1 ? 9 : 0));
            for (uint32_t j = 0; j < high; ++j) src[j] = 0x4000u + j % 16;
          } else if (pattern == 2) {
            for (size_t j = 0; j < n; ++j) src[j] = 0xbf8fu - j % 16;
            for (size_t j = 0; j < std::min(n, size_t{5}); ++j) src[j] = 0x4000u + j;
          } else if (pattern == 3) {
            src[n - 1] = 0x7fc1u;
          } else if (pattern == 4) {
            for (size_t j = 0; j < n; ++j) src[j] = j % 2 ? 0x8000u : 0;
          } else {
            std::fill_n(src, n, uint16_t{0xff80});
            if (rep % 2 == 1) src[n - 1] = 0x7f80u;
          }
        }
        Check(cudaMemcpyAsync(logits.data, input.data(), input.size() * 2,
                              cudaMemcpyHostToDevice, main));
        Check(cudaMemsetAsync(values.data, 0xcc, out.size() * 2, main));
        Check(cudaMemsetAsync(ids.data, 0xcc, indices.size() * 4, main));
        Check(cudaGraphLaunch(captured.exec, main));
        Check(cudaStreamSynchronize(main));
        Check(cudaMemcpy(out.data(), values.data, out.size() * 2, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(indices.data(), ids.data, indices.size() * 4, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(saved.data(), logits.data, saved.size() * 2, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(reports.data(), device_reports.data, reports.size() * sizeof(reports[0]),
                         cudaMemcpyDeviceToHost));
        Require(saved == input, "graph modified its input");
        Require(out.front() == 0xcccc && out.back() == 0xcccc &&
                    indices.front() == 0xcccccccc && indices.back() == 0xcccccccc,
                "graph output overrun");
        CheckGraphOutput(input, out, indices, shape, 1);
        for (uint32_t row = 0; row < shape.batch; ++row) {
          const auto pattern = (row + rep) % 6;
          const auto& report = reports[row];
          if (pattern == 0)
            Require(report.route == GraphRoute::exact, "expected exact route");
          if (pattern == 1 && n > 1024)
            Require(report.route == GraphRoute::early_sort, "expected early sort");
          if (pattern == 2 && n > 1024)
            Require(report.route == GraphRoute::refine && report.first_winners == 5 &&
                        report.passes == 2, "expected refine with independent winners");
          if (pattern == 3)
            Require(report.route == GraphRoute::invalid_nan, "expected NaN rejection");
        }
      }
    }
  }
  std::cout << "PASS graph BF16: 108 captured replays, changed routes and logits, "
               "both schedules, mixed rows, signed zeros, infinities, NaN, guards and arena reuse.\n";
}

void BenchGraph(int device, cudaStream_t stream, const std::string& input_source,
                r2sk::BFloat16GraphShape shape) {
  const size_t count = size_t(shape.batch) * shape.vocabulary;
  const size_t kept = size_t(shape.batch) * shape.k;
  std::vector<uint16_t> input(count), values(kept);
  std::vector<uint32_t> indices(kept);
  if (input_source == "gaussian") {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    for (auto& w : input) {
      float value = normal(rng);
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      w = static_cast<uint16_t>(bits >> 16);
    }
  } else {
    std::ifstream file(input_source, std::ios::binary);
    Require(static_cast<bool>(file.read(reinterpret_cast<char*>(input.data()), count * 2)) &&
                file.peek() == std::ifstream::traits_type::eof(), "graph fixture size");
  }

  size_t radik_bytes = 0;
  using Bf16 = hie::bfloat16;
  allspark::cuda::TopKRadixGetWorkspaceSize<Bf16>(&radik_bytes, shape.batch, shape.vocabulary);
  allspark::cuda::R2skSampling original(device);
  const size_t original_bytes = original.ReserveBytes(shape.batch, shape.vocabulary);
  const auto graph_bytes = r2sk::bfloat16_graph_bytes(shape);
  const size_t bytes = std::max({radik_bytes, original_bytes, size_t(graph_bytes)});
  DeviceBuffer<unsigned char> scratch(bytes);
  DeviceBuffer<uint16_t> logits(count), out(kept);
  DeviceBuffer<uint32_t> ids(kept);
  Stream winners;
  Event ready, done;
  const r2sk::BFloat16GraphFork fork{winners.value, ready.value, done.value};
  Check(cudaMemcpy(logits.data, input.data(), count * 2, cudaMemcpyHostToDevice));
  auto enqueue = [&](r2sk::BFloat16GraphFork resources) {
    r2sk::enqueue_bfloat16_graph(shape, logits.data, out.data, ids.data,
        scratch.data, bytes, stream, resources);
  };
  CapturedSelector overlap, serial;
  for (auto* captured : {&overlap, &serial}) {
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue(captured == &overlap ? fork : r2sk::BFloat16GraphFork{});
    captured->Finish(stream);
  }
  auto invoke = [&](int method) {
    switch (method) {
      case 0:
        allspark::cuda::TopKRadixKernelLauncher(reinterpret_cast<Bf16*>(out.data),
            reinterpret_cast<int*>(ids.data), reinterpret_cast<const Bf16*>(logits.data),
            scratch.data, shape.batch, shape.vocabulary, shape.k, stream);
        break;
      case 1:
        Require(original.Try(logits.data, out.data, reinterpret_cast<int*>(ids.data),
            shape.batch, shape.vocabulary, shape.k, scratch.data, bytes, stream),
            "original R2S-K declined benchmark");
        break;
      case 2:
        enqueue(fork);
        break;
      case 3:
        Check(cudaGraphLaunch(overlap.exec, stream));
        break;
      case 4:
        Check(cudaGraphLaunch(serial.exec, stream));
        break;
    }
  };
  constexpr int methods = 5;
  for (int method = 0; method < methods; ++method) {
    Check(cudaMemsetAsync(ids.data, 0xff, kept * 4, stream));
    invoke(method);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(values.data(), out.data, kept * 2, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(indices.data(), ids.data, kept * 4, cudaMemcpyDeviceToHost));
    CheckGraphOutput(input, values, indices, shape);
  }
  const char* names[methods] = {"radik_eager", "r2sk_pr_eager", "device_eager",
                                "graph_overlap", "graph_serial"};
  const size_t arena_bytes[methods] = {radik_bytes, original_bytes, size_t(graph_bytes),
                                        size_t(graph_bytes), size_t(graph_bytes)};
  for (int rep = 0; rep < 12; ++rep)
    for (int method = 0; method < methods; ++method) {
      invoke(method);
      Check(cudaStreamSynchronize(stream));
    }
  Event start(cudaEventDefault), stop(cudaEventDefault);
  std::cout << "rep,method,batch,n,k,scratch_bytes,host_us,event_us\n";
  for (int rep = 0; rep < 51; ++rep) {
    for (int position = 0; position < methods; ++position) {
      const int method = (position + rep) % methods;
      const auto tick = std::chrono::steady_clock::now();
      invoke(method);
      Check(cudaStreamSynchronize(stream));
      const auto elapsed = std::chrono::steady_clock::now() - tick;
      // Measure a second complete call for device timing so the host result
      // excludes the event API cost and each arm has identical accounting.
      Check(cudaEventRecord(start.value, stream));
      invoke(method);
      Check(cudaEventRecord(stop.value, stream));
      Check(cudaEventSynchronize(stop.value));
      float gpu_ms;
      Check(cudaEventElapsedTime(&gpu_ms, start.value, stop.value));
      std::cout << rep << ',' << names[method] << ',' << shape.batch << ','
                << shape.vocabulary << ',' << shape.k << ',' << arena_bytes[method] << ','
                << std::chrono::duration<double, std::micro>(elapsed).count() << ','
                << gpu_ms * 1000 << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device;
    Check(cudaGetDevice(&device));
    int major;
    Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 8) {
      std::cout << "R2S-K sampling tests require compute capability 8.0 or newer\n";
      return 77;
    }
    Stream stream;
    if (argc == 6 && std::string(argv[1]) == "--replay") {
      Replay(device, stream.value, argv[2], std::stoi(argv[3]),
             std::stoi(argv[4]), std::stoi(argv[5]));
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--graph-test") {
      TestGraph(stream.value);
      return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "--graph-replay") {
      BenchGraph(device, stream.value, argv[2],
                 {static_cast<uint32_t>(std::stoul(argv[3])),
                  static_cast<uint32_t>(std::stoul(argv[4])),
                  static_cast<uint32_t>(std::stoul(argv[5]))});
      return 0;
    }
    Require(argc == 1, "usage: cpp_r2sk_sampling_test [--graph-test | --graph-replay FILE B N K | --replay FILE B N K]");
    TestSampling(device, stream.value);
    std::cout << "R2S-K sampling tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
