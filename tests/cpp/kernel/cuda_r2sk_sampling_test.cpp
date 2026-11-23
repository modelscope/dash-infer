// SPDX-License-Identifier: Apache-2.0
#include <core/kernel/cuda/topk_r2sk/sampling.h>
#include <core/kernel/cuda/cuda_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
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
    Require(argc == 1, "usage: cpp_r2sk_sampling_test [--replay FILE B N K]");
    TestSampling(device, stream.value);
    std::cout << "R2S-K sampling tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
