// Histogram and staged compaction adapted from ideas in RadiK.
// Copyright (c) 2023-present Alibaba Inc. MIT license: LICENSE.RadiK.
// Kernels below are rewritten for single/batched rows, bounded dual
// destinations, scalar/packed loads, reusable allocations, and optional carried
// indices.
#include "r2sk/cuda.hpp"
#include "r2sk/bfloat16_graph.hpp"
#include "r2sk/primitive_plan.hpp"
#include "r2sk/workspace_plan.hpp"
#include <math_constants.h>
#include <functional>
#include <new>
#include <utility>

namespace r2sk {
void cuda_check(cudaError_t e, const char* op) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
#define CK(x) cuda_check((x), #x)
namespace {
constexpr int BLOCK = 256;
template <class Value>
using KernelJobT = detail::KernelJobT<Value>;
#include "primitives.cuh"
template <bool INDEXED, class Value = float>
__device__ void put(ValueView<Value> src, uint32_t j, ValueBuffer<Value> dst,
                    uint32_t pos) {
  dst.v[pos] = src.v[j];
  if (INDEXED) dst.i[pos] = src.i ? src.i[j] : j;
}
template <int BINS, bool BATCHED = false, class Value = float>
__global__ void count_bin(ValueView<Value> src, uint32_t n, int pass,
                          uint32_t* hist,
                          const KernelJobT<Value>* jobs = nullptr,
                          uint32_t* scale_unsafe = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    src = job.src;
    n = job.n;
    pass = job.pass;
    hist = job.hist;
    scale_unsafe = job.scale_unsafe;
  }
  __shared__ uint32_t local[BINS];
  float reference =
      src.shift_reference && pass < 3 ? *src.shift_reference : 0.f;
  for (int j = threadIdx.x; j < BINS; j += BLOCK) local[j] = 0;
  __syncthreads();
  int unsafe = 0;
  bool shifted = src.shift_reference && pass < 3;
  for (uint64_t j = uint64_t(blockIdx.x) * BLOCK + threadIdx.x; j < n;
       j += uint64_t(gridDim.x) * BLOCK) {
    float value = src.v[j],
          key_value = shifted ? shift_value(value, reference) : value;
    atomicAdd(local + digit(key_value, pass), 1u);
    if (scale_unsafe &&
        (shifted ? !shift_reversible(value, key_value, reference)
                 : !finite_key(value)))
      unsafe = 1;
  }
  if (scale_unsafe) {
    int bad = __syncthreads_or(unsafe);
    if (threadIdx.x == 0 && bad) atomicOr(scale_unsafe, 1u);
  } else
    __syncthreads();
  for (int j = threadIdx.x; j < BINS; j += BLOCK)
    if (local[j]) atomicAdd(hist + j, local[j]);
}
// Optional capture retains upper bins of each disjoint 256-value tile while
// every loaded float is still in a register. Logical counts include overflow;
// no truncated capture can pass the subsequent coverage check.
template <int BINS, bool INDEXED, class Value = float>
__global__ void count_capture(ValueView<Value> src, uint32_t n, int pass,
                              uint32_t* hist, uint32_t* captured_hist,
                              ValueBuffer<Value> captured, uint32_t capacity,
                              uint32_t* total, uint32_t local_rank,
                              uint32_t* scale_unsafe = nullptr) {
  __shared__ uint32_t local[BINS], warp_max[BLOCK / 32], nibble[16];
  __shared__ uint32_t cutoff, residual, accepted, write_base;
  __shared__ float values[BLOCK];
  __shared__ uint32_t indices[INDEXED ? BLOCK : 1], digits[BLOCK];
  float reference =
      src.shift_reference && pass < 3 ? *src.shift_reference : 0.f;
  for (int j = threadIdx.x; j < BINS; j += BLOCK) local[j] = 0;
  __syncthreads();
  int unsafe = 0;
  bool shifted = src.shift_reference && pass < 3;
  for (uint64_t base = uint64_t(blockIdx.x) * BLOCK; base < n;
       base += uint64_t(gridDim.x) * BLOCK) {
    uint64_t j = base + threadIdx.x;
    bool valid = j < n;
    float value = valid ? src.v[j] : 0.f,
          key_value = shifted ? shift_value(value, reference) : value;
    uint32_t d = valid ? digit(key_value, pass) : 0;
    if (valid && scale_unsafe && !shift_reversible(value, key_value, reference))
      unsafe = 1;
    if (valid) atomicAdd(local + d, 1u);
    if (local_rank == 1) {
      uint32_t maximum = d;
      for (int delta = 16; delta; delta >>= 1) {
        uint32_t other = __shfl_down_sync(0xffffffffu, maximum, delta);
        maximum = maximum > other ? maximum : other;
      }
      if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x / 32] = maximum;
      __syncthreads();
      if (threadIdx.x == 0) {
        cutoff = 0;
        for (int w = 0; w < BLOCK / 32; ++w)
          cutoff = cutoff > warp_max[w] ? cutoff : warp_max[w];
      }
      __syncthreads();
    } else {
      // A small radix select on DIGITS, not values. Three four-bit
      // rounds (two for pass 3) find the q-th digit, preserving bin ties.
      if (threadIdx.x == 0) {
        cutoff = 0;
        uint32_t remaining = uint32_t(n - base);
        residual = local_rank < remaining ? local_rank : remaining;
      }
      __syncthreads();
      for (int shift = digit_width(pass) - 4; shift >= 0; shift -= 4) {
        if (threadIdx.x < 16) nibble[threadIdx.x] = 0;
        __syncthreads();
        if (valid && (d >> (shift + 4)) == cutoff)
          atomicAdd(nibble + ((d >> shift) & 15u), 1u);
        __syncthreads();
        if (threadIdx.x == 0) {
          for (int bin = 15; bin >= 0; --bin) {
            if (residual <= nibble[bin]) {
              cutoff = (cutoff << 4) | uint32_t(bin);
              break;
            }
            residual -= nibble[bin];
          }
        }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0) accepted = 0;
    __syncthreads();
    if (valid && d >= cutoff) {
      uint32_t position = atomicAdd(&accepted, 1u);
      values[position] = value;
      digits[position] = d;
      if (INDEXED) indices[position] = src.i ? src.i[j] : uint32_t(j);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      write_base = atomicAdd(total, accepted);
      if (local_rank == 1) atomicAdd(captured_hist + cutoff, accepted);
    }
    __syncthreads();
    uint32_t t = threadIdx.x;
    if (t < accepted) {
      if (local_rank != 1) atomicAdd(captured_hist + digits[t], 1u);
      if (write_base + t < capacity) {
        captured.v[write_base + t] = values[t];
        if (INDEXED) captured.i[write_base + t] = indices[t];
      }
    }
    __syncthreads();  // staging and cutoff can be reused only after every
                      // reader finishes
  }
  if (scale_unsafe) {
    int bad = __syncthreads_or(unsafe);
    if (threadIdx.x == 0 && bad) atomicOr(scale_unsafe, 1u);
  }
  for (int j = threadIdx.x; j < BINS; j += BLOCK)
    if (local[j]) atomicAdd(hist + j, local[j]);
}
// Reversed inclusive scan: scan[i] counts bins >= BINS-1-i.
// All threads participate in every barrier, including partial input tiles.
template <int BINS, bool BATCHED = false, bool CAPTURE = false,
          class Value = float>
__global__ void select_bin(const uint32_t* hist, uint32_t rank, Decision* d,
                           const KernelJobT<Value>* jobs = nullptr,
                           const uint32_t* captured_hist = nullptr,
                           const uint32_t* captured_count = nullptr,
                           const uint32_t* scale_unsafe = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    hist = job.hist;
    rank = job.rank;
    d = job.decision;
    scale_unsafe = job.scale_unsafe;
  }
  __shared__ uint32_t scan[BINS];
  for (int j = threadIdx.x; j < BINS; j += BLOCK) scan[j] = hist[BINS - 1 - j];
  __syncthreads();
  for (int step = 1; step < BINS; step <<= 1) {
    uint32_t tmp[BINS / BLOCK];
#pragma unroll
    for (int t = 0; t < BINS / BLOCK; ++t) {
      int j = threadIdx.x + t * BLOCK;
      tmp[t] = scan[j] + (j >= step ? scan[j - step] : 0u);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < BINS / BLOCK; ++t)
      scan[threadIdx.x + t * BLOCK] = tmp[t];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    d->total = scan[BINS - 1];
    d->scale_unsafe = scale_unsafe ? *scale_unsafe : 0;
    if constexpr (!CAPTURE) {
      d->captured = 0;
      d->captured_retained = 0;
    }
  }
  for (int j = threadIdx.x; j < BINS; j += BLOCK) {
    uint32_t before = j ? scan[j - 1] : 0;
    if (before < rank && rank <= scan[j]) {
      d->bin = BINS - 1 - j;
      d->h = before;
      d->B = scan[j] - before;
    }
  }
  if constexpr (CAPTURE) {
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t retained = 0;
      for (int bin = d->bin; bin < BINS; ++bin) retained += captured_hist[bin];
      d->captured = *captured_count;
      d->captured_retained = retained;
    }
  }
}
// Two shared staging arrays. Each tile reserves at most one global range per
// destination; no global atomic per accepted value. Reservations count ALL E,
// but writes are bounded by ecap (terminal ties only). Null G discards winners.
template <bool INDEXED, bool FILTER, bool BATCHED = false, class Value = float>
__global__ void gather(ValueView<Value> src, uint32_t n, int pass,
                       uint32_t boundary, ValueBuffer<Value> G, uint32_t gcap,
                       ValueBuffer<Value> E, uint32_t ecap, uint32_t* counters,
                       const KernelJobT<Value>* jobs = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    src = job.src;
    n = job.n;
    pass = job.pass;
    boundary =
        FILTER && job.pivot.v ? ordered_key(job.pivot.v[0]) : job.boundary;
    G = job.G;
    gcap = job.gcap;
    E = job.E;
    ecap = job.ecap;
    counters = job.counters;
  }
  __shared__ float gv[BLOCK], ev[BLOCK];
  __shared__ uint32_t gi[INDEXED ? BLOCK : 1], ei[INDEXED ? BLOCK : 1];
  __shared__ uint32_t gc, ec, gb, eb;
  float reference =
      src.shift_reference && pass < 3 ? *src.shift_reference : 0.f;
  for (uint64_t base = uint64_t(blockIdx.x) * BLOCK; base < n;
       base += uint64_t(gridDim.x) * BLOCK) {
    if (threadIdx.x == 0) {
      gc = 0;
      ec = 0;
    }
    __syncthreads();
    uint64_t j = base + threadIdx.x;
    if (j < n) {
      float v = src.v[j];
      uint32_t d =
          FILTER ? ordered_key(v)
                 : digit(v, pass, reference, src.shift_reference != nullptr);
      if (d > boundary && G.v) {
        uint32_t pos = atomicAdd(&gc, 1u);
        gv[pos] = v;
        if (INDEXED) gi[pos] = src.i ? src.i[j] : uint32_t(j);
      } else if (d == boundary) {
        uint32_t pos = atomicAdd(&ec, 1u);
        ev[pos] = v;
        if (INDEXED) ei[pos] = src.i ? src.i[j] : uint32_t(j);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      gb = gc ? atomicAdd(counters, gc) : 0;
      eb = ec ? atomicAdd(counters + 1, ec) : 0;
    }
    __syncthreads();
    uint32_t t = threadIdx.x;
    if (t < gc && gb + t < gcap) {
      G.v[gb + t] = gv[t];
      if (INDEXED) G.i[gb + t] = gi[t];
    }
    if (t < ec && eb + t < ecap) {
      E.v[eb + t] = ev[t];
      if (INDEXED) E.i[eb + t] = ei[t];
    }
    __syncthreads();  // staging may be reused by the next tile
  }
}
// Host launch helpers are shared by the single-array and batched backends.
int histogram_threads(PrimitiveMode mode, int pass) {
  return mode == PrimitiveMode::radik && pass == 0 ? 1024 : BLOCK;
}
bool buffered_mode(PrimitiveMode mode) {
  return mode >= PrimitiveMode::buffered;
}
uint32_t primitive_blocks(uint32_t n, uint32_t cap, PrimitiveMode mode,
                          int threads = BLOCK) {
  uint32_t tile = uint32_t(threads) * (mode == PrimitiveMode::scalar ? 1u : 4u);
  return std::min(cap, std::max(1u, (n + tile - 1) / tile));
}
template <bool BATCHED = false, class Value = float>
void launch_histogram(ValueView<Value> src, uint32_t n, int pass,
                      uint32_t* hist, const KernelJobT<Value>* jobs, dim3 shape,
                      PrimitiveMode mode, cudaStream_t stream,
                      uint32_t* scale_unsafe = nullptr) {
  if (mode == PrimitiveMode::scalar) {
    if (digit_width(pass) == 8)
      count_bin<256, BATCHED, Value>
          <<<shape, BLOCK, 0, stream>>>(src, n, pass, hist, jobs, scale_unsafe);
    else
      count_bin<4096, BATCHED, Value>
          <<<shape, BLOCK, 0, stream>>>(src, n, pass, hist, jobs, scale_unsafe);
  } else if (histogram_threads(mode, pass) == 1024) {
    count_packed<4096, 1024, BATCHED, Value>
        <<<shape, 1024, 0, stream>>>(src, n, pass, hist, jobs, scale_unsafe);
  } else {
    if (digit_width(pass) == 8)
      count_packed<256, BLOCK, BATCHED, Value>
          <<<shape, BLOCK, 0, stream>>>(src, n, pass, hist, jobs, scale_unsafe);
    else
      count_packed<4096, BLOCK, BATCHED, Value>
          <<<shape, BLOCK, 0, stream>>>(src, n, pass, hist, jobs, scale_unsafe);
  }
}
template <bool BATCHED = false, bool CAPTURE = false, class Value = float>
void launch_selection(const uint32_t* hist, uint32_t rank, Decision* d,
                      const KernelJobT<Value>* jobs, int pass, dim3 shape,
                      PrimitiveMode mode, cudaStream_t stream,
                      const uint32_t* captured_hist = nullptr,
                      const uint32_t* captured_count = nullptr,
                      const uint32_t* scale_unsafe = nullptr) {
  if (mode >= PrimitiveMode::scan) {
    if (digit_width(pass) == 8)
      select_bin_warp<256, BATCHED, CAPTURE, Value><<<shape, 256, 0, stream>>>(
          hist, rank, d, jobs, captured_hist, captured_count, scale_unsafe);
    else
      select_bin_warp<4096, BATCHED, CAPTURE, Value>
          <<<shape, 1024, 0, stream>>>(hist, rank, d, jobs, captured_hist,
                                       captured_count, scale_unsafe);
  } else {
    if (digit_width(pass) == 8)
      select_bin<256, BATCHED, CAPTURE, Value><<<shape, BLOCK, 0, stream>>>(
          hist, rank, d, jobs, captured_hist, captured_count, scale_unsafe);
    else
      select_bin<4096, BATCHED, CAPTURE, Value><<<shape, BLOCK, 0, stream>>>(
          hist, rank, d, jobs, captured_hist, captured_count, scale_unsafe);
  }
}
template <bool INDEXED, bool FILTER, bool BATCHED = false, class Value = float>
void launch_partition(ValueView<Value> src, uint32_t n, int pass, uint32_t bin,
                      ValueBuffer<Value> G, uint32_t gc, ValueBuffer<Value> E,
                      uint32_t ec, uint32_t* counters,
                      const KernelJobT<Value>* jobs, dim3 shape,
                      uint32_t max_gcap, PrimitiveMode mode,
                      cudaStream_t stream) {
  if (mode == PrimitiveMode::scalar) {
    gather<INDEXED, FILTER, BATCHED, Value><<<shape, BLOCK, 0, stream>>>(
        src, n, pass, bin, G, gc, E, ec, counters, jobs);
  } else {
    uint32_t bytes =
        gather_shared_bytes(max_gcap, INDEXED, buffered_mode(mode));
    if (buffered_mode(mode))
      gather_packed<INDEXED, FILTER, true, BATCHED, Value>
          <<<shape, BLOCK, bytes, stream>>>(src, n, pass, bin, G, gc, E, ec,
                                            counters, jobs);
    else
      gather_packed<INDEXED, FILTER, false, BATCHED, Value>
          <<<shape, BLOCK, bytes, stream>>>(src, n, pass, bin, G, gc, E, ec,
                                            counters, jobs);
  }
}
template <bool INDEXED, bool BATCHED = false, class Value = float>
__global__ void copy_items(ValueView<Value> src, uint32_t n,
                           ValueBuffer<Value> dst,
                           const KernelJobT<Value>* jobs = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    src = job.src;
    n = job.n;
    dst = job.G;
  }
  for (uint64_t j = uint64_t(blockIdx.x) * BLOCK + threadIdx.x; j < n;
       j += uint64_t(gridDim.x) * BLOCK)
    put<INDEXED, Value>(src, uint32_t(j), dst, uint32_t(j));
}
template <bool INDEXED, bool BATCHED = false, class Value = float>
__global__ void sort_tiles(ValueBuffer<Value> data, uint32_t n, uint32_t tile,
                           const KernelJobT<Value>* jobs = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    data = job.G;
    n = job.n;
    tile = job.tile;
  }
  extern __shared__ uint32_t smem[];
  float* v = reinterpret_cast<float*>(smem);
  uint32_t* ix = smem + tile;
  uint64_t base = uint64_t(blockIdx.x) * tile;
  if (base >= n) return;  // uniform CTA exit, before any barrier
  for (uint32_t t = threadIdx.x; t < tile; t += BLOCK) {
    bool valid = base + t < n;
    v[t] = valid ? data.v[base + t] : -CUDART_INF_F;
    if (INDEXED) ix[t] = valid ? data.i[base + t] : 0xffffffffu;
  }
  __syncthreads();
  for (uint32_t size = 2; size <= tile; size <<= 1)
    for (uint32_t stride = size >> 1; stride; stride >>= 1) {
      for (uint32_t t = threadIdx.x; t < tile; t += BLOCK) {
        uint32_t other = t ^ stride;
        if (other > t) {
          bool descending = (t & size) == 0;
          bool swap = descending ? v[t] < v[other] : v[t] > v[other];
          if (swap) {
            float tmp = v[t];
            v[t] = v[other];
            v[other] = tmp;
            if (INDEXED) {
              uint32_t z = ix[t];
              ix[t] = ix[other];
              ix[other] = z;
            }
          }
        }
      }
      __syncthreads();
    }
  for (uint32_t t = threadIdx.x; t < tile; t += BLOCK)
    if (base + t < n) {
      data.v[base + t] = v[t];
      if (INDEXED) data.i[base + t] = ix[t];
    }
}
// Correct general fallback: parallel binary-rank merge of adjacent sorted runs.
// Equal values from the left run precede the right run, yielding unique ranks.
template <bool INDEXED, bool BATCHED = false, class Value = float>
__global__ void merge_runs(ValueView<Value> src, ValueBuffer<Value> dst,
                           uint32_t n, uint32_t width,
                           const KernelJobT<Value>* jobs = nullptr) {
  if constexpr (BATCHED) {
    auto job = jobs[blockIdx.y];
    src = job.src;
    dst = job.G;
    n = job.n;
    width = job.width;
  }
  for (uint64_t pos = uint64_t(blockIdx.x) * BLOCK + threadIdx.x; pos < n;
       pos += uint64_t(gridDim.x) * BLOCK) {
    uint64_t base = (pos / (2ull * width)) * (2ull * width);
    uint64_t mid = base + width < n ? base + width : n;
    uint64_t end = base + 2ull * width < n ? base + 2ull * width : n;
    bool left = pos < mid;
    uint64_t lo = left ? mid : base, hi = left ? end : mid;
    uint64_t other_base = lo;
    float value = src.v[pos];
    while (lo < hi) {
      uint64_t m = (lo + hi) / 2;
      bool before = left ? src.v[m] > value : src.v[m] >= value;
      if (before)
        lo = m + 1;
      else
        hi = m;
    }
    uint64_t rank = (pos - (left ? base : mid)) + (lo - other_base);
    put<INDEXED, Value>(src, uint32_t(pos), dst, uint32_t(base + rank));
  }
}
struct Allocation {
  void* ptr = nullptr;
  bool owned = true;
  ~Allocation() {
    if (ptr && owned) cudaFree(ptr);
  }
  void allocate(size_t bytes) {
    if (bytes) CK(cudaMalloc(&ptr, bytes));
  }
  void borrow(void* arena, detail::ArenaRegion region) {
    if (ptr && owned)
      throw std::logic_error("cannot replace owned storage with arena storage");
    owned = false;
    ptr = region.bytes ? static_cast<char*>(arena) + region.offset : nullptr;
  }
};
template <class Value>
struct DeviceBufferT {
  Allocation values, indices;
  void allocate(uint64_t n, bool indexed) {
    values.allocate(uint64_t(sizeof(Value)) * n);
    if (indexed) indices.allocate(4ull * n);
  }
  ValueBuffer<Value> get() {
    return {static_cast<Value*>(values.ptr),
            static_cast<uint32_t*>(indices.ptr)};
  }
};
using KernelJob = KernelJobT<float>;
using DeviceBuffer = DeviceBufferT<float>;
}  // namespace

struct Workspace::Impl {
  uint32_t max_n, max_k, max_limit;
  bool indexed, run_resources, capture_resources;
  DeviceBuffer c[2], w, p, scratch, winner_scratch;
  Allocation hist, meta, counters, captured_hist;
  Decision* host = nullptr;
  uint32_t* host_counts = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  cudaEvent_t ready[6]{}, done[6]{}, winner_start = nullptr,
                                     winner_end = nullptr;
  cudaStream_t winner_stream = nullptr;
  uint32_t run_slot = 0;
  int pending_winner = -1;
  cudaStream_t stream = nullptr;
  Options options;
  Stats* stats = nullptr;
  Impl(uint32_t n, uint32_t k, uint32_t limit, bool idx, bool enable_runs,
       bool enable_capture)
      : max_n(n),
        max_k(k),
        max_limit(limit),
        indexed(idx),
        run_resources(enable_runs),
        capture_resources(enable_capture) {
    Options o;
    o.sort_limit = limit;
    validate(n, k, o);
    try {
      int dev = 0;
      cudaDeviceProp prop{};
      CK(cudaGetDevice(&dev));
      CK(cudaGetDeviceProperties(&prop, dev));
      if (prop.sharedMemPerBlock < 4096u * 4u * (idx ? 2u : 1u))
        throw std::runtime_error("insufficient small-sort shared memory");
      for (auto& b : c) b.allocate(n, idx);
      w.allocate(k, idx);
      p.allocate(std::max(k, limit), idx);
      scratch.allocate(std::max(k, limit), idx);
      hist.allocate(4097 * 4);
      meta.allocate(sizeof(Decision));
      counters.allocate(2 * 4);
      if (capture_resources) captured_hist.allocate(4096 * 4);
      CK(cudaMallocHost(reinterpret_cast<void**>(&host),
                        sizeof(Decision) + 2 * 4));
      host_counts = reinterpret_cast<uint32_t*>(host + 1);
      CK(cudaEventCreate(&start));
      CK(cudaEventCreate(&end));
      if (run_resources) {
        winner_scratch.allocate(k, idx);
        CK(cudaStreamCreateWithFlags(&winner_stream, cudaStreamNonBlocking));
        for (uint32_t j = 0; j < 6; ++j) {
          CK(cudaEventCreateWithFlags(&ready[j], cudaEventDisableTiming));
          CK(cudaEventCreateWithFlags(&done[j], cudaEventDisableTiming));
        }
        CK(cudaEventCreate(&winner_start));
        CK(cudaEventCreate(&winner_end));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void cleanup() {
    if (winner_stream) {
      cudaStreamSynchronize(winner_stream);
      cudaStreamDestroy(winner_stream);
    }
    for (uint32_t j = 0; j < 6; ++j) {
      if (ready[j]) cudaEventDestroy(ready[j]);
      if (done[j]) cudaEventDestroy(done[j]);
    }
    if (winner_start) cudaEventDestroy(winner_start);
    if (winner_end) cudaEventDestroy(winner_end);
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (host) cudaFreeHost(host);
  }
  ~Impl() { cleanup(); }
  uint32_t blocks(uint32_t n) const {
    return std::min(options.block_cap, (n + BLOCK - 1) / BLOCK);
  }
  template <class F>
  void stage(const char* label, F f) {
    if (options.instrument) CK(cudaEventRecord(start, stream));
    f();
    CK(cudaGetLastError());
    if (options.instrument) {
      CK(cudaEventRecord(end, stream));
      CK(cudaEventSynchronize(end));
      float ms;
      CK(cudaEventElapsedTime(&ms, start, end));
      stats->stages.push_back({label, ms});
    }
  }
  void begin(Options o, Stats* s) {
    options = o;
    stats = s;
    run_slot = 0;
    pending_winner = -1;
  }
  Buffer candidate(int j) { return c[j].get(); }
  Buffer winners() { return w.get(); }
  Buffer potential() { return p.get(); }
  uint32_t* scale_flag(View src, int pass) {
    return src.shift_reference && pass == 2
               ? static_cast<uint32_t*>(hist.ptr) + 4096
               : nullptr;
  }
  Decision histogram(View src, uint32_t n, int pass, uint32_t rank) {
    auto unsafe = scale_flag(src, pass);
    stage("histogram", [&] {
      CK(cudaMemsetAsync(hist.ptr, 0, (1u << digit_width(pass)) * 4, stream));
      if (unsafe) CK(cudaMemsetAsync(unsafe, 0, 4, stream));
      auto mode = options.primitives;
      dim3 shape(primitive_blocks(n, options.block_cap, mode,
                                  histogram_threads(mode, pass)));
      launch_histogram<false, float>(src, n, pass,
                                     static_cast<uint32_t*>(hist.ptr), nullptr,
                                     shape, mode, stream, unsafe);
    });
    stage("select_and_metadata", [&] {
      launch_selection<false, false, float>(
          static_cast<uint32_t*>(hist.ptr), rank,
          static_cast<Decision*>(meta.ptr), nullptr, pass, dim3(1),
          options.primitives, stream, nullptr, nullptr, unsafe);
      CK(cudaMemcpyAsync(host, meta.ptr, sizeof(Decision),
                         cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
    });
    return *host;
  }
  template <int BINS>
  void capture_kernel(View src, uint32_t n, int pass, Buffer dst,
                      uint32_t capacity, uint32_t q) {
    if (indexed)
      count_capture<BINS, true><<<blocks(n), BLOCK, 0, stream>>>(
          src, n, pass, static_cast<uint32_t*>(hist.ptr),
          static_cast<uint32_t*>(captured_hist.ptr), dst, capacity,
          static_cast<uint32_t*>(counters.ptr), q, scale_flag(src, pass));
    else
      count_capture<BINS, false><<<blocks(n), BLOCK, 0, stream>>>(
          src, n, pass, static_cast<uint32_t*>(hist.ptr),
          static_cast<uint32_t*>(captured_hist.ptr), dst, capacity,
          static_cast<uint32_t*>(counters.ptr), q, scale_flag(src, pass));
  }
  Decision capture_histogram(View src, uint32_t n, int pass, uint32_t rank,
                             Buffer dst, uint32_t capacity, uint32_t q) {
    if (src.v == dst.v)
      throw std::logic_error("capture source aliases destination");
    if (options.debug && capacity < max_n) {
      CK(cudaMemsetAsync(dst.v + capacity, 0xa5, 4, stream));
      if (indexed) CK(cudaMemsetAsync(dst.i + capacity, 0xa5, 4, stream));
    }
    stage("histogram_capture", [&] {
      size_t bytes = (1u << digit_width(pass)) * 4;
      CK(cudaMemsetAsync(hist.ptr, 0, bytes, stream));
      CK(cudaMemsetAsync(captured_hist.ptr, 0, bytes, stream));
      if (auto unsafe = scale_flag(src, pass))
        CK(cudaMemsetAsync(unsafe, 0, 4, stream));
      CK(cudaMemsetAsync(counters.ptr, 0, 8, stream));
      if (digit_width(pass) == 8)
        capture_kernel<256>(src, n, pass, dst, capacity, q);
      else
        capture_kernel<4096>(src, n, pass, dst, capacity, q);
    });
    stage("select_capture_and_metadata", [&] {
      launch_selection<false, true, float>(
          static_cast<uint32_t*>(hist.ptr), rank,
          static_cast<Decision*>(meta.ptr), nullptr, pass, dim3(1),
          options.primitives, stream, static_cast<uint32_t*>(captured_hist.ptr),
          static_cast<uint32_t*>(counters.ptr), scale_flag(src, pass));
      CK(cudaMemcpyAsync(host, meta.ptr, sizeof(Decision),
                         cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
    });
    if (options.debug && capacity < max_n) {
      uint32_t guard = 0, index_guard = 0xa5a5a5a5;
      CK(cudaMemcpyAsync(&guard, dst.v + capacity, 4, cudaMemcpyDeviceToHost,
                         stream));
      if (indexed)
        CK(cudaMemcpyAsync(&index_guard, dst.i + capacity, 4,
                           cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
      if (guard != 0xa5a5a5a5 || index_guard != 0xa5a5a5a5)
        throw std::logic_error("capture write exceeded capacity");
    }
    return *host;
  }
  template <bool FILTER>
  void launch_gather(View src, uint32_t n, int pass, uint32_t bin, Buffer G,
                     uint32_t gc, Buffer E, uint32_t ec) {
    CK(cudaMemsetAsync(counters.ptr, 0, 8, stream));
    // Pivot-only reference candidates do not need indices: its final input
    // filter derives them. Winner-preserving paths must carry them.
    dim3 shape(primitive_blocks(n, options.block_cap, options.primitives));
    if (indexed && (FILTER || options.method != Method::reference))
      launch_partition<true, FILTER, false, float>(
          src, n, pass, bin, G, gc, E, ec, static_cast<uint32_t*>(counters.ptr),
          nullptr, shape, gc, options.primitives, stream);
    else
      launch_partition<false, FILTER, false, float>(
          src, n, pass, bin, G, gc, E, ec, static_cast<uint32_t*>(counters.ptr),
          nullptr, shape, gc, options.primitives, stream);
  }
  void check_counts(uint32_t g, uint32_t e) {
    if (!options.debug) return;
    CK(cudaMemcpyAsync(host_counts, counters.ptr, 8, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    if (host_counts[0] != g || host_counts[1] != e)
      throw std::logic_error("GPU gather count invariant");
  }
  void partition(View src, uint32_t n, int pass, int bin, Buffer G, uint32_t gc,
                 Buffer E, uint32_t ec, uint32_t eg, uint32_t ee) {
    stage("partition", [&] {
      launch_gather<false>(src, n, pass, uint32_t(bin), G, gc, E, ec);
    });
    check_counts(G.v ? eg : 0, ee);
  }
  void filter(View src, uint32_t n, uint32_t pivot, uint32_t greater,
              uint32_t ties, Buffer dst) {
    stage("original_filter", [&] {
      launch_gather<true>(src, n, 0, pivot, dst, greater, offset(dst, greater),
                          ties);
    });
    // Equal count can exceed the number required. Check G and E's lower bound.
    if (options.debug) {
      CK(cudaMemcpyAsync(host_counts, counters.ptr, 8, cudaMemcpyDeviceToHost,
                         stream));
      CK(cudaStreamSynchronize(stream));
      if (host_counts[0] != greater || host_counts[1] < ties)
        throw std::logic_error("GPU filter invariant");
    }
  }
  void launch_copy(View src, uint32_t n, Buffer dst, cudaStream_t target) {
    if (indexed)
      copy_items<true><<<blocks(n), BLOCK, 0, target>>>(src, n, dst);
    else
      copy_items<false><<<blocks(n), BLOCK, 0, target>>>(src, n, dst);
  }
  void copy(View src, uint32_t n, Buffer dst) {
    if (!n) return;
    stage("copy", [&] { launch_copy(src, n, dst, stream); });
  }
  uint32_t first_key(View src) {
    float value;
    stage("pivot_metadata", [&] {
      CK(cudaMemcpyAsync(&value, src.v, 4, cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
    });
    return ordered_key(value);
  }
  Buffer sort_buffer(Buffer src, uint32_t n, Buffer dst, cudaStream_t target) {
    uint32_t tile = std::min(4096u, padded(n));
    uint32_t grid = (n + tile - 1) / tile;
    size_t smem = tile * 4 * (indexed ? 2 : 1);
    if (indexed)
      sort_tiles<true><<<grid, BLOCK, smem, target>>>(src, n, tile);
    else
      sort_tiles<false><<<grid, BLOCK, smem, target>>>(src, n, tile);
    CK(cudaGetLastError());
    for (uint64_t width = tile; width < n; width *= 2) {
      if (indexed)
        merge_runs<true, false, float>
            <<<blocks(n), BLOCK, 0, target>>>(src, dst, n, uint32_t(width));
      else
        merge_runs<false, false, float>
            <<<blocks(n), BLOCK, 0, target>>>(src, dst, n, uint32_t(width));
      CK(cudaGetLastError());
      std::swap(src, dst);
    }
    return src;
  }
  void sort(Buffer src, uint32_t n, Buffer out, uint32_t k) {
    stage("sort", [&] { src = sort_buffer(src, n, scratch.get(), stream); });
    copy(src, k, out);
  }
  void sort_winner_run(uint32_t base, uint32_t n) {
    if (n < 2)
      return;  // one occurrence is already sorted; its write stays on the main
               // stream
    if (!run_resources || run_slot >= 6)
      throw std::logic_error("winner-run workspace capacity");
    cudaStream_t target =
        options.winner_sort == WinnerSort::overlap ? winner_stream : stream;
    uint32_t slot = run_slot++;
    if (options.winner_sort == WinnerSort::overlap) {
      // G/E are produced together. Only after this dependency is satisfied
      // may sorting read G; future E work has no dependency on the sort.
      CK(cudaEventRecord(ready[slot], stream));
      CK(cudaStreamWaitEvent(target, ready[slot], 0));
    }
    if (options.instrument) CK(cudaEventRecord(winner_start, target));
    Buffer region = offset(winners(), base);
    Buffer sorted = sort_buffer(region, n, winner_scratch.get(), target);
    if (sorted.v != region.v) launch_copy(sorted, n, region, target);
    CK(cudaGetLastError());
    if (options.instrument) {
      CK(cudaEventRecord(winner_end, target));
      CK(cudaEventSynchronize(winner_end));
      float ms;
      CK(cudaEventElapsedTime(&ms, winner_start, winner_end));
      stats->stages.push_back({"winner_run_sort", ms});
    }
    if (options.winner_sort == WinnerSort::overlap) {
      CK(cudaEventRecord(done[slot], target));
      pending_winner = int(slot);
    }
  }
  void join_winners() {
    if (pending_winner >= 0)
      stage("winner_join",
            [&] { CK(cudaStreamWaitEvent(stream, done[pending_winner], 0)); });
  }
  std::vector<float> values(View src, uint32_t n) {
    std::vector<float> x(n);
    if (n)
      CK(cudaMemcpyAsync(x.data(), src.v, 4ull * n, cudaMemcpyDeviceToHost,
                         stream));
    CK(cudaStreamSynchronize(stream));
    return x;
  }
  void audit(View original, uint32_t n, View C, uint32_t count, uint32_t wn,
             uint32_t r, uint32_t k, uint32_t better, Method method) {
    auto top = [](std::vector<float> x, uint32_t length) {
      if (length < x.size())
        std::nth_element(x.begin(), x.begin() + length, x.end(),
                         std::greater<float>());
      x.resize(length);
      std::sort(x.begin(), x.end(), std::greater<float>());
      return x;
    };
    if (!r || r > count) throw std::logic_error("debug residual bounds");
    auto expected = top(values(original, n), k),
         remaining = top(values(C, count), r);
    if (method == Method::reference) {
      if (better + r != k || !std::equal(remaining.begin(), remaining.end(),
                                         expected.begin() + better))
        throw std::logic_error("debug reference boundary invariant");
    } else {
      auto saved = values(winners(), wn);
      if (options.winner_sort != WinnerSort::deferred &&
          !std::is_sorted(saved.begin(), saved.end(), std::greater<float>()))
        throw std::logic_error("debug sorted winner-prefix invariant");
      remaining.insert(remaining.end(), saved.begin(), saved.end());
      std::sort(remaining.begin(), remaining.end(), std::greater<float>());
      if (wn + r != k || remaining != expected)
        throw std::logic_error("debug W + top-r(C) invariant");
    }
  }
};
Workspace::Workspace(uint32_t n, uint32_t k, uint32_t limit, bool idx,
                     bool enable_runs, bool enable_capture)
    : impl_(new Impl(n, k, limit, idx, enable_runs, enable_capture)) {}
Workspace::~Workspace() = default;
uint64_t Workspace::device_bytes() const {
  auto& b = *impl_;
  uint64_t factor = b.indexed ? 8 : 4;
  return factor *
             (2ull * b.max_n + b.max_k + 2ull * std::max(b.max_k, b.max_limit) +
              (b.run_resources ? b.max_k : 0)) +
         4096 * 4 * (b.capture_resources ? 2 : 1) + 4 + sizeof(Decision) + 8;
}
uint64_t Workspace::pinned_bytes() const { return sizeof(Decision) + 8; }
Stats topk(const float* input, uint32_t n, uint32_t k, float* output,
           uint32_t* indices, Workspace& w, Options o, cudaStream_t stream) {
  validate(n, k, o);
  auto& b = *w.impl_;
  if (!input || !output || input == output ||
      b.indexed != (indices != nullptr) || n > b.max_n || k > b.max_k ||
      o.sort_limit > b.max_limit ||
      (o.winner_sort != WinnerSort::deferred && !b.run_resources) ||
      (o.capture_rank && !b.capture_resources))
    throw std::invalid_argument(
        "pointer, indexed mode, or workspace capacity mismatch");
  b.stream = stream;
  return run(b, {input, nullptr}, n, k, {output, indices}, o);
}

template <class Value>
struct BatchBackend {
  using View = ValueView<Value>;
  using Buffer = ValueBuffer<Value>;
  using KernelJob = KernelJobT<Value>;
  using DeviceBuffer = DeviceBufferT<Value>;
  using HistJob = HistJobT<Value>;
  using CopyJob = CopyJobT<Value>;
  using PartitionJob = PartitionJobT<Value>;
  using FilterJob = FilterJobT<Value>;
  using SortJob = SortJobT<Value>;

  // Each upload has a distinct slot until stream completion. Reusing one
  // pinned descriptor array between asynchronous copies would race the DMA.
  // At most six rounds, each with <=19 merge levels (INT_MAX / 4096),
  // on each of the main and winner streams, plus uploads/copies/filters.
  // Distinct slots keep both pinned DMA sources and device descriptors live
  // until the caller's main-stream completion (which joins the winner stream).
  static constexpr uint32_t MAX_SLOTS = 512;
  uint32_t max_batch, max_n, max_k, max_limit, slot = 0, descriptor_slots = 0;
  bool indexed, run_resources, borrowed = false;
  detail::BatchArenaLayout arena_layout{};
  DeviceBuffer c[2], w, p, scratch, winner_scratch;
  Allocation histogram_data, metadata, counter_data, job_data;
  KernelJob* host_jobs = nullptr;
  Decision* host_meta = nullptr;
  uint32_t* host_counts = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  cudaEvent_t ready[6]{}, done[6]{};
  cudaStream_t winner_stream = nullptr;
  uint32_t run_slot = 0;
  int pending_winner = -1;
  cudaStream_t stream = nullptr;
  Options options;
  BatchStats* stats = nullptr;
  BatchBackend(uint32_t batch, uint32_t n, uint32_t k, uint32_t limit, bool idx,
               bool runs, void* arena = nullptr, uint64_t arena_bytes = 0)
      : max_batch(batch),
        max_n(n),
        max_k(k),
        max_limit(limit),
        indexed(idx),
        run_resources(runs),
        borrowed(arena != nullptr) {
    if (!batch || batch > 65535)
      throw std::invalid_argument("batch capacity must be in [1,65535]");
    Options o;
    o.sort_limit = limit;
    validate(n, k, o);
    // Conservative per-round bound: up to three sorts (tile + merges +
    // copy), two entry copies, histogram, partition, filter and W copy.
    // One final prefix copy. Usually 128 slots; 512 only at extreme k.
    descriptor_slots = detail::batch_descriptor_slots(k, limit);
    if (descriptor_slots > MAX_SLOTS)
      throw std::logic_error("batch descriptor bound");
    if (arena_bytes && !arena)
      throw std::invalid_argument("null borrowed arena");
    if (borrowed) {
      arena_layout =
          detail::batch_arena_layout<Value>(batch, n, k, limit, idx, runs);
      bind_arena(arena, arena_bytes);
    }
    try {
      int device;
      cudaDeviceProp prop{};
      CK(cudaGetDevice(&device));
      CK(cudaGetDeviceProperties(&prop, device));
      if (prop.sharedMemPerBlock < 4096u * 4u * (idx ? 2u : 1u))
        throw std::runtime_error("insufficient batch sort shared memory");
      if (!borrowed) {
        for (auto& b : c) b.allocate(uint64_t(batch) * n, idx);
        w.allocate(uint64_t(batch) * k, idx);
        p.allocate(uint64_t(batch) * std::max(k, limit), idx);
        scratch.allocate(uint64_t(batch) * std::max(k, limit), idx);
        histogram_data.allocate(uint64_t(batch) * 4097 * 4);
        metadata.allocate(uint64_t(batch) * sizeof(Decision));
        counter_data.allocate(uint64_t(batch) * 8);
        job_data.allocate(uint64_t(descriptor_slots) * batch *
                          sizeof(KernelJob));
        if (run_resources) winner_scratch.allocate(uint64_t(batch) * k, idx);
      }
      CK(cudaMallocHost(
          reinterpret_cast<void**>(&host_jobs),
          uint64_t(descriptor_slots) * batch * sizeof(KernelJob)));
      for (uint64_t j = 0; j < uint64_t(descriptor_slots) * batch; ++j)
        new (host_jobs + j) KernelJob{};
      CK(cudaMallocHost(reinterpret_cast<void**>(&host_meta),
                        uint64_t(batch) * sizeof(Decision)));
      for (uint32_t j = 0; j < batch; ++j) new (host_meta + j) Decision{};
      CK(cudaMallocHost(reinterpret_cast<void**>(&host_counts),
                        uint64_t(batch) * 8));
      CK(cudaEventCreate(&start));
      CK(cudaEventCreate(&end));
      if (run_resources) {
        CK(cudaStreamCreateWithFlags(&winner_stream, cudaStreamNonBlocking));
        for (int j = 0; j < 6; ++j) {
          CK(cudaEventCreateWithFlags(ready + j, cudaEventDisableTiming));
          CK(cudaEventCreateWithFlags(done + j, cudaEventDisableTiming));
        }
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void cleanup() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    for (int j = 0; j < 6; ++j) {
      if (ready[j]) cudaEventDestroy(ready[j]);
      if (done[j]) cudaEventDestroy(done[j]);
    }
    if (winner_stream) cudaStreamDestroy(winner_stream);
    if (host_jobs) cudaFreeHost(host_jobs);
    if (host_meta) cudaFreeHost(host_meta);
    if (host_counts) cudaFreeHost(host_counts);
  }
  ~BatchBackend() { cleanup(); }
  void bind_arena(void* arena, uint64_t bytes) {
    if (!borrowed)
      throw std::logic_error("workspace does not borrow device storage");
    detail::validate_arena(arena, bytes, arena_layout);
    DeviceBuffer* buffers[] = {&c[0], &c[1], &w, &p, &scratch, &winner_scratch};
    for (size_t j = 0; j < 6; ++j) {
      buffers[j]->values.borrow(arena, arena_layout.values[j]);
      buffers[j]->indices.borrow(arena, arena_layout.indices[j]);
    }
    histogram_data.borrow(arena, arena_layout.histogram);
    metadata.borrow(arena, arena_layout.metadata);
    counter_data.borrow(arena, arena_layout.counters);
    job_data.borrow(arena, arena_layout.jobs);
  }
  void begin(Options o, BatchStats* s) {
    options = o;
    stats = s;
    slot = 0;
    run_slot = 0;
    pending_winner = -1;
  }
  Buffer candidate(uint32_t t, int j) {
    return offset(c[j].get(), uint64_t(t) * max_n);
  }
  Buffer winners(uint32_t t) { return offset(w.get(), uint64_t(t) * max_k); }
  Buffer potential(uint32_t t) {
    return offset(p.get(), uint64_t(t) * std::max(max_k, max_limit));
  }
  Buffer temporary(uint32_t t) {
    return offset(scratch.get(), uint64_t(t) * std::max(max_k, max_limit));
  }
  dim3 grid(uint32_t n, size_t jobs, bool packed = false, int threads = BLOCK) {
    uint32_t cap = std::max(1u, options.block_cap / uint32_t(jobs));
    return dim3(primitive_blocks(
                    n, cap, packed ? options.primitives : PrimitiveMode::scalar,
                    threads),
                uint32_t(jobs));
  }
  const KernelJob* upload(const std::vector<KernelJob>& jobs,
                          cudaStream_t target) {
    if (jobs.empty() || jobs.size() > max_batch || slot >= descriptor_slots)
      throw std::logic_error("batch descriptor capacity");
    uint64_t base = uint64_t(slot++) * max_batch;
    std::copy(jobs.begin(), jobs.end(), host_jobs + base);
    auto dst = static_cast<KernelJob*>(job_data.ptr) + base;
    CK(cudaMemcpyAsync(dst, host_jobs + base, jobs.size() * sizeof(KernelJob),
                       cudaMemcpyHostToDevice, target));
    return dst;
  }
  const KernelJob* upload(const std::vector<KernelJob>& jobs) {
    return upload(jobs, stream);
  }
  template <class F>
  void stage_on(const char* label, cudaStream_t target, F fn) {
    if (options.instrument) CK(cudaEventRecord(start, target));
    fn();
    CK(cudaGetLastError());
    if (options.instrument) {
      CK(cudaEventRecord(end, target));
      CK(cudaEventSynchronize(end));
      float ms;
      CK(cudaEventElapsedTime(&ms, start, end));
      stats->stages.push_back({label, ms});
    }
  }
  template <class F>
  void stage(const char* label, F fn) {
    stage_on(label, stream, fn);
  }
  std::vector<Decision> histograms(const std::vector<HistJob>& requests) {
    std::vector<KernelJob> jobs;
    uint32_t largest = 0;
    int pass = requests.front().pass;
    for (auto& r : requests) {
      if (r.pass != pass) throw std::logic_error("mixed batch digits");
      KernelJob j;
      j.src = r.src;
      j.n = r.n;
      j.pass = r.pass;
      j.rank = r.rank;
      j.hist =
          static_cast<uint32_t*>(histogram_data.ptr) + uint64_t(r.task) * 4097;
      // The extra histogram word carries either the entry finite-input
      // check or the final shifted-digit round-trip check.
      if ((options.reject_nonfinite && r.pass == 0) ||
          (r.src.shift_reference && r.pass == 2))
        j.scale_unsafe = j.hist + 4096;
      j.decision = static_cast<Decision*>(metadata.ptr) + r.task;
      jobs.push_back(j);
      largest = std::max(largest, r.n);
    }
    const KernelJob* device_jobs = nullptr;
    stage("batch_histogram", [&] {
      device_jobs = upload(jobs);
      CK(cudaMemsetAsync(histogram_data.ptr, 0, uint64_t(max_batch) * 4097 * 4,
                         stream));
      launch_histogram<true, Value>(
          {}, 0, pass, nullptr, device_jobs,
          grid(largest, jobs.size(), true,
               histogram_threads(options.primitives, pass)),
          options.primitives, stream);
    });
    stage("batch_select_and_metadata", [&] {
      dim3 shape(1, uint32_t(jobs.size()));
      launch_selection<true, false, Value>(nullptr, 0, nullptr, device_jobs,
                                           pass, shape, options.primitives,
                                           stream);
      CK(cudaMemcpyAsync(host_meta, metadata.ptr,
                         uint64_t(max_batch) * sizeof(Decision),
                         cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
    });
    std::vector<Decision> result;
    for (auto& r : requests) {
      if (options.reject_nonfinite && pass == 0 &&
          host_meta[r.task].scale_unsafe)
        throw NonFiniteInput{};
      result.push_back(host_meta[r.task]);
    }
    return result;
  }
  void copies_on(const std::vector<CopyJob>& requests, cudaStream_t target,
                 const char* label) {
    if (requests.empty()) return;
    std::vector<KernelJob> jobs;
    uint32_t largest = 0;
    for (auto& r : requests) {
      if (!r.n) continue;
      KernelJob j;
      j.src = r.src;
      j.n = r.n;
      j.G = r.dst;
      jobs.push_back(j);
      largest = std::max(largest, r.n);
    }
    if (jobs.empty()) return;
    stage_on(label, target, [&] {
      auto device_jobs = upload(jobs, target);
      if (indexed)
        copy_items<true, true, Value>
            <<<grid(largest, jobs.size()), BLOCK, 0, target>>>({}, 0, {},
                                                               device_jobs);
      else
        copy_items<false, true, Value>
            <<<grid(largest, jobs.size()), BLOCK, 0, target>>>({}, 0, {},
                                                               device_jobs);
    });
  }
  void copies(const std::vector<CopyJob>& requests) {
    copies_on(requests, stream, "batch_copy");
  }
  template <bool FILTER>
  void launch_gathers(const std::vector<KernelJob>& jobs, uint32_t largest) {
    auto device_jobs = upload(jobs);
    CK(cudaMemsetAsync(counter_data.ptr, 0, uint64_t(max_batch) * 8, stream));
    uint32_t max_gcap = 0;
    for (auto& job : jobs) max_gcap = std::max(max_gcap, job.gcap);
    if (indexed && (FILTER || options.method != Method::reference))
      launch_partition<true, FILTER, true, Value>(
          {}, 0, 0, 0, {}, 0, {}, 0, nullptr, device_jobs,
          grid(largest, jobs.size(), true), max_gcap, options.primitives,
          stream);
    else
      launch_partition<false, FILTER, true, Value>(
          {}, 0, 0, 0, {}, 0, {}, 0, nullptr, device_jobs,
          grid(largest, jobs.size(), true), max_gcap, options.primitives,
          stream);
  }
  void read_counts() {
    CK(cudaMemcpyAsync(host_counts, counter_data.ptr, uint64_t(max_batch) * 8,
                       cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
  }
  void partitions(const std::vector<PartitionJob>& requests) {
    if (requests.empty()) return;
    std::vector<KernelJob> jobs;
    uint32_t largest = 0;
    for (auto& r : requests) {
      KernelJob j;
      j.src = r.src;
      j.n = r.n;
      j.pass = r.pass;
      j.boundary = r.bin;
      j.G = r.G;
      j.gcap = r.gcap;
      j.E = r.E;
      j.ecap = r.ecap;
      j.counters =
          static_cast<uint32_t*>(counter_data.ptr) + uint64_t(r.task) * 2;
      jobs.push_back(j);
      largest = std::max(largest, r.n);
    }
    stage("batch_partition", [&] { launch_gathers<false>(jobs, largest); });
    if (options.debug) {
      read_counts();
      for (auto& r : requests)
        if (host_counts[2 * r.task] != (r.G.v ? r.expected_g : 0) ||
            host_counts[2 * r.task + 1] != r.expected_e)
          throw std::logic_error("batch partition counts");
    }
  }
  void filters(const std::vector<FilterJob>& requests) {
    if (requests.empty()) return;
    std::vector<KernelJob> jobs;
    uint32_t largest = 0;
    for (auto& r : requests) {
      KernelJob j;
      j.src = r.src;
      j.n = r.n;
      j.boundary = r.pivot;
      j.pivot = r.pivot_value;
      j.G = r.dst;
      j.gcap = r.greater;
      j.E = offset(r.dst, r.greater);
      j.ecap = r.ties;
      j.counters =
          static_cast<uint32_t*>(counter_data.ptr) + uint64_t(r.task) * 2;
      jobs.push_back(j);
      largest = std::max(largest, r.n);
    }
    stage("batch_original_filter",
          [&] { launch_gathers<true>(jobs, largest); });
    if (options.debug) {
      read_counts();
      for (auto& r : requests)
        if (host_counts[2 * r.task] != r.greater ||
            host_counts[2 * r.task + 1] < r.ties)
          throw std::logic_error("batch filter counts");
    }
  }
  void sorts_on(const std::vector<SortJob>& requests, cudaStream_t target,
                bool winner) {
    if (requests.empty()) return;
    auto active = requests;
    stage_on(winner ? "batch_winner_run_sort" : "batch_sort", target, [&] {
      std::vector<KernelJob> jobs;
      uint32_t tiles = 0, max_tile = 0, max_length = 0;
      for (auto& r : active) {
        KernelJob j;
        j.G = r.src;
        j.n = r.n;
        j.tile = std::min(4096u, padded(r.n));
        tiles = std::max(tiles, (r.n + j.tile - 1) / j.tile);
        max_tile = std::max(max_tile, j.tile);
        max_length = std::max(max_length, r.n);
        jobs.push_back(j);
      }
      auto device_jobs = upload(jobs, target);
      dim3 shape(tiles, uint32_t(jobs.size()));
      size_t smem = max_tile * 4 * (indexed ? 2 : 1);
      if (indexed)
        sort_tiles<true, true, Value>
            <<<shape, BLOCK, smem, target>>>({}, 0, 0, device_jobs);
      else
        sort_tiles<false, true, Value>
            <<<shape, BLOCK, smem, target>>>({}, 0, 0, device_jobs);
      CK(cudaGetLastError());
      std::vector<Buffer> destinations;
      for (auto& r : active)
        destinations.push_back(
            winner ? offset(winner_scratch.get(), uint64_t(r.task) * max_k)
                   : temporary(r.task));
      for (uint64_t width = 4096; width < max_length; width *= 2) {
        jobs.clear();
        uint32_t largest = 0;
        for (size_t t = 0; t < active.size(); ++t) {
          auto& r = active[t];
          if (r.n <= width) continue;
          KernelJob j;
          j.src = r.src;
          j.G = destinations[t];
          j.n = r.n;
          j.width = uint32_t(width);
          jobs.push_back(j);
          largest = std::max(largest, r.n);
          std::swap(r.src, destinations[t]);
        }
        device_jobs = upload(jobs, target);
        if (indexed)
          merge_runs<true, true, Value>
              <<<grid(largest, jobs.size()), BLOCK, 0, target>>>({}, {}, 0, 0,
                                                                 device_jobs);
        else
          merge_runs<false, true, Value>
              <<<grid(largest, jobs.size()), BLOCK, 0, target>>>({}, {}, 0, 0,
                                                                 device_jobs);
        CK(cudaGetLastError());
      }
    });
    std::vector<CopyJob> output;
    for (auto& r : active)
      if (r.src.v != r.out.v || r.src.i != r.out.i)
        output.push_back({r.task, r.src, r.k, r.out});
    copies_on(output, target, winner ? "batch_winner_copy" : "batch_copy");
  }
  void sorts(const std::vector<SortJob>& requests) {
    sorts_on(requests, stream, false);
  }
  void winner_sorts(const std::vector<WinnerJob>& requests) {
    std::vector<SortJob> jobs;
    for (auto& r : requests)
      if (r.n > 1) {
        auto region = offset(winners(r.task), r.base);
        jobs.push_back({r.task, region, r.n, region, r.n});
      }
    if (jobs.empty()) return;
    if (!run_resources || run_slot >= 6)
      throw std::logic_error("batch winner-run workspace capacity");
    uint32_t j = run_slot++;
    cudaStream_t target =
        options.winner_sort == WinnerSort::overlap ? winner_stream : stream;
    if (options.winner_sort == WinnerSort::overlap) {
      CK(cudaEventRecord(ready[j], stream));
      CK(cudaStreamWaitEvent(target, ready[j], 0));
    }
    sorts_on(jobs, target, true);
    if (options.winner_sort == WinnerSort::overlap) {
      CK(cudaEventRecord(done[j], target));
      pending_winner = int(j);
    }
  }
  void join_winners() {
    if (pending_winner >= 0)
      stage("batch_winner_join",
            [&] { CK(cudaStreamWaitEvent(stream, done[pending_winner], 0)); });
  }
  std::vector<float> values(View src, uint32_t n) {
    std::vector<Value> raw(n);
    if (n)
      CK(cudaMemcpyAsync(raw.data(), src.v, uint64_t(sizeof(Value)) * n,
                         cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    return std::vector<float>(raw.begin(), raw.end());
  }
  void audit(uint32_t t, View original, uint32_t n, View C, uint32_t count,
             uint32_t wn, uint32_t r, uint32_t k, uint32_t better,
             Method method) {
    auto top = [](std::vector<float> x, uint32_t length) {
      if (length < x.size())
        std::nth_element(x.begin(), x.begin() + length, x.end(),
                         std::greater<float>());
      x.resize(length);
      std::sort(x.begin(), x.end(), std::greater<float>());
      return x;
    };
    auto expected = top(values(original, n), k),
         remaining = top(values(C, count), r);
    if (method == Method::reference) {
      if (better + r != k || !std::equal(remaining.begin(), remaining.end(),
                                         expected.begin() + better))
        throw std::logic_error("batch reference audit");
    } else {
      auto saved = values(winners(t), wn);
      remaining.insert(remaining.end(), saved.begin(), saved.end());
      std::sort(remaining.begin(), remaining.end(), std::greater<float>());
      if (wn + r != k || remaining != expected)
        throw std::logic_error("batch winner audit");
    }
  }
};
struct BatchWorkspace::Impl : BatchBackend<float> {
  using BatchBackend<float>::BatchBackend;
};
struct BFloat16BatchWorkspace::Impl : BatchBackend<BFloat16> {
  using BatchBackend<BFloat16>::BatchBackend;
};
template <class Value>
uint64_t batch_device_bytes(const BatchBackend<Value>& b) {
  if (b.borrowed) return b.arena_layout.bytes;
  return (sizeof(Value) + (b.indexed ? 4ull : 0ull)) * b.max_batch *
             (2ull * b.max_n + b.max_k + 2ull * std::max(b.max_k, b.max_limit) +
              (b.run_resources ? b.max_k : 0)) +
         uint64_t(b.max_batch) *
             (4097 * 4 + sizeof(Decision) + 8 +
              b.descriptor_slots * sizeof(KernelJobT<Value>));
}
template <class Value>
uint64_t batch_pinned_bytes(const BatchBackend<Value>& b) {
  return uint64_t(b.max_batch) *
         (sizeof(Decision) + 8 +
          b.descriptor_slots * sizeof(KernelJobT<Value>));
}
template <class Value>
BatchStats topk_batch_impl(BatchBackend<Value>& b, const Value* input,
                           const uint32_t* lengths, uint32_t batch, uint32_t k,
                           Value* output, uint32_t* indices, Options options,
                           cudaStream_t stream) {
  if (!input || !output || input == output || !lengths || !batch ||
      batch > b.max_batch || k > b.max_k || options.sort_limit > b.max_limit ||
      b.indexed != (indices != nullptr) ||
      (options.winner_sort != WinnerSort::deferred && !b.run_resources))
    throw std::invalid_argument("batch pointer or workspace mismatch");
  if (options.reject_nonfinite &&
      (options.entry_sort || options.scale != ScaleMode::off))
    throw std::invalid_argument(
        "finite input guard requires the unshifted entry histogram");
  std::vector<uint32_t> shape(lengths, lengths + batch);
  batch_offsets(shape, k, options);
  for (auto n : shape)
    if (n > b.max_n) throw std::invalid_argument("batch row exceeds workspace");
  b.stream = stream;
  return run_batch(b, ValueView<Value>{input, nullptr}, shape, k,
                   ValueBuffer<Value>{output, indices}, options);
}
BatchWorkspace::BatchWorkspace(uint32_t b, uint32_t n, uint32_t k,
                               uint32_t limit, bool indexed, bool runs)
    : impl_(new Impl(b, n, k, limit, indexed, runs)) {}
BatchWorkspace::~BatchWorkspace() = default;
uint64_t BatchWorkspace::device_bytes() const {
  return batch_device_bytes<float>(*impl_);
}
uint64_t BatchWorkspace::pinned_bytes() const {
  return batch_pinned_bytes<float>(*impl_);
}
BatchStats topk_batch(const float* input, const uint32_t* lengths,
                      uint32_t batch, uint32_t k, float* output,
                      uint32_t* indices, BatchWorkspace& workspace,
                      Options options, cudaStream_t stream) {
  return topk_batch_impl<float>(*workspace.impl_, input, lengths, batch, k,
                                output, indices, options, stream);
}
BFloat16BatchWorkspace::BFloat16BatchWorkspace(uint32_t b, uint32_t n,
                                               uint32_t k, uint32_t limit,
                                               bool indexed, bool runs)
    : impl_(new Impl(b, n, k, limit, indexed, runs)) {}
BFloat16BatchWorkspace::BFloat16BatchWorkspace(uint32_t b, uint32_t n,
                                               uint32_t k, uint32_t limit,
                                               bool indexed, bool runs,
                                               void* storage, uint64_t bytes)
    : impl_(nullptr) {
  if (!storage) throw std::invalid_argument("null borrowed storage");
  impl_.reset(new Impl(b, n, k, limit, indexed, runs, storage, bytes));
}
uint64_t BFloat16BatchWorkspace::required_device_bytes(uint32_t b, uint32_t n,
                                                       uint32_t k,
                                                       uint32_t limit,
                                                       bool indexed,
                                                       bool runs) {
  return detail::batch_arena_layout<BFloat16>(b, n, k, limit, indexed, runs)
      .bytes;
}
void BFloat16BatchWorkspace::bind_device_storage(void* storage,
                                                 uint64_t bytes) {
  impl_->bind_arena(storage, bytes);
}
uint64_t BFloat16BatchWorkspace::owned_device_bytes() const {
  return impl_->borrowed ? 0 : device_bytes();
}
BFloat16BatchWorkspace::~BFloat16BatchWorkspace() = default;
uint64_t BFloat16BatchWorkspace::device_bytes() const {
  return batch_device_bytes<BFloat16>(*impl_);
}
uint64_t BFloat16BatchWorkspace::pinned_bytes() const {
  return batch_pinned_bytes<BFloat16>(*impl_);
}
BatchStats topk_batch_bf16(const BFloat16* input, const uint32_t* lengths,
                           uint32_t batch, uint32_t k, BFloat16* output,
                           uint32_t* indices, BFloat16BatchWorkspace& workspace,
                           Options options, cudaStream_t stream) {
  return topk_batch_impl<BFloat16>(*workspace.impl_, input, lengths, batch, k,
                                   output, indices, options, stream);
}
#include "bfloat16_graph.cuh"

}  // namespace r2sk
