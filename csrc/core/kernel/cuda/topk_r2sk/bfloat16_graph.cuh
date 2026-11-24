// SPDX-License-Identifier: Apache-2.0
// Included in the CUDA translation unit to reuse the packed radix primitives.

namespace {

using GraphJob = KernelJobT<BFloat16>;
using GraphBuffer = ValueBuffer<BFloat16>;
constexpr int kGraphPlanThreads = 128;
constexpr int kGraphFirstBins = 4096;
constexpr int kGraphLastBins = 16;
constexpr int kGraphHistogramStride = kGraphFirstBins + 1;
constexpr uint32_t kGraphMaxSort = 1024;

struct GraphSort {
  GraphBuffer source;
  GraphBuffer output;
  uint32_t count;
  uint32_t keep;
};

struct GraphStorage {
  GraphJob* first;
  GraphJob* last;
  Decision* decisions;
  uint32_t* histogram;
  uint32_t* counters;
  GraphSort* first_sort;
  GraphSort* last_sort;
  GraphBuffer candidate;
  GraphBuffer potential;
};

struct GraphLayout {
  detail::ArenaRegion first, last, decisions, histogram, counters;
  detail::ArenaRegion first_sort, last_sort, candidate_values,
      candidate_indices;
  detail::ArenaRegion potential_values, potential_indices;
  uint64_t bytes;
};

GraphLayout graph_layout(BFloat16GraphShape s) {
  if (!s.batch || s.batch > 65535 || s.vocabulary < 2 ||
      s.vocabulary > 0x7fffffffu || !s.k || s.k > 256 || s.k > s.vocabulary ||
      s.sort_limit < s.k || s.sort_limit > kGraphMaxSort ||
      (s.sort_limit & (s.sort_limit - 1)) || !s.block_cap ||
      s.block_cap > 65535) {
    throw std::invalid_argument("invalid BF16 graph shape or launch limits");
  }

  const uint64_t batch = s.batch;
  detail::ArenaPlanner planner;
  return {
      planner.add(batch, sizeof(GraphJob)),
      planner.add(batch, sizeof(GraphJob)),
      planner.add(batch, sizeof(Decision)),
      planner.add(batch * kGraphHistogramStride, sizeof(uint32_t)),
      planner.add(batch * 2, sizeof(uint32_t)),
      planner.add(batch, sizeof(GraphSort)),
      planner.add(batch, sizeof(GraphSort)),
      planner.add(batch * s.vocabulary, sizeof(BFloat16)),
      planner.add(batch * s.vocabulary, sizeof(uint32_t)),
      planner.add(batch * s.sort_limit, sizeof(BFloat16)),
      planner.add(batch * s.sort_limit, sizeof(uint32_t)),
      planner.finish(),
  };
}

template <class T>
T* graph_region(void* arena, detail::ArenaRegion region) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(arena) +
                              region.offset);
}

__device__ GraphBuffer row_buffer(GraphBuffer buffer, uint64_t offset) {
  return {buffer.v + offset, buffer.i + offset};
}

__global__ void init_graph_jobs(BFloat16GraphShape shape, const BFloat16* input,
                                GraphStorage storage) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= shape.batch) return;
  GraphJob first{};
  first.src.v = input + uint64_t(row) * shape.vocabulary;
  first.n = shape.vocabulary;
  first.rank = shape.k;
  first.hist = storage.histogram + uint64_t(row) * kGraphHistogramStride;
  first.scale_unsafe = first.hist + kGraphFirstBins;
  first.decision = storage.decisions + row;
  first.counters = storage.counters + uint64_t(row) * 2;
  storage.first[row] = first;
}

__global__ void plan_graph_partition(BFloat16GraphShape shape,
                                     GraphStorage storage, GraphBuffer output,
                                     BFloat16GraphRow* rows) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= shape.batch) return;
  GraphJob first = storage.first[row];
  const Decision decision = storage.decisions[row];
  GraphJob last{};
  GraphSort sort{};
  GraphRoute route = GraphRoute::invalid_nan;

  if (decision.scale_unsafe) {
    first.n = 0;
  } else {
    const auto destination = row_buffer(output, uint64_t(row) * shape.k);
    const auto potential =
        row_buffer(storage.potential, uint64_t(row) * shape.sort_limit);
    const uint32_t retained = decision.h + decision.B;
    first.boundary = static_cast<uint32_t>(decision.bin);
    first.gcap = decision.h;
    first.ecap = decision.B;
    if (retained <= shape.sort_limit) {
      route = retained == shape.k ? GraphRoute::exact : GraphRoute::early_sort;
      first.G = potential;
      first.E = row_buffer(potential, decision.h);
      sort = {potential, destination, retained, shape.k};
    } else {
      route = GraphRoute::refine;
      const auto candidate =
          row_buffer(storage.candidate, uint64_t(row) * shape.vocabulary);
      first.G = destination;
      first.E = candidate;
      sort = {destination, destination, decision.h, decision.h};
      last.src = {candidate.v, candidate.i};
      last.n = decision.B;
      last.rank = shape.k - decision.h;
      last.pass = 1;
      // The first 12 bits select one sign. In the next original digit
      // the low eight bits are constant for every BF16 key in this group.
      last.boundary = first.boundary < 2048 ? 255 : 0;
      last.hist = first.hist;
      last.counters = first.counters;
      last.G = row_buffer(destination, decision.h);

      // An all-retained row can directly reuse the source and its indices.
      if (decision.B == first.n) {
        last.src = first.src;
        first.n = 0;
      }
      for (int bin = 0; bin < kGraphLastBins; ++bin) last.hist[bin] = 0;
    }
  }
  first.counters[0] = 0;
  first.counters[1] = 0;
  storage.first[row] = first;
  storage.last[row] = last;
  storage.first_sort[row] = sort;
  if (rows) {
    uint32_t passes = route == GraphRoute::refine ? 2 : 1;
    rows[row] = {route, decision.h, decision.B, passes};
  }
}

__global__ void plan_graph_tail(BFloat16GraphShape shape,
                                GraphStorage storage) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= shape.batch) return;
  GraphJob last = storage.last[row];
  GraphSort sort{};
  if (last.n) {
    uint32_t above = 0;
    uint32_t ties = 0;
    int bin = kGraphLastBins - 1;
    for (; bin >= 0; --bin) {
      ties = last.hist[bin];
      if (above + ties >= last.rank) break;
      above += ties;
    }
    const auto potential =
        row_buffer(storage.potential, uint64_t(row) * shape.sort_limit);
    sort = {potential, last.G, last.rank, last.rank};
    last.G = potential;
    last.E = row_buffer(potential, above);
    last.gcap = above;
    last.ecap = last.rank - above;
    last.boundary |= static_cast<uint32_t>(bin) << 8;
    // The top bin covers the whole input. Any rank occurrences are valid.
    if (ties == last.n) last.n = last.rank;
    last.counters[0] = 0;
    last.counters[1] = 0;
    storage.last[row] = last;
  }
  storage.last_sort[row] = sort;
}

__global__ void sort_graph_run(const GraphSort* jobs) {
  const GraphSort job = jobs[blockIdx.x];
  if (!job.count) return;
  if (job.count == 1) {
    if (threadIdx.x == 0 && job.source.v != job.output.v) {
      job.output.v[0] = job.source.v[0];
      job.output.i[0] = job.source.i[0];
    }
    return;
  }

  extern __shared__ uint32_t storage[];
  const uint32_t width = 1u << (32 - __clz(job.count - 1));
  uint32_t* keys = storage;
  uint32_t* indices = storage + width;
  for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const bool valid = i < job.count;
    keys[i] = valid ? ordered_key(job.source.v[i]) >> 16 : 0;
    indices[i] = valid ? job.source.i[i] : 0;
  }
  __syncthreads();
  for (uint32_t size = 2; size <= width; size <<= 1) {
    for (uint32_t stride = size >> 1; stride; stride >>= 1) {
      for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        const uint32_t other = i ^ stride;
        if (other > i) {
          const bool descending = (i & size) == 0;
          const uint32_t a = keys[i], b = keys[other];
          if (descending ? a < b : a > b) {
            keys[i] = b;
            keys[other] = a;
            const uint32_t index = indices[i];
            indices[i] = indices[other];
            indices[other] = index;
          }
        }
      }
      __syncthreads();
    }
  }
  for (uint32_t i = threadIdx.x; i < job.keep; i += blockDim.x) {
    const uint32_t key = keys[i];
    const uint32_t mask = key & 0x8000u ? 0x8000u : 0xffffu;
    job.output.v[i].bits = static_cast<uint16_t>(key ^ mask);
    job.output.i[i] = indices[i];
  }
}

}  // namespace

uint64_t bfloat16_graph_bytes(BFloat16GraphShape shape) {
  return graph_layout(shape).bytes;
}

void enqueue_bfloat16_graph(BFloat16GraphShape shape, const void* input,
                            void* values, uint32_t* indices, void* arena,
                            uint64_t bytes, cudaStream_t stream,
                            BFloat16GraphFork fork, BFloat16GraphRow* rows) {
  const auto layout = graph_layout(shape);
  const auto address = reinterpret_cast<uintptr_t>(arena);
  if (!arena || address % detail::ArenaPlanner::alignment ||
      bytes < layout.bytes || layout.bytes > UINTPTR_MAX - address || !input ||
      !values || !indices) {
    throw std::invalid_argument("invalid BF16 graph buffers");
  }
  if ((fork.stream && (!fork.ready || !fork.done || fork.ready == fork.done ||
                       fork.stream == stream)) ||
      (!fork.stream && (fork.ready || fork.done))) {
    throw std::invalid_argument("invalid BF16 graph fork");
  }
  const GraphStorage storage{
      graph_region<GraphJob>(arena, layout.first),
      graph_region<GraphJob>(arena, layout.last),
      graph_region<Decision>(arena, layout.decisions),
      graph_region<uint32_t>(arena, layout.histogram),
      graph_region<uint32_t>(arena, layout.counters),
      graph_region<GraphSort>(arena, layout.first_sort),
      graph_region<GraphSort>(arena, layout.last_sort),
      {graph_region<BFloat16>(arena, layout.candidate_values),
       graph_region<uint32_t>(arena, layout.candidate_indices)},
      {graph_region<BFloat16>(arena, layout.potential_values),
       graph_region<uint32_t>(arena, layout.potential_indices)},
  };
  const uint32_t plan_blocks =
      (shape.batch + kGraphPlanThreads - 1) / kGraphPlanThreads;
  const uint32_t row_cap = std::max(1u, shape.block_cap / shape.batch);
  const auto grid = [&](int threads) {
    return dim3(primitive_blocks(shape.vocabulary, row_cap,
                                 PrimitiveMode::radik, threads),
                shape.batch);
  };
  const uint32_t gather_bytes = gather_shared_bytes(shape.k, true, true);
  const uint32_t sort_bytes = shape.sort_limit * 2 * sizeof(uint32_t);
  const GraphBuffer output{static_cast<BFloat16*>(values), indices};

  CK(cudaMemsetAsync(storage.histogram, 0, layout.histogram.bytes, stream));
  init_graph_jobs<<<plan_blocks, kGraphPlanThreads, 0, stream>>>(
      shape, static_cast<const BFloat16*>(input), storage);
  count_packed<4096, 1024, true, BFloat16, true>
      <<<grid(1024), 1024, 0, stream>>>({}, 0, 0, nullptr, storage.first);
  select_bin_warp<4096, true, false, BFloat16>
      <<<dim3(1, shape.batch), 1024, 0, stream>>>(nullptr, 0, nullptr,
                                                  storage.first);
  plan_graph_partition<<<plan_blocks, kGraphPlanThreads, 0, stream>>>(
      shape, storage, output, rows);
  gather_packed<true, false, true, true, BFloat16, true>
      <<<grid(BLOCK), BLOCK, gather_bytes, stream>>>({}, 0, 0, 0, {}, 0, {}, 0,
                                                     nullptr, storage.first);

  const auto sort_stream = fork.stream ? fork.stream : stream;
  if (fork.stream) {
    CK(cudaEventRecord(fork.ready, stream));
    CK(cudaStreamWaitEvent(fork.stream, fork.ready, 0));
  }
  sort_graph_run<<<shape.batch, BLOCK, sort_bytes, sort_stream>>>(
      storage.first_sort);
  if (fork.stream) CK(cudaEventRecord(fork.done, fork.stream));

  count_packed<16, BLOCK, true, BFloat16, true>
      <<<grid(BLOCK), BLOCK, 0, stream>>>({}, 0, 1, nullptr, storage.last);
  plan_graph_tail<<<plan_blocks, kGraphPlanThreads, 0, stream>>>(shape,
                                                                 storage);
  gather_packed<true, false, true, true, BFloat16, true>
      <<<grid(BLOCK), BLOCK, gather_bytes, stream>>>({}, 0, 1, 0, {}, 0, {}, 0,
                                                     nullptr, storage.last);
  sort_graph_run<<<shape.batch, BLOCK, sort_bytes, stream>>>(storage.last_sort);

  // Pass-one sorts write a row's prefix. Final sorts write only its tail,
  // and rows that ended early never touch their potential buffer again.
  if (fork.stream) CK(cudaStreamWaitEvent(stream, fork.done, 0));
  CK(cudaGetLastError());
}
