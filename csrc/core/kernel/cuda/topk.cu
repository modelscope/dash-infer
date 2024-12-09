/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

namespace impl {
enum HeapType { kMinHeap, kMaxHeap };
enum PreferIndices { kLower, kHigher };

template <typename T>
struct Entry {
  int index;
  T value;

  __device__ Entry(int i, T v) : index(i), value(v) {}
  __device__ Entry(const Entry& rhs) : index(rhs.index), value(rhs.value) {}
  __device__ Entry& operator=(const Entry& rhs) {
    index = rhs.index;
    value = rhs.value;
    return *this;
  }

  static inline bool greater(const Entry<T>& a, const Entry<T>& b) {
    if (a.value == b.value) {
      return a.index < b.index;
    }
    return a.value > b.value;
  }
};

template <typename T>
struct LinearData {
  typedef impl::Entry<T> Entry;
  __device__ Entry& operator[](std::size_t index) const { return data[index]; }
  __device__ int get_index(int i) const { return data[i].index; }
  __device__ T get_value(int i) const { return data[i].value; }

  Entry* const data;

  __device__ LinearData(Entry* const InData) : data(InData) {}
  __device__ LinearData(const LinearData& InData) : data(InData.data) {}
};

template <typename T>
struct IndirectLinearData {
  typedef impl::Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const { return data[index]; }
  __device__ int get_index(int i) const {
    return backing_data[data[i].index].index;
  }
  __device__ T get_value(int i) const { return data[i].value; }

  Entry* const data;
  Entry* const backing_data;
  __device__ IndirectLinearData(Entry* const InData, Entry* const InData2)
      : data(InData), backing_data(InData2) {}
  __device__ IndirectLinearData(const IndirectLinearData& InData)
      : data(InData.data), backing_data(InData.backing_data) {}
};

template <typename T>
struct StridedData {
  typedef impl::Entry<T> Entry;

  Entry* const data;
  __device__ StridedData(Entry* const InData) : data(InData) {}
  __device__ Entry& operator[](std::size_t index) const {
    return data[index * blockDim.x + threadIdx.x];
  }

  __device__ int get_index(int i) const { return (*this)[i].index; }
  __device__ T get_value(int i) const { return (*this)[i].value; }
};

// A heap of Entry<T> that can either work as a min-heap or as a max-heap.
template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
class IndexedHeap {
 public:
  typedef typename Data<T>::Entry Entry;
  const Data<T> data;

  __device__ IndexedHeap(const Data<T>& InData) : data(InData) {}

  __device__ bool is_above(int left, int right) {
    T left_value = data.get_value(left);
    T right_value = data.get_value(right);
    if (left_value == right_value) {
      if (preferIndices == kLower) {
        return data.get_index(left) < data.get_index(right);
      } else {
        return data.get_index(left) > data.get_index(right);
      }
    }
    if (heapType == kMinHeap) {
      return left_value < right_value;
    } else {
      return left_value > right_value;
    }
  }
  __device__ void assign(int i, const Entry& entry) { data[i] = entry; }
  __device__ void push_up(int i) {
    int child = i;
    int parent;
    for (; child > 0; child = parent) {
      parent = (child - 1) / 2;
      if (!is_above(child, parent)) {
        break;
      }
      swap(child, parent);
    }
  }
  __device__ void swap(int a, int b) {
    Entry tmp = data[b];
    data[b] = data[a];
    data[a] = tmp;
  }

  __device__ void push_root_down(int k) { push_down(0, k); }
  __device__ void push_down(int node, int k) {
    while (true) {
      const int left = 2 * node + 1;
      const int right = left + 1;
      int smallest = node;
      if (left < k && is_above(left, smallest)) {
        smallest = left;
      }
      if (right < k && is_above(right, smallest)) {
        smallest = right;
      }
      if (smallest == node) {
        break;
      }
      swap(smallest, node);
      node = smallest;
    }
  }
  __device__ void build(int k) {
    for (int node = (k - 1) / 2; node >= 0; node--) {
      push_down(node, k);
    }
  }
  __device__ void remove_root(int k) {
    data[0] = data[k - 1];
    push_root_down(k - 1);
  }
  __device__ void sort(int k) {
    for (int slot = k - 1; slot > 0; slot--) {
      swap(slot, 0);
      push_root_down(slot);
    }
  }
  __device__ void replace_root(const Entry& entry, int k) {
    data[0] = entry;
    push_root_down(k);
  }
  __device__ const Entry& root() { return data[0]; }
};

template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T> make_indexed_heap(
    typename Data<T>::Entry* data) {
  return IndexedHeap<heapType, preferIndices, Data, T>(Data<T>(data));
}

// heapTopK walks over [input, input+length) with `step_size` stride starting at
// `start_index`. It builds a top-`k` heap that is stored in `heap_entries`
// using `Accessor` to access elements in `heap_entries`. If sorted=true, the
// elements will be sorted at the end.
template <typename T, template <typename> class Data>
__device__ void heapTopK(const T* __restrict__ input, int length, int k,
                         Entry<T>* __restrict__ heap_entries,
                         bool sorted = false, int start_index = 0,
                         int step_size = 1) {
  Data<T> _data(heap_entries);
  IndexedHeap<kMinHeap, kHigher, Data, T> heap(_data);
  int heap_end_index = start_index + k * step_size;
  if (heap_end_index > length) {
    heap_end_index = length;
  }

  // Initialize the min-heap.
  for (int index = start_index, slot = 0; index < heap_end_index;
       index += step_size, slot++) {
    Entry<T> entry(index, input[index]);
    heap.assign(slot, entry);
  }

  heap.build(k);

  // Now iterate over the remaining items.
  // If an item is smaller than the min element, it is not amongst the top k.
  // Otherwise, replace the min element with it and push upwards.
  for (int index = heap_end_index; index < length; index += step_size) {
    // We prefer elements with lower indices. This is given here.
    // Later elements automatically have higher indices, so can be
    // discarded.
    if (input[index] > heap.root().value) {
      // This element should replace the min.
      heap.replace_root(Entry<T>(index, input[index]), k);
    }
  }

  // Sort if wanted.
  if (sorted) {
    heap.sort(k);
  }
}

// mergeShards performs a top-k merge on `num_shards` many sorted streams that
// are sorted and stored in `entries` in a strided way:
// |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
// The overall top k elements are written to `top_k_values` and their indices
// to top_k_indices.
// `top_k_heap` is used as temporary storage for the merge heap.
template <typename T>
__device__ void mergeShards(int num_shards, int64_t k,
                            Entry<T>* __restrict__ entries,
                            Entry<T>* __restrict__ top_k_heap, T* top_k_values,
                            int* top_k_indices) {
  // If k < num_shards, we can use a min-heap with k elements to get the top k
  // of the sorted blocks. If k > num_shards, we can initialize a min-heap
  // with the top element from each sorted block.
  const int heap_size = k < num_shards ? k : num_shards;

  // Min-heap part.
  {
    IndexedHeap<kMinHeap, kHigher, IndirectLinearData, T> min_heap(
        IndirectLinearData<T>(top_k_heap, entries));
    // Initialize the heap as a min-heap.
    for (int slot = 0; slot < heap_size; slot++) {
      min_heap.assign(slot, Entry<T>(slot, entries[slot].value));
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards >
    // heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      const Entry<T>& entry = entries[shard];
      const Entry<T>& root = min_heap.root();
      if (entry.value < root.value) {
        continue;
      }
      if (entry.value == root.value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root(Entry<T>(shard, entry.value), heap_size);
    }
  }

  // Max-part.
  {
    // Turn the min-heap into a max-heap in-place.
    IndexedHeap<kMaxHeap, kLower, IndirectLinearData, T> max_heap(
        IndirectLinearData<T>(top_k_heap, entries));
    // Heapify into a max heap.
    max_heap.build(heap_size);

    // Now extract the minimum k-1 times.
    // k is treated specially.
    const int last_k = k - 1;
    for (int rank = 0; rank < last_k; rank++) {
      const Entry<T>& max_element = max_heap.root();
      top_k_values[rank] = max_element.value;
      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      int next_shard_index = shard_index + num_shards;
      // For rank < k-1, each top k heap still contains at least 1
      // element,
      max_heap.replace_root(
          Entry<T>(next_shard_index, entries[next_shard_index].value),
          heap_size);
    }

    // rank == last_k.
    const Entry<T>& max_element = max_heap.root();
    top_k_values[last_k] = max_element.value;
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
  }
}

extern __shared__ char shared_memory[];

template <typename T>
__global__ void TopKKernel(const T* input, int length, int64_t k, T* output,
                           int* output_indices) {
  const int batch_index = blockIdx.x;
  const T* batch_input = input + batch_index * length;

  const int thread_index = threadIdx.x;
  const int thread_count = blockDim.x;

  Entry<T>* shared_entries = (Entry<T>*)shared_memory;
  heapTopK<T, StridedData>(batch_input, length, k, shared_entries, true,
                           thread_index, thread_count);

  __syncthreads();

  if (thread_index == 0) {
    const int offset = batch_index * k;
    T* batch_output = output + offset;
    int* batch_indices = output_indices + offset;
    Entry<T>* top_k_heap = shared_entries + thread_count * k;

    mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
                batch_indices);
  }
}
}  // namespace impl
template <typename T>
void TopKKernelLauncher(T* output, int* output_indices, const T* input,
                        int batch_size, int length, int64_t k,
                        cudaStream_t stream) {
  const int max_shared_memory_size = 48 << 10;
  const int heap_size = k * sizeof(impl::Entry<T>);
  int num_shards = max_shared_memory_size / heap_size - 1;
  if (num_shards <= 0) {
    num_shards = 1;
  }
  int shard_size = length / num_shards;
  int min_shard_size = 2 * k;
  if (shard_size < min_shard_size) {
    num_shards = length / min_shard_size;
  }

  if (num_shards <= 0) {
    num_shards = 1;
  } else if (num_shards > 512) {
    num_shards = 512;
  }

  int shared_memory_size = (num_shards + 1) * k * sizeof(impl::Entry<T>);
  impl::TopKKernel<T><<<batch_size, num_shards, shared_memory_size, stream>>>(
      input, length, k, output, output_indices);
}
#ifdef ENABLE_FP16
template void TopKKernelLauncher<half>(half* output, int* output_indices,
                                       const half* input, int batch_size,
                                       int length, int64_t k,
                                       cudaStream_t stream);
#endif
template void TopKKernelLauncher<float>(float* output, int* output_indices,
                                        const float* input, int batch_size,
                                        int length, int64_t k,
                                        cudaStream_t stream);
template void TopKKernelLauncher<hie::bfloat16>(hie::bfloat16* output,
                                                int* output_indices,
                                                const hie::bfloat16* input,
                                                int batch_size, int length,
                                                int64_t k, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
