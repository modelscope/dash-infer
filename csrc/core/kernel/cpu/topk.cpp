/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk.cpp
 */

#include <algorithm>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void TopKKernel(T* output, int* output_indices, const T* input, int batch_size,
                int length, int64_t k) {
  parallel_for(batch_size, [&](int b) {
    std::vector<std::pair<T, int> > data_vec(length);
    parallel_for(length, [&](int i) {
      data_vec[i] = std::make_pair(input[b * length + i], i);
    });
    std::partial_sort(data_vec.begin(), data_vec.begin() + k, data_vec.end(),
                      [](std::pair<T, int> lhs, std::pair<T, int> rhs) {
                        if (lhs.first > rhs.first) return true;
                        if (lhs.first < rhs.first) return false;
                        return lhs.second < rhs.second;
                      });
    parallel_for(k, [&](int i) {
      output[b * k + i] = data_vec[i].first;
      output_indices[b * k + i] = data_vec[i].second;
    });
  });
}
template void TopKKernel<float>(float* output, int* output_indices,
                                const float* input, int batch_size, int length,
                                int64_t k);
#ifdef ENABLE_FP16
template void TopKKernel<half>(half* output, int* output_indices,
                                const half* input, int batch_size, int length,
                                int64_t k);
#endif
#ifdef ENABLE_BF16
template void TopKKernel<hie::bfloat16>(hie::bfloat16* output, int* output_indices,
                                const hie::bfloat16* input, int batch_size, int length,
                                int64_t k);
#endif
}  // namespace cpu
}  // namespace allspark
