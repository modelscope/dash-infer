/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp.cpp
 */

#include <algorithm>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void TopPKernel(T* input, int* k_arr, float* p_arr, int batch, int length) {
  parallel_for(batch, [&](int b) {
    if (p_arr[b] > 1e-7) {
      T* input_now = input + b * length;
      float sum = 0;
      int last = k_arr[b];
      for (int i = 0; i < k_arr[b]; i++) {
        sum += (float)input_now[i];
        if (sum > p_arr[b]) {
          last = i + 1;
          break;
        }
      }
      k_arr[b] = last;
    }
  });
}

template void TopPKernel<float>(float* input, int* k_arr, float* p_arr,
                                int batch, int length);

}  // namespace cpu
}  // namespace allspark
