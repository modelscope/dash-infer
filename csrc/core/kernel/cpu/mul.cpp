/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mul.cpp
 */

#include <cstring>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void MulKernelLauncher(T* out, const T* in, int64_t count, float alpha) {
  int N = count;
  parallel_for(N, [&](int tid) { out[tid] = in[tid] * alpha; });
}

template void MulKernelLauncher<float>(float* out, const float* in,
                                       int64_t count, float alpha);

}  // namespace cpu
}  // namespace allspark