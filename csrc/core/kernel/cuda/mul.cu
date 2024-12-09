/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mul.cu
 */

#include <cmath>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "elementwise.cuh"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void mul_kernel(int N, T* out, const T* in, float alpha) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    out[tid] = in[tid] * alpha;
  }
}

template <typename T>
void MulKernelLauncher(T* out, const T* in, int64_t count, float alpha,
                       cudaStream_t stream) {
  int N = count;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  mul_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(N, out, in, alpha);
}

template void MulKernelLauncher<float>(float* out, const float* in,
                                       int64_t count, float alpha,
                                       cudaStream_t stream);

#ifdef ENABLE_FP16
template void MulKernelLauncher<half>(half* out, const half* in, int64_t count,
                                      float alpha, cudaStream_t stream);
#endif
template void MulKernelLauncher<hie::bfloat16>(hie::bfloat16* out,
                                               const hie::bfloat16* in,
                                               int64_t count, float alpha,
                                               cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
