/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    change_dtype.cu
 */

#include <cmath>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void ConvertToFp32(float* out, T* in, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    out[idx] = float(in[idx]);
  }
}

template <typename T>
void ToFloatKernelLauncher(float* out, const T* in, int64_t count,
                           cudaStream_t stream) {
  int N = count;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  ConvertToFp32<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(out, in, count);
}

template void ToFloatKernelLauncher<float>(float* out, const float* in,
                                           int64_t count, cudaStream_t stream);

#ifdef ENABLE_FP16
template void ToFloatKernelLauncher<half>(float* out, const half* in,
                                          int64_t count, cudaStream_t stream);
#endif
template void ToFloatKernelLauncher<hie::bfloat16>(float* out,
                                                   const hie::bfloat16* in,
                                                   int64_t count,
                                                   cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
