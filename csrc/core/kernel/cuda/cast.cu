/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {
template <typename T0, typename T1>
__global__ static void cast_kernel(const T0* in, T1* out, int size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = static_cast<T1>(in[tid]);
  }
}
template <typename T0, typename T1>
void CastKernelLauncher(const T0* in, T1* out, int size, cudaStream_t stream) {
  const int block_num = (size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  cast_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(in, out, size);
}
template void CastKernelLauncher<int64_t, float>(const int64_t* in, float* out,
                                                 int size, cudaStream_t stream);
template void CastKernelLauncher<float, float>(const float* in, float* out,
                                               int size, cudaStream_t stream);
#ifdef ENABLE_FP16
template void CastKernelLauncher<half, float>(const half* in, float* out,
                                              int size, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void CastKernelLauncher<hie::bfloat16, float>(const hie::bfloat16* in,
                                                       float* out, int size,
                                                       cudaStream_t stream);
#endif
}  // namespace cuda
}  // namespace allspark
