/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transpose.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void transpose_axis_01_kernel(T* out, T* in, const int dim0,
                                         const int dim1, const int dim2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dim0 * dim1 * dim2) {
    const int input_dim2_index = index % dim2;
    index = (index - input_dim2_index) / dim2;
    const int input_dim1_index = index % dim1;
    index = (index - input_dim1_index) / dim1;
    const int input_dim0_index = index % dim0;

    out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 +
        input_dim2_index] = in[input_dim0_index * dim1 * dim2 +
                               input_dim1_index * dim2 + input_dim2_index];
  }
}
template <typename T>
void transpose_axis_01_kernelLauncher(T* out, T* in, const int dim0,
                                      const int dim1, const int dim2,
                                      cudaStream_t stream) {
  int N = dim0 * dim1 * dim2;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  transpose_axis_01_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      out, in, dim0, dim1, dim2);
}

template void transpose_axis_01_kernelLauncher<float>(float* out, float* in,
                                                      const int dim0,
                                                      const int dim1,
                                                      const int dim2,
                                                      cudaStream_t stream);
#ifdef ENABLE_FP16
template void transpose_axis_01_kernelLauncher<half>(half* out, half* in,
                                                     const int dim0,
                                                     const int dim1,
                                                     const int dim2,
                                                     cudaStream_t stream);
#endif
template void transpose_axis_01_kernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, hie::bfloat16* in, const int dim0, const int dim1,
    const int dim2, cudaStream_t stream);
template void transpose_axis_01_kernelLauncher<int8_t>(int8_t* out, int8_t* in,
                                                       const int dim0,
                                                       const int dim1,
                                                       const int dim2,
                                                       cudaStream_t stream);

template <typename T>
__global__ void transpose_axis_12_kernel(T* out, T* in, const int dim0,
                                         const int dim1, const int dim2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dim0 * dim1 * dim2) {
    const int input_dim2_index = index % dim2;
    index = (index - input_dim2_index) / dim2;
    const int input_dim1_index = index % dim1;
    index = (index - input_dim1_index) / dim1;
    const int input_dim0_index = index % dim0;

    out[input_dim0_index * dim2 * dim1 + input_dim2_index * dim1 +
        input_dim1_index] = in[input_dim0_index * dim1 * dim2 +
                               input_dim1_index * dim2 + input_dim2_index];
  }
}
template <typename T>
void transpose_axis_12_kernelLauncher(T* out, T* in, const int dim0,
                                      const int dim1, const int dim2,
                                      cudaStream_t stream) {
  int N = dim0 * dim1 * dim2;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  transpose_axis_12_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      out, in, dim0, dim1, dim2);
}

template void transpose_axis_12_kernelLauncher<float>(float* out, float* in,
                                                      const int dim0,
                                                      const int dim1,
                                                      const int dim2,
                                                      cudaStream_t stream);
#ifdef ENABLE_FP16
template void transpose_axis_12_kernelLauncher<half>(half* out, half* in,
                                                     const int dim0,
                                                     const int dim1,
                                                     const int dim2,
                                                     cudaStream_t stream);
#endif
template void transpose_axis_12_kernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, hie::bfloat16* in, const int dim0, const int dim1,
    const int dim2, cudaStream_t stream);

template void transpose_axis_12_kernelLauncher<int8_t>(int8_t* out, int8_t* in,
                                                       const int dim0,
                                                       const int dim1,
                                                       const int dim2,
                                                       cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark