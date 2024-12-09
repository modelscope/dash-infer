/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_low_reduce.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "reduce.cuh"
namespace allspark {
namespace cuda {
template <typename T>
__global__ static void softmax_low_reduce_kernel(const T* input, T* output,
                                                 const int step) {
  float tmp = -1.f / 0.f;
  float qk = 0.0f;
  int64_t qk_offset = blockIdx.x * step + threadIdx.x;
  __shared__ float s_sum, s_max;
  if (threadIdx.x < step) {
    qk = static_cast<float>(input[qk_offset]);
    tmp = qk;
  }
  float max_val = tmp;
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    qk = expf(tmp - s_max);
  }
  float sum_val = qk;
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    output[qk_offset] = qk / s_sum;
  }
}

template <typename T>
void SoftmaxLowReduceKernelLauncher(const T* input, T* output, int out_dim,
                                    int inner_dim, cudaStream_t stream) {
  dim3 block, grid;
  grid.x = out_dim;
  if (inner_dim <= 32) {
    block.x = 32;
    softmax_low_reduce_kernel<<<grid, block, 0, stream>>>(input, output,
                                                          inner_dim);
  } else if (inner_dim <= 64) {
    block.x = 64;
    softmax_low_reduce_kernel<<<grid, block, 0, stream>>>(input, output,
                                                          inner_dim);
  } else if (inner_dim <= 128) {
    block.x = 128;
    softmax_low_reduce_kernel<<<grid, block, 0, stream>>>(input, output,
                                                          inner_dim);
  } else if (inner_dim <= 256) {
    block.x = 256;
    softmax_low_reduce_kernel<<<grid, block, 0, stream>>>(input, output,
                                                          inner_dim);
  } else {
    throw AsException(
        "inner_dim > 256 not supported yet in SoftmaxLowReduceKernelLauncher");
  }
}
template void SoftmaxLowReduceKernelLauncher<float>(const float* input,
                                                    float* output, int out_dim,
                                                    int inner_dim,
                                                    cudaStream_t stream);
#ifdef ENABLE_FP16
template void SoftmaxLowReduceKernelLauncher<half>(const half* input,
                                                   half* output, int out_dim,
                                                   int inner_dim,
                                                   cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void SoftmaxLowReduceKernelLauncher<hie::bfloat16>(
    const hie::bfloat16* input, hie::bfloat16* output, int out_dim,
    int inner_dim, cudaStream_t stream);
#endif
}  // namespace cuda
}  // namespace allspark
