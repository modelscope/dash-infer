/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "reduce.cuh"
namespace allspark {
namespace cuda {
template <typename T>
__global__ static void logsoftmax_kernel(const T* input, T* output,
                                         const int step) {
  float tmp = -1e20f;
  float qk = 0.0f;
  int qk_offset = blockIdx.x * step + threadIdx.x;
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

  float sum_val = expf(tmp - s_max);
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = logf(sum_val + 1e-12f);
  }
  __syncthreads();
  if (threadIdx.x < step) {
    output[qk_offset] = qk - s_max - s_sum;
  }
}
template <typename T, int BLOCK = 1024, int UNROLL>
__launch_bounds__(1024, 1) __global__
    void logsoftmax_kernel_UNROLL(const T* input, T* output, const int step) {
  float tmp[UNROLL];
  float qk[UNROLL];
  __shared__ float s_sum, s_max;
  float max_val = -1e20f;
  float sum_val = 0.0f;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset = blockIdx.x * step + tid;
      qk[i] = static_cast<float>(input[qk_offset]);
      tmp[i] = qk[i];
      max_val = max(max_val, tmp[i]);
    }
  }
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      tmp[i] = expf(tmp[i] - s_max);
      sum_val += tmp[i];
    }
  }
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = logf(sum_val + 1e-12f);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset = blockIdx.x * step + tid;
      output[qk_offset] = (T)(qk[i] - s_max - s_sum);
    }
  }
}

template <typename T>
void LogSoftmaxKernelLauncher(const T* input, T* output, int out_dim,
                              int inner_dim, cudaStream_t stream) {
  dim3 block, grid;
  grid.x = out_dim;
  if (inner_dim <= 1024) {
    block.x = (inner_dim + 31) / 32 * 32;
    logsoftmax_kernel<<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else if (inner_dim <= 2048) {
    const int unroll = 2;
    block.x = 1024;
    logsoftmax_kernel_UNROLL<T, 1024, unroll>
        <<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else if (inner_dim <= 3072) {
    const int unroll = 3;
    block.x = 1024;
    logsoftmax_kernel_UNROLL<T, 1024, unroll>
        <<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else if (inner_dim <= 4096) {
    const int unroll = 4;
    block.x = 1024;
    logsoftmax_kernel_UNROLL<T, 1024, unroll>
        <<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else if (inner_dim <= 8192) {
    const int unroll = 8;
    block.x = 1024;
    logsoftmax_kernel_UNROLL<T, 1024, unroll>
        <<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else if (inner_dim <= 8192 * 8) {
    const int unroll = 8 * 8;
    block.x = 1024;
    logsoftmax_kernel_UNROLL<T, 1024, unroll>
        <<<grid, block, 0, stream>>>(input, output, inner_dim);
  } else {
    throw AsException(
        "inner_dim > 8192 * 8 not supported yet in LogSoftmaxKernelLauncher()");
  }
}
template void LogSoftmaxKernelLauncher<float>(const float* input, float* output,
                                              int out_dim, int inner_dim,
                                              cudaStream_t stream);
#ifdef ENABLE_FP16
template void LogSoftmaxKernelLauncher<half>(const half* input, half* output,
                                             int out_dim, int inner_dim,
                                             cudaStream_t stream);
#endif
template void LogSoftmaxKernelLauncher<hie::bfloat16>(
    const hie::bfloat16* input, hie::bfloat16* output, int out_dim,
    int inner_dim, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
