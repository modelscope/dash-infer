/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    calc_expert.cu
 */

#include <math.h>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

template <typename T>
__global__ void calc_expert_kernel(int N, T* out, T* in, T* expert_weight,
                                   int total_token, int hidden_size,
                                   int num_export) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int bs = tid / hidden_size;
    out[tid] = in[tid] * expert_weight[bs];
  }
}

template <typename T>
void CalcExpertKernelLauncher(T* output, T* input, T* expert_weight,
                              int total_token, int hidden_size, int num_expert,
                              cudaStream_t stream) {
  int N = total_token * hidden_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  calc_expert_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, expert_weight, total_token, hidden_size, num_expert);
}

template void CalcExpertKernelLauncher<float>(float* output, float* input,
                                              float* expert_weight,
                                              int total_token, int hidden_size,
                                              int num_expert,
                                              cudaStream_t stream);

#ifdef ENABLE_FP16
template void CalcExpertKernelLauncher<half>(half* output, half* input,
                                             half* expert_weight,
                                             int total_token, int hidden_size,
                                             int num_expert,
                                             cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void CalcExpertKernelLauncher<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, hie::bfloat16* expert_weight,
    int total_token, int hidden_size, int num_expert, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark