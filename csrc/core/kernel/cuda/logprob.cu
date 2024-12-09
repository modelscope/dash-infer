/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    logprob.cu
 */

#include <cmath>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void selet_token_logprb_kernel(T* logprob, float* out,
                                          int64_t* out_tokens, int length,
                                          int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    out[idx] = float(logprob[idx * length + out_tokens[idx]]);
  }
}

template <typename T>
void SelectBatchTokenLogprobLauncher(T* typed_logprobs,
                                     float* float_token_logprobs,
                                     int64_t* out_tokens, int batch_size,
                                     int length, cudaStream_t stream) {
  int N = batch_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  selet_token_logprb_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      typed_logprobs, float_token_logprobs, out_tokens, length, N);
}

template void SelectBatchTokenLogprobLauncher<float>(
    float* typed_logprobs, float* float_token_logprobs, int64_t* out_tokens,
    int batch_size, int length, cudaStream_t stream);

#ifdef ENABLE_FP16
template void SelectBatchTokenLogprobLauncher<half>(half* typed_logprobs,
                                                    float* float_token_logprobs,
                                                    int64_t* out_tokens,
                                                    int batch_size, int length,
                                                    cudaStream_t stream);
#endif
// #ifndef ENABLE_BF16
template void SelectBatchTokenLogprobLauncher<hie::bfloat16>(
    hie::bfloat16* typed_logprobs, float* float_token_logprobs,
    int64_t* out_tokens, int batch_size, int length, cudaStream_t stream);
// #endif
}  // namespace cuda
}  // namespace allspark
