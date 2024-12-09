/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    replace_embedding.cu
 */

#include <cmath>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "elementwise.cuh"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void replace_embedding_kernel(int N, T* out, float* in) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    out[tid] = (T)in[tid];
  }
}

template <typename T>
void ReplaceEmbedding(T* output, float* input, int64_t count,
                      cudaStream_t stream) {
  int N = count;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  replace_embedding_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input);
}

template void ReplaceEmbedding<float>(float* output, float* input,
                                      int64_t count, cudaStream_t stream);

#ifdef ENABLE_FP16
template void ReplaceEmbedding<half>(half* output, float* input, int64_t count,
                                     cudaStream_t stream);
#endif
template void ReplaceEmbedding<hie::bfloat16>(hie::bfloat16* output,
                                              float* input, int64_t count,
                                              cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
