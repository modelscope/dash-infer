/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embeddingT5.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

template <bool DEC_OPT, typename T>
__global__ static void embedding_kernel(T* out_tensor, const int64_t* word_ids,
                                        const T* embed_table, int N,
                                        int seq_len, int hidden_size,
                                        int vocab_size) {
  uint32_t n_thread = gridDim.x * blockDim.x;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t idx = tid; idx < N; idx += n_thread) {
    int row_idx = idx / hidden_size;
    int col_idx = idx % hidden_size;
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] = embed_table[word_ids[row_idx] * hidden_size + col_idx];
    }
  }
}

template <bool DEC_OPT, typename T>
void EmbeddingT5KernelLauncher(T* out_tensor, const int64_t* word_ids,
                               const T* embedding_table, int batch_size,
                               int seq_len, int hidden_size, int vocab_size,
                               cudaStream_t stream) {
  int N = 0;
  if (DEC_OPT) {
    N = batch_size * hidden_size;
  } else {
    N = batch_size * seq_len * hidden_size;
  }
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  embedding_kernel<DEC_OPT, T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      out_tensor, word_ids, embedding_table, N, seq_len, hidden_size,
      vocab_size);
}

template void EmbeddingT5KernelLauncher<true, float>(
    float* out_tensor, const int64_t* word_ids, const float* embedding_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size,
    cudaStream_t stream);
template void EmbeddingT5KernelLauncher<false, float>(
    float* out_tensor, const int64_t* word_ids, const float* embedding_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size,
    cudaStream_t stream);

#ifdef ENABLE_FP16
template void EmbeddingT5KernelLauncher<true, half>(
    half* out_tensor, const int64_t* word_ids, const half* embedding_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size,
    cudaStream_t stream);
template void EmbeddingT5KernelLauncher<false, half>(
    half* out_tensor, const int64_t* word_ids, const half* embedding_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size,
    cudaStream_t stream);
#endif
template void EmbeddingT5KernelLauncher<true, hie::bfloat16>(
    hie::bfloat16* out_tensor, const int64_t* word_ids,
    const hie::bfloat16* embedding_table, int batch_size, int seq_len,
    int hidden_size, int vocab_size, cudaStream_t stream);
template void EmbeddingT5KernelLauncher<false, hie::bfloat16>(
    hie::bfloat16* out_tensor, const int64_t* word_ids,
    const hie::bfloat16* embedding_table, int batch_size, int seq_len,
    int hidden_size, int vocab_size, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
