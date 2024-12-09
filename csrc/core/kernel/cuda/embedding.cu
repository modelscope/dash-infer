/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embedding.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

template <bool DEC_OPT, typename T>
__global__ static void embedding_with_token_kernel(
    T* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const T* embed_table, const T* pos_table, const T* token_type_table, int N,
    int seq_len, int hidden_size, int vocab_size, int* offset_list,
    int force_offset) {
  uint32_t n_thread = gridDim.x * blockDim.x;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t idx = tid; idx < N; idx += n_thread) {
    int row_idx = idx / hidden_size;
    int col_idx = idx % hidden_size;
    int pos_idx;
    int token_idx;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (DEC_OPT) {
      pos_idx = (seq_len + offset) * hidden_size + col_idx;
      token_idx = col_idx;
    } else {
      pos_idx = (row_idx % seq_len + offset) * hidden_size + col_idx;
      token_idx = token_type_ids[row_idx] * hidden_size + col_idx;
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] = embed_table[word_ids[row_idx] * hidden_size + col_idx] +
                        pos_table[pos_idx] + token_type_table[token_idx];
    }
  }
}
template <bool DEC_OPT, typename T>
__global__ static void embedding_kernel(T* out_tensor, const int64_t* word_ids,
                                        const T* embed_table,
                                        const T* pos_table, int N, int seq_len,
                                        int hidden_size, int vocab_size,
                                        int* offset_list, int force_offset) {
  uint32_t n_thread = gridDim.x * blockDim.x;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t idx = tid; idx < N; idx += n_thread) {
    int row_idx = idx / hidden_size;
    int col_idx = idx % hidden_size;
    int pos_idx;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (DEC_OPT) {
      pos_idx = seq_len;
    } else {
      pos_idx = row_idx % seq_len;
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] = embed_table[word_ids[row_idx] * hidden_size + col_idx] +
                        pos_table[(pos_idx + offset) * hidden_size + col_idx];
    }
  }
}

template <bool DEC_OPT, typename T>
void EmbeddingKernelLauncher(T* out_tensor, const int64_t* word_ids,
                             const int64_t* token_type_ids,
                             const T* embedding_table, const T* pos_table,
                             const T* token_type_table, int batch_size,
                             int seq_len, int hidden_size, int vocab_size,
                             int* offset, int force_offset,
                             cudaStream_t stream) {
  int N = 0;
  if (DEC_OPT) {
    N = batch_size * hidden_size;
  } else {
    N = batch_size * seq_len * hidden_size;
  }
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  if (token_type_table != nullptr) {
    embedding_with_token_kernel<DEC_OPT, T>
        <<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
            out_tensor, word_ids, token_type_ids, embedding_table, pos_table,
            token_type_table, N, seq_len, hidden_size, vocab_size, offset,
            force_offset);
  } else {
    embedding_kernel<DEC_OPT, T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        out_tensor, word_ids, embedding_table, pos_table, N, seq_len,
        hidden_size, vocab_size, offset, force_offset);
  }
}

template void EmbeddingKernelLauncher<true, float>(
    float* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const float* embedding_table, const float* pos_table,
    const float* token_type_table, int batch_size, int seq_len, int hidden_size,
    int vocab_size, int* offset, int force_offset, cudaStream_t stream);
template void EmbeddingKernelLauncher<false, float>(
    float* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const float* embedding_table, const float* pos_table,
    const float* token_type_table, int batch_size, int seq_len, int hidden_size,
    int vocab_size, int* offset, int force_offset, cudaStream_t stream);

#ifdef ENABLE_FP16
template void EmbeddingKernelLauncher<true, half>(
    half* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const half* embedding_table, const half* pos_table,
    const half* token_type_table, int batch_size, int seq_len, int hidden_size,
    int vocab_size, int* offset, int force_offset, cudaStream_t stream);
template void EmbeddingKernelLauncher<false, half>(
    half* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const half* embedding_table, const half* pos_table,
    const half* token_type_table, int batch_size, int seq_len, int hidden_size,
    int vocab_size, int* offset, int force_offset, cudaStream_t stream);
#endif
template void EmbeddingKernelLauncher<true, hie::bfloat16>(
    hie::bfloat16* out_tensor, const int64_t* word_ids,
    const int64_t* token_type_ids, const hie::bfloat16* embedding_table,
    const hie::bfloat16* pos_table, const hie::bfloat16* token_type_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size, int* offset,
    int force_offset, cudaStream_t stream);
template void EmbeddingKernelLauncher<false, hie::bfloat16>(
    hie::bfloat16* out_tensor, const int64_t* word_ids,
    const int64_t* token_type_ids, const hie::bfloat16* embedding_table,
    const hie::bfloat16* pos_table, const hie::bfloat16* token_type_table,
    int batch_size, int seq_len, int hidden_size, int vocab_size, int* offset,
    int force_offset, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
