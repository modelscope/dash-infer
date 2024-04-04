/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embeddingT5.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void EmbeddingKernel(T* out_tensor, const int64_t* word_ids,
                     const T* embedding_table, int batch_size, int seq_len,
                     int hidden_size, int vocab_size) {
  int N = batch_size * seq_len * hidden_size;
  parallel_for(N, [&](int idx) {
    int row_idx = idx / hidden_size;
    int col_idx = idx % hidden_size;
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] =
          embedding_table[word_ids[row_idx] * hidden_size + col_idx];
    }
  });
}
template <typename T>
void EmbeddingT5KernelLauncher(T* out_tensor, const int64_t* word_ids,
                               const T* embedding_table, int batch_size,
                               int seq_len, int hidden_size, int vocab_size,
                               bool use_decoder) {
  EmbeddingKernel(out_tensor, word_ids, embedding_table, batch_size, seq_len,
                  hidden_size, vocab_size);
}
template void EmbeddingT5KernelLauncher<float>(float* out_tensor,
                                               const int64_t* word_ids,
                                               const float* embedding_table,
                                               int batch_size, int seq_len,
                                               int hidden_size, int vocab_size,
                                               bool use_decoder);
}  // namespace cpu
}  // namespace allspark
