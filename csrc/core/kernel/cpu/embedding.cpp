/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embedding.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void EmbeddingKernel(T* out_tensor, const int64_t* word_ids,
                     const int64_t* token_type_ids, const T* embedding_table,
                     const T* pos_table, const T* token_type_table,
                     int batch_size, int seq_len, int hidden_size,
                     int vocab_size, int* offset_list, int force_offset) {
  int N = batch_size * seq_len * hidden_size;
  parallel_for(N, [&](int idx) {
    int row_idx = idx / hidden_size;
    int seq_idx = row_idx % seq_len;
    int col_idx = idx % hidden_size;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] =
          embedding_table[word_ids[row_idx] * hidden_size + col_idx] +
          pos_table[(seq_idx + offset) * hidden_size + col_idx] +
          token_type_table[token_type_ids[row_idx] * hidden_size + col_idx];
    }
  });
}
template <typename T>
void EmbeddingNoTokenTypeKernel(T* out_tensor, const int64_t* word_ids,
                                const int64_t* token_type_ids,
                                const T* embedding_table, const T* pos_table,
                                const T* token_type_table, int batch_size,
                                int seq_len, int hidden_size, int vocab_size,
                                int* offset_list, int force_offset) {
  int N = batch_size * seq_len * hidden_size;
  parallel_for(N, [&](int idx) {
    int row_idx = idx / hidden_size;
    int seq_idx = row_idx % seq_len;
    int col_idx = idx % hidden_size;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] =
          embedding_table[word_ids[row_idx] * hidden_size + col_idx] +
          pos_table[(seq_idx + offset) * hidden_size + col_idx];
    }
  });
}
template <typename T>
void EmbeddingDecoderKernel(T* out_tensor, const int64_t* word_ids,
                            const int64_t* token_type_ids,
                            const T* embedding_table, const T* pos_table,
                            const T* token_type_table, int batch_size, int step,
                            int hidden_size, int vocab_size, int* offset_list,
                            int force_offset) {
  int N = batch_size * hidden_size;
  parallel_for(N, [&](int idx) {
    int row_idx = idx / hidden_size;
    int seq_idx = step;
    int col_idx = idx % hidden_size;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] =
          embedding_table[word_ids[row_idx] * hidden_size + col_idx] +
          pos_table[(seq_idx + offset) * hidden_size + col_idx] +
          token_type_table[col_idx];
    }
  });
}
template <typename T>
void EmbeddingDecoderNoTokenTypeKernel(
    T* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const T* embedding_table, const T* pos_table, const T* token_type_table,
    int batch_size, int step, int hidden_size, int vocab_size, int* offset_list,
    int force_offset) {
  int N = batch_size * hidden_size;
  parallel_for(N, [&](int idx) {
    int row_idx = idx / hidden_size;
    int seq_idx = step;
    int col_idx = idx % hidden_size;
    int offset = force_offset;
    if (offset_list != nullptr) {
      offset += offset_list[row_idx];
    }
    if (word_ids[row_idx] < 0 || word_ids[row_idx] >= vocab_size) {
      // Illegal words
    } else {
      out_tensor[idx] =
          embedding_table[word_ids[row_idx] * hidden_size + col_idx] +
          pos_table[(seq_idx + offset) * hidden_size + col_idx];
    }
  });
}
template <typename T>
void EmbeddingKernelLauncher(T* out_tensor, const int64_t* word_ids,
                             const int64_t* token_type_ids,
                             const T* embedding_table, const T* pos_table,
                             const T* token_type_table, int batch_size,
                             int seq_len, int hidden_size, int vocab_size,
                             int* offset, int force_offset, bool use_decoder) {
  if (!use_decoder) {
    if (token_type_ids != nullptr) {
      EmbeddingKernel(out_tensor, word_ids, token_type_ids, embedding_table,
                      pos_table, token_type_table, batch_size, seq_len,
                      hidden_size, vocab_size, offset, force_offset);
    } else {
      EmbeddingNoTokenTypeKernel(out_tensor, word_ids, token_type_ids,
                                 embedding_table, pos_table, token_type_table,
                                 batch_size, seq_len, hidden_size, vocab_size,
                                 offset, force_offset);
    }
  } else {
    if (token_type_table != nullptr) {
      EmbeddingDecoderKernel(out_tensor, word_ids, token_type_ids,
                             embedding_table, pos_table, token_type_table,
                             batch_size, seq_len, hidden_size, vocab_size,
                             offset, force_offset);
    } else {
      EmbeddingDecoderNoTokenTypeKernel(
          out_tensor, word_ids, token_type_ids, embedding_table, pos_table,
          token_type_table, batch_size, seq_len, hidden_size, vocab_size,
          offset, force_offset);
    }
  }
}
template void EmbeddingKernelLauncher<float>(
    float* out_tensor, const int64_t* word_ids, const int64_t* token_type_ids,
    const float* embedding_table, const float* pos_table,
    const float* token_type_table, int batch_size, int seq_len, int hidden_size,
    int vocab_size, int* offset_list, int force_offset, bool use_decoder);
}  // namespace cpu
}  // namespace allspark
