/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    process_id.cpp
 */

#include <cstring>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void CopyMatrix(const int M, const int N, const T* A, const int lda, T* B,
                const int ldb) {
  parallel_for(M,
               [&](int i) { memcpy(B + i * ldb, A + i * lda, N * sizeof(T)); });
}
template void CopyMatrix<float>(const int M, const int N, const float* A,
                                const int lda, float* B, const int ldb);
template <typename T>
void PreProcessForGeneration(T* dec_ids, T* max_dec_ids, const T* in_ids, T bos,
                             int batch_size, int num_beam, int max_len,
                             int in_len) {
  if (bos != -1) {
    int N = batch_size * num_beam;
    parallel_for(N, [&](int i) {
      dec_ids[i] = bos;
      max_dec_ids[i * max_len] = bos;
    });
  } else {
    int N = batch_size * num_beam * in_len;
    parallel_for(N, [&](int i) { dec_ids[i] = in_ids[i]; });
  }
}
template void PreProcessForGeneration<int64_t>(
    int64_t* dec_ids, int64_t* max_dec_ids, const int64_t* in_ids, int64_t bos,
    int batch_size, int num_beam, int max_len, int seq_len);

template <typename T>
void UpdateId(T* out, const T* in, const int* beam_idx, T* tmp_id,
              int batch_size, int beam_size, int* step_list, int max_length,
              int seq_len) {
  // if (beam_idx != nullptr) {
  //     CopyMatrix(batch_size * beam_size, step, out, max_length, tmp_id,
  //     max_length); int N = batch_size * beam_size * step; parallel_for(N,
  //     [&](int i) {
  //         int batch = i / (beam_size * step);
  //         int beam = i % (beam_size * step) / step;
  //         int idx2 = i % step;
  //         out[(batch * beam_size + beam)* max_length + idx2] = tmp_id[(batch
  //         * beam_size + beam_idx[batch * beam_size + beam]) * max_length +
  //         idx2];
  //     });
  // }
  int N = batch_size * seq_len;
  int length = seq_len;
  parallel_for(N, [&](int tid) {
    int batch = tid / (length);
    int seq_len = tid % length;
    out[batch * max_length + step_list[batch] + seq_len] =
        in[batch * length + seq_len];
  });
}

template void UpdateId<int64_t>(int64_t* out, const int64_t* in,
                                const int* beam_idx, int64_t* tmp_id,
                                int batch_size, int beam_size, int* step_list,
                                int max_length, int seq_len);

template <typename T>
void PostProcessId(T* out_ids, const T* in_ids, int batch_size, int in_stride,
                   int out_stride) {
  CopyMatrix(batch_size, out_stride, in_ids, in_stride, out_ids, out_stride);
}

template void PostProcessId<int64_t>(int64_t* out_ids, const int64_t* in_ids,
                                     int batch_size, int in_stride,
                                     int out_stride);

}  // namespace cpu
}  // namespace allspark