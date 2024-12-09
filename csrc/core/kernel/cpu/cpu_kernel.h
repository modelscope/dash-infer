/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_kernel.h
 */

#pragma once
#include <common.h>
#include <stdint.h>

#include <vector>
namespace allspark {
namespace cpu {
template <typename T>
void SimpleAdd(T* out, const T* in1, const T* in2, int count);
template <typename T>
void EmbeddingKernelLauncher(T* out_tensor, const int64_t* word_ids,
                             const int64_t* token_type_ids,
                             const T* embedding_table, const T* pos_table,
                             const T* token_type_table, int batch_size,
                             int seq_len, int hidden_size, int vocab_size,
                             int* offset_list, int force_offset,
                             bool use_decoder);
template <typename T>
void EmbeddingT5KernelLauncher(T* out_tensor, const int64_t* word_ids,
                               const T* embedding_table, int batch_size,
                               int seq_len, int hidden_size, int vocab_size,
                               bool use_decoder);

template <typename T>
void SelfScaledDpAttention(T* output, const T* query, const T* key,
                           const T* value, int q_num_heads, int kv_num_heads,
                           int size_per_head, int o_stride, int q_stride,
                           int kv_stride, int batch_size,
                           const int* input_seq_lens, const int* past_seq_lens,
                           void* workspace, int src_blk, int tgt_blk,
                           const float* mask, float scale, int num_thread);

template <typename T>
void GetBatchArrayLauncher(T* q, T* k, T* v, T* score, T* out, T** q_array,
                           T** k_array, T** v_array, T** score_array,
                           T** out_array, int batch_size, int beam_size,
                           int num_heads, int size_per_head, int step,
                           int q_stride, int kv_stride, int score_stride,
                           int out_stride);
template <typename T>
void MultiQueryGetBatchArrayLauncher(
    T* q, T* k, T* v, T* score, T* out, T** q_array, T** k_array, T** v_array,
    T** score_array, T** out_array, int batch_size, int beam_size,
    int num_heads, int size_per_head, int group_num, int step, int q_stride,
    int kv_stride, int score_stride, int out_stride);
template <typename T>
void BatchGemmWraper(void** matrix_C, void** matrix_A, void** matrix_B, int m,
                     int n, int k, bool transA, bool transB, float alpha,
                     float beta, int lda, int ldb, int ldc, int batch);
template <typename T>
void BatchSoftmax(T* score, const float* mask, int batch_size, int beam_size,
                  int num_heads, int seq_len, int step);
template <typename T>
void BatchDecoderSoftmax(T* score, const float* mask, int batch_size,
                         int beam_size, int num_heads, int seq_len, int step,
                         int input_len);
template <typename T>
void GemmWraper(T* matrix_C, const T* matrix_A, const T* matrix_B,
                const T* bias, int m, int n, int k, bool transA, bool transB,
                int lda, int ldb, int ldc, float alpha, float beta,
                const T* bin_res);
template <typename T>
void StridedBatchGemmWraper(T* matrix_C, const T* matrix_A, const T* matrix_B,
                            const T* bias, int m, int n, int k, bool transA,
                            bool transB, int lda, int ldb, int ldc, float alpha,
                            float beta, int batch, const T* bin_res);
template <typename T>
void MHAKernel(T* out, const T* q, const T* k, const T* v, const float* mask,
               T* score, int batch_size, int num_heads, int seq_length,
               int step, int hidden_size, int size_per_head, float alpha);
template <typename T0, typename T1>
void CastKernelLauncher(const T0* in, T1* out, int size);
template <typename T>
void LayerNormKernel(T* data_out, const T* data_in, const T* bias,
                     const T* gamma, const T* beta, int m, int n, float eps);
template <typename T>
void LayerNormNoBetaKernel(T* data_out, const T* data_in, const T* bias,
                           const T* gamma, int m, int n, float eps);
template <typename T>
void TransMaskKernel(T* out, const int64_t* mask, int batch_size,
                     int seq_length, bool seq_mask, bool blank);
template <typename T>
void RelativePEKernel(T* out, const T* attention_bias, int batch_size,
                      int seq_length, int k, int step, bool is_decoder);
template <typename T>
void ALiBiPEKernelLauncher(T* out, int* batch_offset, int batch_size,
                           int seq_length, int num_heads, int ori_num_heads,
                           int step, int rank);
template <typename T>
void MHAKernel(T* out, const T* q, const T* k, const T* v, const float* mask,
               T* score, int beam_size, int batch_size, int num_heads,
               int seq_length, int step, int hidden_size, int size_per_head,
               int q_stride, int kv_stride, int max_seq_len, float alpha);
template <typename T>
void TopKKernel(T* output, int* output_indices, const T* input, int batch_size,
                int length, int64_t k);
template <typename T>
void TopPKernel(T* input, int* k_arr, float* p_arr, int batch, int length);
template <typename T>
void SampleKernel(int64_t* out, void* state, T* in, const int* indice,
                  int batch_size, int* num_arr, int stride);
void SampleKernelInitLauncher(void* state, unsigned long long seed,
                              int batch_size);
template <typename T>
void LogSoftmaxKernel(const T* input, T* output, int outer_dim, int inner_dim);
template <typename T>
void SoftmaxKernel(T* input, int* len_arr, int outer_dim, int inner_dim,
                   float temperature = 1.0);
template <typename T>
void UpdateKVLauncher(T* k, T* v, const T* step_k, const T* step_v,
                      int batch_size, int step, int max_length, int hidden_size,
                      int seq_len, int stride);
template <typename T>
void PreProcessForGeneration(T* dec_ids, T* max_dec_ids, const T* in_ids, T bos,
                             int batch_size, int num_beam, int max_len,
                             int seq_len);
template <typename T>
void UpdateId(T* out, const T* in, const int* beam_idx, T* tmp_id,
              int batch_size, int beam_size, int* step_list, int max_length,
              int seq_len);
template <typename T>
void PostProcessId(T* out_ids, const T* in_ids, int batch_size, int in_stride,
                   int out_stride);
template <typename T>
void BeamScoreInitLauncher(T* beam_score, T* hyps_beam_score,
                           int64_t* hyps_beam_idx, int* eos_count,
                           int batch_size, int beam_size);
template <typename T>
void LogitsProcessor(T* score, const int64_t* in_ids, int64_t in_ids_len,
                     int batch_size, int cur_len, int max_len, int vocab_size,
                     int* bad_words_ids, std::vector<int>& bad_ids_size,
                     GenerateConfig gen_cfg, void* ws_ptr, size_t ws_bytes);
template <typename T>
void AddScoreLauncher(T* next_score, const T* beam_score, int batch,
                      int length);
template <typename T>
void UpdateBeamScoreLauncher(T* beam_score, int64_t* beam_next_token,
                             int* beam_idx, const T* topk_score,
                             const int* topk_indice, int batch_size,
                             int beam_size, int vocab_size, int eos,
                             int* eos_count, int* hyps_beam_idx,
                             T* hyps_beam_score, int64_t* hyps_id,
                             int64_t* in_ids, int cur_len, int loop_len,
                             float length_penalty);
template <typename T>
void ReorderKVCacheLauncher(T* k, T* v, const T* old_k, const T* old_v,
                            const int* beam_idx, int batch_size, int beam_size,
                            int inner_dim);
// copy A[batch,inner] ->B[batch,beam,inner]
template <typename T>
void CopyMatrix(const int M, const int N, const T* A, const int lda, T* B,
                const int ldb);
template <typename T>
void CopyMultiBeam(void* A, void* B, int batch_size, int beam_size,
                   int inner_dim);
template <typename T>
void quantize(int m, int n, const T* data, int8_t* q_data, float* scale,
              int8_t* zero, int* redsum);
template <typename T>
void ChunkKernelLauncher(T* out, T* in, int batch, int seq_len, int hidden_size,
                         int chunk_split);
template <typename T>
void ChunkBinary(T* out, T* in, int batch, int seq_len, int hidden_size,
                 int chunk_split, int type);
template <typename T>
void RotaryKernelLauncher(T* out, T* in, float* inv_freq, int* batch_offset,
                          int batch_size, int seq_len, int num_head,
                          int size_per_head, int* step_list, int stride,
                          int xlogn);
template <typename T>
void RotaryEmbedding2D(T* output, T* input, float* inv_freq, int* batch_offset,
                       int batch, int seq_len, int head, int size_per_head,
                       int step, int stride, int input_len);
template <typename T>
void RotaryEmbeddingHalfInner(T* out, T* in, float* inv_freq, int* batch_offset,
                              int batch_size, int seq_len, int num_head,
                              int inner, int* step_list, int stride);
template <typename T>
void RotaryPctKernelLauncher(T* out, T* in, float* inv_freq, int* batch_offset,
                             int batch_size, int seq_len, int num_head,
                             int inner, int step, int stride, float pct);
template <typename T>
void TransposeAxis01KernelLauncher(T* out, const T* in, int dim0, int dim1,
                                   int dim2);
template <typename T>
void MulKernelLauncher(T* out, const T* in, int64_t count, float alpha);
}  // namespace cpu
}  // namespace allspark
