/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_kernel.h
 */

#pragma once

#include <vector>

#include "cuda_common.h"  //NOLINT
namespace allspark {
namespace cuda {
template <typename T>
void broadcast_kernel_launcher(T* dst, const T* src, int gemm_size,
                               int hidden_size, int batch_size,
                               cudaStream_t stream);
template <typename T>
void MulKernelLauncher(T* out, const T* in, int64_t count, float alpha,
                       cudaStream_t stream);
template <typename T>
void UnaryKernelLauncher(T* out, const T* in, int64_t count, int type,
                         cudaStream_t stream);
template <typename T>
void UnaryGLUKernelLauncher(T* out, const T* in, size_t outer_size,
                            size_t inner_size, int type, cudaStream_t stream);
template <typename T>
void BinaryKernelLauncher(T* out, const T* in1, const T* in2, int count,
                          int type, cudaStream_t stream);
template <bool DEC_OPT, typename T>
void EmbeddingKernelLauncher(T* out_tensor, const int64_t* word_ids,
                             const int64_t* token_type_ids,
                             const T* embedding_table, const T* pos_table,
                             const T* token_type_table, int batch_size,
                             int seq_len, int hidden_size, int vocab_size,
                             int* offset, int force_offset,
                             cudaStream_t stream);
template <bool DEC_OPT, typename T>
void EmbeddingT5KernelLauncher(T* out_tensor, const int64_t* word_ids,
                               const T* embedding_table, int batch_size,
                               int seq_len, int hidden_size, int vocab_size,
                               cudaStream_t stream);
template <typename T>
void LayerNormKernelLauncher(T* out, const T* input, const T* bias,
                             const T* gamma, const T* beta, int m, int n,
                             float eps, cudaStream_t stream);
template <typename T>
void LayerNormNoBetaKernelLauncher(T* out, const T* input, const T* bias,
                                   const T* gamma, int m, int n, float eps,
                                   cudaStream_t stream);
template <typename T>
void GemmWraper(T* matrix_C, const T* matrix_A, const T* matrix_B,
                const T* bias, int m, int n, int k, bool transA, bool transB,
                int lda, int ldb, int ldc, float alpha, float beta,
                const T* bin_res, cublasHandle_t handle, cudaStream_t stream);
template <typename T>
void StridedBatchGemmWraper(T* matrix_C, const T* matrix_A, const T* matrix_B,
                            const T* bias, int m, int n, int k, bool transA,
                            bool transB, int lda, int ldb, int ldc, float alpha,
                            float beta, int batch, const T* bin_res,
                            cublasHandle_t handle, cudaStream_t stream);
template <typename DType>
bool SgmvCutlass(DType* y, const DType* x, const DType** w, const int32_t* s,
                 const int32_t* ranks, void* tmp_d, bool is_k_tensor,
                 bool is_n_tensor, int num_problems, int d_in, int d_out,
                 bool unsplit, int unsplit_n, int max_rank, int CC,
                 cudaStream_t stream);
template <typename DType>
bool SgmvSplitQKV(DType** out_ptrs, const DType* in, const int32_t* s,
                  const int32_t* lora_B_ranks, int max_rank, int num_problems,
                  cudaStream_t stream);
template <typename T>
void SoftmaxKernelLauncher(T* qk_buf, const float* mask, int batch_size,
                           int beam_size, int num_heads, int seq_len, int step,
                           cudaStream_t stream);
template <typename T>
void LogSoftmaxKernelLauncher(const T* input, T* output, int out_dim,
                              int inner_dim, cudaStream_t stream);

template <typename T>
void LognSoftmaxKernelLauncher(T* qkbuf /* score of qxk */,
                               const float* maskp /* padding mask */,
                               int32_t batch, int32_t nhead, int32_t xseql,
                               int32_t cstep, int32_t xlogn,
                               cudaStream_t stream);

template <typename T>
void DecoderSoftmaxKernelLauncher(T* qk_buf, const float* mask, int batch_size,
                                  int beam_size, int num_heads, int seq_len,
                                  int step, int input_len, cudaStream_t stream);
template <typename T>
void AddMaskLauncher(T* score, const float* mask, int batch_size, int beam_size,
                     int num_heads, int enc_seq_len, cudaStream_t stream);

template <typename T>
void GetBatchArrayLauncher(T* q, T* k, T* v, T* score, T* out, T** q_array,
                           T** k_array, T** v_array, T** score_array,
                           T** out_array, int batch_size, int beam_size,
                           int num_heads, int size_per_head, int step,
                           int q_stride, int kv_stride, int score_stride,
                           int out_stride, cudaStream_t stream);
template <typename T>
void MultiQueryGetBatchArrayLauncher(
    T* q, T* k, T* v, T* score, T* out, T** q_array, T** k_array, T** v_array,
    T** score_array, T** out_array, int batch_size, int beam_size,
    int num_heads, int size_per_head, int group_num, int step, int q_stride,
    int kv_stride, int score_stride, int out_stride, cudaStream_t stream);
template <typename T>
void BatchGemmWraper(void** matrix_C, void** matrix_A, void** matrix_B, int m,
                     int n, int k, bool transA, bool transB, float alpha,
                     float beta, int lda, int ldb, int ldc, int batch,
                     cublasHandle_t handle);
void BatchGemmI8Wrapper(void** matrix_C, void** matrix_A, void** matrix_B,
                        int m, int n, int k, bool transA, bool transB,
                        int alpha, int beta, int lda, int ldb, int ldc,
                        int batch, cublasHandle_t handle);
template <typename T0, typename T1>
void CastKernelLauncher(const T0* in, T1* out, int size, cudaStream_t stream);

template <typename T>
void TransMaskKernelLauncher(T* out, const int64_t* mask, int batch_size,
                             int seq_length, bool for_decoder, bool blank,
                             cudaStream_t stream);
template <typename T>
void RelativePEKernelLauncher(T* out, const T* attention_bias, int batch_size,
                              int seq_length, int k, int step, bool is_decoder,
                              cudaStream_t stream);
template <typename T>
void ALiBiPEKernelLauncher(T* out, int* batch_offset, int batch_size,
                           int seq_length, int num_heads, int ori_num_heads,
                           int step, int rank, cudaStream_t stream);
template <typename T>
void AddPostionEmbeddingKernelLauncher(T* score, const T* position_embedding,
                                       int batch_size, int num_heads, int step,
                                       int max_step, cudaStream_t cu_stream);

template <typename T>
void TopKKernelLauncher(T* output, int* output_indices, const T* input,
                        int batch_size, int length, int64_t k,
                        cudaStream_t stream);
template <typename T>
void TopKRadixKernelLauncher(T* output, int* output_indices, const T* input,
                             void* workspace, int batch_size, int length,
                             int64_t k, cudaStream_t stream);
template <typename T>
void TopKRadixGetWorkspaceSize(size_t* sizeInBytes, int batch_size, int length);

template <typename T>
void TopPSoftmaxLauncher(int* topp_count, T* topp_probs, int* topp_indices,
                         const T* input_logits, const float* p_values,
                         const float* temperatures, T* temp_probs,
                         void* workspace, size_t ws_size_in_bytes,
                         int batch_size, int length, bool input_is_sorted,
                         hiednnCudaHandle_t handle, cudaStream_t stream);
template <typename T>
void TopPSoftmaxGetWorkspaceSize(size_t* sizeInBytes, int batch_size,
                                 int length, bool input_is_sorted);

template <typename T>
void SampleKernelLauncher(int64_t* out, void* states, T* in, const int* indice,
                          int batch_size, int* num_arr, int stride,
                          cudaStream_t stream, void* device_prop);
void SampleKernelInitLauncher(void* state, unsigned long long seed,
                              int batch_size, cudaStream_t stream);
template <typename T>
void SampleTorchKernelLauncher(int64_t* out, std::vector<void*>& states, T* in,
                               const int* indice, int batch_size, int* num_arr,
                               int stride, cudaStream_t stream,
                               void* device_prop);
void SampleTorchKernelInitLauncher(void* state, unsigned long long seed,
                                   int batch_size, cudaStream_t stream);
template <typename T>
void UpdateKVLauncher(T* k, T* v, const T* step_k, const T* step_v,
                      int batch_size, int step, int max_length, int hidden_size,
                      int seq_len, int stride, cudaStream_t stream);
template <typename T>
void PreProcessForGeneration(T* dec_ids, T* max_dec_ids, const T* in_ids, T bos,
                             int batch_size, int num_beam, int max_len,
                             int seq_len, cudaStream_t stream);
template <typename T>
void UpdateId(T* out, const T* in, const int* beam_idx, T* tmp_id,
              int batch_size, int beam_size, int* step_list, int max_length,
              int seq_len, cudaStream_t stream);
template <typename T>
void PostProcessId(T* out_ids, const T* in_ids, int batch_size, int in_stride,
                   int out_stride, cudaStream_t stream);

// beam search
template <typename T>
void BeamScoreInitLauncher(T* beam_score, T* hyps_beam_score,
                           int64_t* hyps_beam_idx, int* eos_count,
                           int batch_size, int beam_size, cudaStream_t stream);
template <typename T>
void AddScoreLauncher(T* next_score, const T* beam_score, int batch, int length,
                      cudaStream_t stream);
template <typename T>
void UpdateBeamScoreLauncher(T* beam_score, int64_t* beam_next_token,
                             int* beam_idx, const T* topk_score,
                             const int* topk_indice, int batch_size,
                             int beam_size, int vocab_size, int eos,
                             int* eos_count, int* hyps_beam_idx,
                             T* hyps_beam_score, int64_t* hyps_ids,
                             int64_t* in_ids, int cur_len, int loop_len,
                             float length_penalty, cudaStream_t stream);
template <typename T>
void ReorderKVCacheLauncher(T* k, T* v, const T* old_k, const T* old_v,
                            const int* beam_idx, int batch_size, int beam_size,
                            int inner_dim, cudaStream_t stream);

template <typename T>
void LogitsProcessor(T* score, const int64_t* in_ids, int batch_size,
                     int max_len, int vocab_size, BatchGencfg batch_gencfg,
                     void* ws_ptr, size_t ws_bytes, cudaStream_t stream);
template <typename T>
void UpdateHypsIds(T* out, const T* before_ids, const T* cur_id,
                   const int* beam_idx, int batch_size, int step,
                   int max_length, int beam_size, cudaStream_t stream);
template <typename T>
void transpose_axis_01_kernelLauncher(T* out, T* in, const int dim0,
                                      const int dim1, const int dim2,
                                      cudaStream_t stream);
template <typename T>
void transpose_axis_12_kernelLauncher(T* out, T* in, const int dim0,
                                      const int dim1, const int dim2,
                                      cudaStream_t stream);
#ifdef ENABLE_SPARSE
template <typename T>
void CuSparseGemmCSC(T* matrix_C, const T* matrix_A, const int* col_offset,
                     const int* row_indices, const T* val, int nnz,
                     const T* bias, int m, int n, int k, bool transA,
                     bool transB, int lda, int ldb, int ldc, float alpha,
                     float beta, int batch, cudaStream_t stream);
template <typename T>
void spmm_pattern0(const T* A, const T* B, const int* cscColOffset,
                   const int* cscRowInd, T* C, const int M, const int N,
                   const int K, const int nnz, cudaStream_t stream);
template <typename T>
void spmm_pattern1(const T* A, const T* B, const unsigned short* cscRowInd,
                   T* C, const int M, const int N, const int K, const int nnz,
                   cudaStream_t stream);
#endif
template <typename T>
void CopyMatrix(const int M, const int N, const T* A, const int lda, T* B,
                const int ldb, cudaStream_t stream);
template <typename T>
void CopyMultiBeam(void* A, void* B, int batch_size, int beam_size,
                   int inner_dim, cudaStream_t stream);
template <typename T>
void BatchSoftmaxKernelLauncher(T* input, int* len_list, int batch_size,
                                int stride, float temperature,
                                cudaStream_t stream);

template <typename FType, typename QType, typename ComputeType = float>
void QuantizePerChannelImp(const FType* fdata_ptr, QType* qdata_ptr,
                           ComputeType* scale_ptr, QType* zero_point_ptr,
                           int* redsum_ptr, const int inner_len,
                           const int outer_len, cudaStream_t stream);
template <typename FType, typename QType, typename ComputeType = float>
void DeQuantizePerChannelImp(FType* fdata_ptr, const QType* qdata_ptr,
                             const ComputeType* scale_ptr,
                             const ComputeType* zero_point_ptr,
                             const int* redsum_ptr, const int inner_len,
                             const int outer_len, cudaStream_t stream);
void GemmInt8(int32_t* matrix_C, const int8_t* matrix_A, const int8_t* matrix_B,
              int m, int n, int k, bool transA, bool transB, int lda, int ldb,
              int ldc, int alpha, int beta, cublasHandle_t handle,
              cudaStream_t stream);
// #if __CUDA_ARCH__ >= 720 && CUDART_VERSION >= 10020
void GemmHIEInt8(int32_t* matrix_C, const int8_t* matrix_A,
                 const int8_t* matrix_B, int m, int n, int k,
                 cudaStream_t stream);
// #endif
template <typename FType, typename QType, typename Active>
void postProcessImp(const int lhs_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
                    const float* lhs_scale, const QType* lhs_zero,
                    const int* lhs_redsum, const int rhs_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
                    const float* rhs_scale, const QType* rhs_zero,
                    const int* rhs_redsum, const int bias_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
                    const FType* bias, const int elemwiseA_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims,
                    const int* elemwiseA, const int elemwiseB_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
                    const FType* elemwiseB, const int out_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& out_dims,
                    FType* output_ptr, const FType& alpha, const FType& beta,
                    const int& K, cudaStream_t& stream, Active active);
template <typename T>
void RotaryEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                     int batch, int seq_len, int head, int size_per_head,
                     int step, int stride, cudaStream_t cu_stream);
template <typename T>
void RotaryPctEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                        int batch, int seq_len, int head, int size_per_head,
                        int step, int stride, float pct,
                        cudaStream_t cu_stream);
template <typename T>
void RotaryEmbedding2D(T* output, T* input, float* inv_freq, int* batch_offset,
                       int batch, int seq_len, int head, int size_per_head,
                       int step, int stride, int input_len,
                       cudaStream_t cu_stream);
template <typename T>
void RotaryEmbeddingHalfInner(T* output, T* input, float* inv_freq,
                              int* batch_offset, int batch, int seq_len,
                              int head, int size_per_head, int* step_list,
                              int stride, cudaStream_t cu_stream);
template <typename T>
void RotaryOptEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                        int batch, int seq_len, int head, int size_per_head,
                        int* step_list, int stride, int xlogn, int* positions,
                        int mrope_size, int* mrope_section,
                        cudaStream_t cu_stream);
template <typename T>
void Chunk(T* output, T* input, int batch, int seq_len, int hidden_size,
           int chunk_split, cudaStream_t cu_stream);
template <typename T>
void ChunkBinary(T* output, T* input, int batch, int seq_len, int hidden_size,
                 int chunk_split, int type, cudaStream_t cu_stream);
template <typename T>
void CalcTopP(const T* input, int* k_arr, float* p_arr, int batch, int length,
              cudaStream_t cu_stream);

template <template <class> class ActiveFun>
void hgemm_32x128x16_simt_Aldg1(const half* A, const half* B, const half* bias,
                                half* C, const uint32_t M, const uint32_t N,
                                const uint32_t K, const uint32_t sm_count,
                                void* workspace, cudaStream_t stream);

// strided softmax
template <typename T>
void StridedSoftmaxGetWorkspaceSize(size_t* wsInBytes, int taskNum, int stride);
template <typename T>
void StridedSoftmaxLauncher(T* y, const T* x, const int* taskLenPtr,
                            const float* temperatures, void* workspace,
                            size_t wsSizeInBytes, int taskNum, int stride,
                            cudaStream_t stream);
template <typename T>
void StridedLogSoftmaxLauncher(T* y, const T* x, const int* taskLenPtr,
                               const float* temperatures, void* workspace,
                               size_t wsSizeInBytes, int taskNum, int stride,
                               cudaStream_t stream);

// deprecated
#if 0
// span attention
template <typename T>
class SpanAttention;
template <typename T>
void SpanAttentionCreate(SpanAttention<T>** obj, int batchSize, int nHeads,
                         int nGroups, int headSize, int spanLen,
                         int nSpansPerBatch, int seqLen, int deviceId);
template <typename T>
void SpanAttentionDestroy(SpanAttention<T>* obj);
template <typename T>
void SpanAttentionGetWorkspaceSize(const SpanAttention<T>* obj,
                                   size_t* wsInBytes);
template <typename T>
void SpanAttentionLauncher(const SpanAttention<T>* obj, T* output,
                           const T* query, const void* const* kSpanArray,
                           const void* const* vSpanArray, float QKScale,
                           void* workspace, size_t wsSizeInBytes,
                           cudaStream_t stream);
#endif

// ReplaceEmbedding
template <typename T>
void ReplaceEmbedding(T* output, float* input, int64_t nbytes,
                      cudaStream_t stream);
template <typename T>
void ToFloatKernelLauncher(float* out, const T* in, int64_t count,
                           cudaStream_t stream);

template <typename CT, typename DT>
size_t GetBatchedGEMVWorkspaceSize(int k, int n, int batchSize);
template <typename CT, typename DT>
int BatchedGEMV(int k, int n, int batchSize, const DT* const* xArray,
                const DT* const* matrixArray, int ldMatrix, DT* const* yArray,
                void* ws, size_t wsSize, cudaStream_t stream);
template <typename T>
void SelectBatchTokenLogprobLauncher(T* typed_logprobs,
                                     float* float_token_logprobs,
                                     int64_t* out_tokens, int batch_size,
                                     int length, cudaStream_t stream);
template <typename T>
void CalcExpertKernelLauncher(T* output, T* input, T* expert_weight,
                              int total_token, int hidden_size, int num_expert,
                              cudaStream_t stream);
void ReorderAndPaddingMOE(int64_t* experts_idx, int64_t* experts_seq,
                          int64_t* indice_source, int* input, int batch,
                          int num_experts, int top_k, int block_size,
                          int* total_token_post_pad, cudaStream_t stream);
template <typename T>
void GetReorderData(T* reorder_data, T* input, int64_t* experts_idx,
                    int64_t* experts_seq, int* total_tokens_post_pad,
                    int max_total_tokens, int padding_val, int topk,
                    int hidden_size, int block_size, cudaStream_t stream);
template <typename T>
void MOEGetBatchArrayLauncher(int64_t* experts_idx, int* total_tokens_post_pad,
                              T* data, void** data_array, int max_block,
                              int layout_size, int block_size,
                              cudaStream_t stream);
template <typename T>
void MulAndSilu(T* output, T* gate_out, T* up_proj_out, int m, int n,
                cudaStream_t stream);
template <typename T>
void FinalizeMoeRoutingKernelLauncher(
    T* output, T* fianl_result, float* experts_score, int64_t* indice_source,
    int* expert_for_source_row, int* total_tokens_pad_ptr, int total_token,
    int top_k, int hidden_size, cudaStream_t stream);
template <typename T>
void FinalizeMoeRoutingNewKernelLauncher(T* output, T* fianl_result,
                                         float* experts_score,
                                         int* mid_row_indices,
                                         int* final_row_indices,
                                         int total_token, int top_k,
                                         int hidden_size, cudaStream_t stream);
template <typename T>
void RotaryMultimodalSections(T* output, T* input, float* inv_freq, int batch,
                              int seq_len, int head, int size_per_head,
                              int stride, int* positions, int mrope_size,
                              int* mrope_section, cudaStream_t stream);
template <typename T>
void SoftmaxLowReduceKernelLauncher(const T* input, T* output, int out_dim,
                                    int inner_dim, cudaStream_t stream);
void GetExpertByIndice(int* expert_indices, const int* in_expert_indices,
                       const int* row_indices, int total_token, int topk,
                       int num_expert, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
