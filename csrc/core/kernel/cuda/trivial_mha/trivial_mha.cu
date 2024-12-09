/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    trivial_mha.cu
 */

#include "trivial_mha.h"

#ifdef ENABLE_CUDA
namespace allspark {
namespace cuda {

size_t TrivialMHAParam::workspace_inbytes() const {
  return score_usage_inbytes() + 5 * arrayptr_usage_inbytes();
}

size_t TrivialMHAParam::score_usage_inbytes() const {
  return roundx(32, SizeofType(dtype) * batch * maxlen * nhead * maxlen);
}

size_t TrivialMHAParam::arrayptr_usage_inbytes() const {
  return roundx(32, sizeof(void*) * batch * nhead);
}

template <typename T>
void gpu_dec_single_mha_template(
    void* out, void* score, const void* query, const float* mask,
    const void* position_embedding, const void* k_cache, const void* v_cache,
    void** q_array, void** k_array, void** v_array, void** score_array,
    void** out_array, int batch_size, int beam_size, int seq_len, int step,
    int cache_max_len, int hidden_size, int num_heads, int size_per_head,
    int gemm_batch, float alpha, bool xlogn_enable, int xlogn_len,
    const cublasHandle_t& stream_bind_cublas, const cudaStream_t& stream) {
  auto is_prefix = [&]() -> bool { return seq_len != 1; };
  if (seq_len != 1) {
    step = seq_len;
  }
  int q_stride = hidden_size * 3;
  int kv_stride = hidden_size;
  int out_stride = hidden_size;
  int score_stride = step * num_heads;
  cuda::GetBatchArrayLauncher(
      (T*)query, (T*)k_cache, (T*)v_cache, (T*)score, (T*)out, (T**)q_array,
      (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array, batch_size,
      1, num_heads, size_per_head, step, q_stride * seq_len,
      kv_stride * cache_max_len, score_stride * seq_len, out_stride * seq_len,
      stream);

  // batch gemm 1
  float qxk_scale = alpha;
  if (!is_prefix() && xlogn_enable && step > xlogn_len) {
    // logn logic, if decoder, query require log_xlogn(xseql) * alpha
    // scale if longer than model basic length (xlogn). for encoder /
    // prefix, this logices are implement in softmax kernel.
    qxk_scale *= logf(step) / logf(xlogn_len);
  }
  cuda::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                           size_per_head, false, true, qxk_scale, 0.0f,
                           q_stride, kv_stride, score_stride, gemm_batch,
                           stream_bind_cublas);
  if (position_embedding) {
    cuda::BinaryKernelLauncher((T*)score, (T*)score, (T*)position_embedding,
                               batch_size * num_heads * step * seq_len, 1,
                               stream);
  }
  if (is_prefix() && xlogn_enable) {
    // logn prefix
    if (beam_size != 1) {
      LOG(ERROR) << "Logn attention not support beam search. "
                 << "disgard input beam param = " << beam_size << "."
                 << std::endl;
    }
    cuda::LognSoftmaxKernelLauncher((T*)score, mask, batch_size, num_heads,
                                    seq_len, step, xlogn_len, stream);
  } else {
    cuda::SoftmaxKernelLauncher((T*)score, mask, batch_size, beam_size,
                                num_heads, seq_len, step, stream);
  }
  // batch gemm 2
  cuda::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                           size_per_head, step, false, false, 1.0f, 0.0f,
                           score_stride, kv_stride, out_stride, gemm_batch,
                           stream_bind_cublas);
}

/// @brief only for legacy compability
template <typename T>
void gpu_dec_single_mha_update_kv_template(
    void* out, void* score, const void* query, const void* key,
    const void* value, const float* mask, const void* position_embedding,
    void* k_cache, void* v_cache, void** q_array, void** k_array,
    void** v_array, void** score_array, void** out_array, int batch_size,
    int beam_size, int seq_len, int step, int cache_max_len, int hidden_size,
    int num_heads, int size_per_head, int gemm_batch, float alpha,
    bool xlogn_enable, int xlogn_len, const cublasHandle_t& stream_bind_cublas,
    const cudaStream_t& stream) {
  cuda::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                         (const T*)value, batch_size, step - 1, cache_max_len,
                         hidden_size, seq_len, 3 * hidden_size, stream);
  gpu_dec_single_mha_template<T>(
      out, score, query, mask, position_embedding, k_cache, v_cache, q_array,
      k_array, v_array, score_array, out_array, batch_size, beam_size, seq_len,
      step, cache_max_len, hidden_size, num_heads, size_per_head, gemm_batch,
      alpha, xlogn_enable, xlogn_len, stream_bind_cublas, stream);
}

/// @brief Better warpper of trivial attention kernel for prefill phase only.
void trivial_prefill_attention(
    const TrivialMHAParam& param, const cublasHandle_t& stream_bind_cublas,
    const cudaStream_t& stream,
    const void* concat,  // batch, seqlen, 3, nhead, phead
    const float* mask,   // batch, seqlen, seqlen,
    void* output,        // batch, seqlen, nhead, pheads
    void* kcache,        // kvcache is continuous.
    void* vcache,        // batch, maxlen, nhead, phead, see UpddateKVLauncher()
    void* workspace, size_t beams, float alpha) {
  size_t concat_offset_inbytes =
      param.nhead * param.phead * SizeofType(param.dtype);
  const char* query = reinterpret_cast<const char*>(concat);
  // const char* key   = query + concat_offset_inbytes;
  // const char* value = key   + concat_offset_inbytes;
  char* score = reinterpret_cast<char*>(workspace);
  size_t score_offset_inbytes = param.score_usage_inbytes();
  size_t arrayptr_offset_inbytes = param.arrayptr_usage_inbytes();
  void** qarr = reinterpret_cast<void**>(score + score_offset_inbytes +
                                         0 * arrayptr_offset_inbytes);
  void** karr = reinterpret_cast<void**>(score + score_offset_inbytes +
                                         1 * arrayptr_offset_inbytes);
  void** varr = reinterpret_cast<void**>(score + score_offset_inbytes +
                                         2 * arrayptr_offset_inbytes);
  void** sarr = reinterpret_cast<void**>(score + score_offset_inbytes +
                                         3 * arrayptr_offset_inbytes);
  void** oarr = reinterpret_cast<void**>(score + score_offset_inbytes +
                                         4 * arrayptr_offset_inbytes);
  // legacy:
  // gpu_dec_single_mha(param.dtype, output, score,
  //         query, key, value, mask, nullptr,
  //         kcache, vcache, qarr, karr, varr, sarr, oarr,
  //         param.batch, beams, param.seqlen, param.seqlen, param.maxlen,
  //         param.nhead * param.phead, param.nhead, param.phead, param.batch *
  //         param.nhead, alpha, false, 0, stream_bind_cublas, stream);

#define GPU_DEC_MHA(TEMP)                                                    \
  /*printf("gpu_dec_single_mha_template<" #TEMP                              \
     ">(batch=%d,beams=%d,nhead=%d,phead=%d,seqlen=%d,maxlen=%d,scale=%f)",  \
      param.batch, beams, param.nhead, param.phead, param.seqlen,            \
     param.maxlen, alpha);*/                                                 \
  gpu_dec_single_mha_template<TEMP>(                                         \
      output, score, query, mask, nullptr, kcache, vcache, qarr, karr, varr, \
      sarr, oarr, param.batch, beams, param.seqlen, param.seqlen,            \
      param.maxlen, param.nhead * param.phead, param.nhead, param.phead,     \
      param.batch * param.nhead, alpha, false, 0, stream_bind_cublas, stream);
  switch (param.dtype) {
    case DataType::FLOAT32:
      cublasSetMathMode(stream_bind_cublas, CUBLAS_PEDANTIC_MATH);
      GPU_DEC_MHA(float);
      break;
#ifdef ENABLE_FP16
    case DataType::FLOAT16:
      cublasSetMathMode(stream_bind_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
      GPU_DEC_MHA(half);
      break;
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    case DataType::BFLOAT16:
      // notice: default bf16 not good enough.
      cublasSetMathMode(stream_bind_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
      GPU_DEC_MHA(hie::bfloat16);
      break;
#endif  // ENABLE_BF16
    default: {
      LOG(ERROR) << "unsupported datatype " << DataType_Name(param.dtype)
                 << " for CUDA dispatch";
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }
#undef GPU_DEC_MHA
}

/// @brief only for legacy compability
void gpu_dec_single_mha(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int gemm_batch, float alpha, bool xlogn_enable,
                        int xlogn_len, const cublasHandle_t& stream_bind_cublas,
                        const cudaStream_t& stream) {
  // clear previous errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(WARNING) << "gpu_dec_single_mha: previous error cleared: "
                 << cudaGetErrorString(err);
  }
  if (dtype == FLOAT32) {
    cublasSetMathMode(stream_bind_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
  }

#define GPU_DEC_MHA(TEMP)                                                     \
  gpu_dec_single_mha_update_kv_template<TEMP>(                                \
      out, score, query, key, value, mask, position_embedding, k_cache,       \
      v_cache, q_array, k_array, v_array, score_array, out_array, batch_size, \
      beam_size, seq_len, step, cache_max_len, hidden_size, num_heads,        \
      size_per_head, gemm_batch, alpha, xlogn_enable, xlogn_len,              \
      stream_bind_cublas, stream);
  switch (dtype) {
    case DataType::FLOAT32:
      GPU_DEC_MHA(float);
      break;
#ifdef ENABLE_FP16
    case DataType::FLOAT16:
      GPU_DEC_MHA(half);
      break;
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16:
      GPU_DEC_MHA(hie::bfloat16);
      break;
#endif
    default: {
      LOG(ERROR) << "unsupported datatype " << DataType_Name(dtype)
                 << " for CUDA dispatch";
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }
#undef GPU_DEC_MHA
}

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
