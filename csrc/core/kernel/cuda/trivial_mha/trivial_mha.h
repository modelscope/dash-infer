/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    trivial_mha.h
 */

#pragma once
#include "core/kernel/kernel.h"

#ifdef ENABLE_CUDA
#include "cuda_runtime.h"

namespace allspark {
namespace cuda {

class TrivialMHAParam {
 public:
  DataType dtype;
  size_t maxlen;
  size_t batch, nhead, phead, seqlen;

  size_t workspace_inbytes() const;
  size_t score_usage_inbytes() const;
  size_t arrayptr_usage_inbytes() const;

  static size_t roundx(size_t r, size_t x) { return ((x + r - 1) / r) * r; }
};

using trivial_t = TrivialMHAParam;

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
                        const cudaStream_t& stream);

/// @brief Better warpper of trivial attention kernel for prefill phase only.
void trivial_prefill_attention(
    const TrivialMHAParam& param, const cublasHandle_t& stream_bind_cublas,
    const cudaStream_t& stream,
    const void* concat,  // batch, seqlen, 3, nhead, phead
    const float* mask,   // batch, seqlen, seqlen,
    void* output,        // batch, seqlen, nhead, pheads
    void* kcache,        // kvcache is continuous. kv input.
    void* vcache,        // batch, maxlen, nhead, phead, see UpddateKVLauncher()
    void* workspace, size_t beams, float alpha);

}  // namespace cuda
}  // namespace allspark

#endif  // ENABLE_CUDA
