/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    attn_kv_cache.hpp
 */

#ifndef __MHA_QUANT_CACHE__ATTN_KV_CACHE_HPP__
#define __MHA_QUANT_CACHE__ATTN_KV_CACHE_HPP__

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
enum QuantType : int32_t { INT8 = 0, UINT4 = 1 };
}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark

#ifdef ENABLE_CUDA
#include "hie_bfloat16.hpp"
namespace allspark {
namespace cuda {
namespace mha_quant_cache {
template <typename FT, typename QT>
void load_and_quant_to_kv_cache_context(cudaStream_t stream, const FT* qkv,
                                        QT* kc, float* kz, float* ks, QT* vc,
                                        float* vz, float* vs, int32_t batch,
                                        int32_t nhead, int32_t phead,
                                        int32_t cache, int32_t xseql,
                                        QuantType quant_type);

template <typename FT, typename QT>
void load_and_quant_to_kv_cache_decoder(cudaStream_t stream, const FT* qkv,
                                        QT* kc, float* kz, float* ks, QT* vc,
                                        float* vz, float* vs, int32_t batch,
                                        int32_t nhead, int32_t phead,
                                        int32_t cache, int32_t offset,
                                        QuantType quant_type);

template <typename FT>
void load_and_quant_to_i8_query_decoder(cudaStream_t stream, const FT* qkv,
                                        int8_t* qq, float* qz, float* qr,
                                        float* qs, int32_t batch, int32_t nhead,
                                        int32_t phead);

template <typename FT, typename QT>
void score_gemv_w8_position_embedding(
    cudaStream_t stream, const FT* qkv, const float* mask,
    const float* position_embedding, const QT* kc, const float* kz,
    const float* ks, FT* score, float alpha, int32_t batch, int32_t nhead,
    int32_t phead, int32_t cache, int32_t xseql, int32_t inlen,
    QuantType quant_type);

template <typename FT, typename QT>
void context_gemv_w8(cudaStream_t stream, const FT* score, const QT* vcache,
                     const float* vzero, const float* vscale, FT* context,
                     int32_t batch, int32_t nhead, int32_t phead, int32_t cache,
                     int32_t xseql, QuantType quant_type);

template <typename FT>
void inplace_softmax(cudaStream_t stream, FT* score, int32_t batch,
                     int32_t nhead, int32_t xseql);

}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA

#endif  // __MHA_QUANT_CACHE__ATTN_KV_CACHE_HPP__
