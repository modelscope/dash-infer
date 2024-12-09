/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax.hpp
 */

#pragma once
#include <cuda/cuda_common.h>

#include "library_types.h"

namespace allspark {
namespace cuda {
namespace attention {
template <typename T>
struct toDataType {
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_type = cudaDataType::CUDA_R_32F;
#endif  // __LIBRARY_TYPES_H__
};

template <>
struct toDataType<float> {
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_type = cudaDataType::CUDA_R_32F;
#endif  // __LIBRARY_TYPES_H__
};

#ifdef ENABLE_FP16
template <>
struct toDataType<half> {
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_type = cudaDataType::CUDA_R_16F;
#endif  // __LIBRARY_TYPES_H__
};
#endif

#ifdef ENABLE_BF16
using bfloat16 = hie::bfloat16;
template <>
struct toDataType<bfloat16> {
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_type = cudaDataType::CUDA_R_16BF;
#endif  // __LIBRARY_TYPES_H__
};
#endif
}  // namespace attention

// fallback
void softmax_4d_fallback(
    cudaStream_t stream, cudaDataType_t dt, const void* input,
    void* output,       //  [datah, datam, datal, align]
    const void* masks,  //  [maskh, maskm, maskl, maska]
    float alpha,        //  scaling factor, a.k.a 1/sqrt(d)
    int32_t datah, int32_t datam, int32_t datal,
    int32_t align,  // loop on align dim
    int32_t maskh, int32_t maskm, int32_t maskl,
    int32_t maska,  // if beam search, make sure beams = datah / maskh, other
                    // dim, defatult set 1, not 0.
    /* features flag and params below this line */
    bool mask_decoder,  // decoder style mask, mask l-dim, only use maskl - 1
                        // index.
    bool mask_with_10,  // mask with 1 or 0
    bool log_softmax,   // fuse log after softmax
    bool attn_logn, int32_t lognm,
    int32_t lognl  // query logn scale before softmax, logn only on single dim,
                   // and the other dim should be 0
);

void softmax_4d_test_only(
    cudaStream_t stream, cudaDataType_t dt, const void* input,
    void* output,       //  [datah, datam, datal, align]
    const void* masks,  //  [maskh, maskm, maskl, maska]
    float alpha,        //  scaling factor, a.k.a 1/sqrt(d)
    int32_t datah, int32_t datam, int32_t datal,
    int32_t align,  // loop on align dim
    int32_t maskh, int32_t maskm, int32_t maskl, int32_t maska,
    /* features flag and params below this line */
    bool mask_decoder,  // decoder style mask, maskl dim keep use with maskl - 1
                        // index.
    bool mask_with_10,  // mask with 1 or 0
    bool log_softmax,   // fuse log after softmax
    bool attn_logn, int32_t lognm,
    int32_t lognl  // query logn scale before softmax, logn only on single dim,
                   // and the other dim should be 0
);

// unroll
bool softmax_unroll_nomask_valid(cudaDataType_t dt, float alpha, int data_batch,
                                 int data_xseql, int data_align,
                                 bool enable_attn_logn, int base_logn,
                                 bool enable_fuse_log);

void softmax_unroll_nomask(cudaStream_t stream, cudaDataType_t dt,
                           const void* input,
                           void* output,  //  [batch, xseql, align]
                           float alpha, int data_batch, int data_xseql,
                           int data_align, bool enable_attn_logn, int base_logn,
                           bool enable_fuse_log);

bool softmax_unroll_3dmask_valid(cudaDataType_t dt, float alpha, int data_batch,
                                 int data_xseql, int data_nhead, int data_align,
                                 int mask_batch, int mask_xseql, int mask_align,
                                 bool decoder_layout_mask, bool mask_with_10,
                                 bool enable_attn_logn, int base_logn);

void softmax_unroll_3dmask(
    cudaStream_t stream, cudaDataType_t dt, const void* input,
    void* output,       //  [batch, xseql, nhead, align]
    const void* masks,  //  [batch, xseql,        align]
    float alpha, int data_batch, int data_xseql, int data_nhead, int data_align,
    int mask_batch, int mask_xseql, int mask_align,
    bool decoder_layout_mask,  //  [batch, inlen,     1, inlen]
    bool mask_with_10, bool enable_attn_logn, int base_logn);

}  // namespace cuda
}  // namespace allspark
