/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    decoder_cache_append_bf16.cu
 */

#ifdef ENABLE_BF16
#include "decoder_cache_append.cuh"
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"

namespace allspark {
namespace cuda {

template void DecoderCacheAppendLauncher(
    void* const* kSpanArray, void* const* vSpanArray, hie::bfloat16* queryOut,
    const hie::bfloat16* src, const uint32_t* oldSeqLens, int batchSize,
    int nHeads, int nGroups, int headSize, int spanLen, int nSpansPerBatch,
    span::QuantMode cacheMode, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
