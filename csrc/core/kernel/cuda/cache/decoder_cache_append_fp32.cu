/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    decoder_cache_append_fp32.cu
 */

#include "decoder_cache_append.cuh"

namespace allspark {
namespace cuda {

template void DecoderCacheAppendLauncher(
    void* const* kSpanArray, void* const* vSpanArray, float* queryOut,
    const float* src, const uint32_t* oldSeqLens, int batchSize, int nHeads,
    int nGroups, int headSize, int spanLen, int nSpansPerBatch,
    span::QuantMode cacheMode, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
