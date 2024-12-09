/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_simt.cuh
 */

#pragma once

#include "attn/common/config.h"
#include "attn/common/dispatch.hpp"
#include "attn/qk/qk.hpp"
#include "attn/qk/qk_gemv.cuh"
#include "common/check_cuda.h"

namespace span {

template <typename FType>
struct QKLauncher<SaArch::SIMT, FType> {
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      QuantMode qMode, float QKScale, int stride, int nGroups,
                      int nChunk, int chunkSize, int headSize,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    return DispatchCacheMode<FType, detail::QKGemv>(
        qMode, chunkSize, headSize, headsPerGroup, Q, KCachePtrs, QK,
        seqLengths, qkMappingWs, QKScale, stride, nGroups, nChunk,
        headsPerGroup, seqBlocks, stream);
  }
};

}  // namespace span
