/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_sm80.cuh
 */

#pragma once

#include "attn/common/config.h"
#include "attn/common/dispatch.hpp"
#include "attn/qk/qk.hpp"
#include "attn/qk/sm80/launcher.cuh"
#include "common/check_cuda.h"

namespace span {

template <typename FType>
struct QKLauncher<SaArch::SM80, FType> {
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      QuantMode qMode, float QKScale, int stride, int nGroups,
                      int nChunk, int chunkSize, int headSize,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    return DispatchCacheMode<FType, detail::QKGemmSm80>(
        qMode, chunkSize, headSize, headsPerGroup, Q, KCachePtrs, QK,
        seqLengths, qkMappingWs, QKScale, stride, nGroups, nChunk,
        headsPerGroup, seqBlocks, stream);
  }
};

}  // namespace span
