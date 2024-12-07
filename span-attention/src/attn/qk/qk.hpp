/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qk.hpp
 */

#pragma once

#include <span_attn.h>

#include "attn/common/arch.h"
#include "common/logger.h"

namespace span {
template <SaArch Arch, typename FType>
struct QKLauncher {
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      QuantMode qMode, float QKScale, int stride, int nGroups,
                      int nChunk, int chunkSize, int headSize,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const;
};

template <>
struct QKLauncher<SaArch::SM80, float> {
  SaStatus operator()(const float* Q, const void* const* KCachePtrs, float* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      QuantMode qMode, float QKScale, int stride, int nGroups,
                      int nChunk, int chunkSize, int headSize,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    LOG(ERROR) << "SM80 QK GEMM does not support float as FType" << std::endl;
    return SaStatus::INTERNAL_ERROR;
  }
};
}  // namespace span
