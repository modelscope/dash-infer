/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qkv.hpp
 */

#pragma once

#include <span_attn.h>

#include "attn/common/arch.h"
#include "common/logger.h"

namespace span {

template <SaArch Arch, typename FType>
struct QKVWorkspaceBytes {
  SaStatus operator()(size_t* qkvWorkspaceBytes, int batch, int nHead,
                      int stride, int headSize, int seqTileSize) const;
};

template <SaArch Arch, typename FType>
struct QKVLauncher {
  SaStatus operator()(const FType* QK, const void* const* VCachePtrs,
                      FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, QuantMode qMode, int stride,
                      int batch, int nGroups, int nChunk, int chunkSize,
                      int headSize, int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const;
};

template <>
struct QKVWorkspaceBytes<SaArch::SM80, float> {
  SaStatus operator()(size_t* qkvWorkspaceBytes, int batch, int nHead,
                      int stride, int headSize, int seqTileSize) const {
    LOG(ERROR) << "SM80 QKV GEMM does not support float as FType" << std::endl;
    return SaStatus::INTERNAL_ERROR;
  }
};

template <>
struct QKVLauncher<SaArch::SM80, float> {
  SaStatus operator()(const float* QK, const void* const* VCachePtrs,
                      float* QKVReduceWs, float* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, QuantMode qMode, int stride,
                      int batch, int nGroups, int nChunk, int chunkSize,
                      int headSize, int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    LOG(ERROR) << "SM80 QKV GEMM does not support float as FType" << std::endl;
    return SaStatus::INTERNAL_ERROR;
  }
};

}  // namespace span
