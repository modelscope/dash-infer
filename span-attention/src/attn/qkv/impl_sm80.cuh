/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_sm80.cuh
 */

#pragma once

#include "attn/common/config.h"
#include "attn/common/dispatch.hpp"
#include "attn/qkv/qkv.hpp"
#include "attn/qkv/sm80/launcher.cuh"
#include "common/check_cuda.h"

namespace span {

template <typename FType>
struct QKVWorkspaceBytes<SaArch::SM80, FType> {
  SaStatus operator()(size_t* qkvWorkspaceBytes, int batch, int nHead,
                      int stride, int headSize, int seqTileSize) const {
    uint32_t maxQKVSeqBlocks = U32DivRU(stride, seqTileSize);
    *qkvWorkspaceBytes = AlignedMemSize<MEM_ALIGN>(
        batch * nHead * maxQKVSeqBlocks * headSize * sizeof(FType));
    return SaStatus::SUCCESS;
  }
};

template <typename FType>
struct QKVLauncher<SaArch::SM80, FType> {
  SaStatus operator()(const FType* QK, const void* const* VCachePtrs,
                      FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, QuantMode qMode, int stride,
                      int batch, int nGroups, int nChunk, int chunkSize,
                      int headSize, int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    return DispatchCacheMode<FType, detail::QKVGemmSm80>(
        qMode, chunkSize, headSize, headsPerGroup, QK, VCachePtrs, QKVReduceWs,
        O, seqLengths, qkvMappingWs, stride, batch, nGroups, nChunk,
        headsPerGroup, seqBlocks, stream);
  }
};

}  // namespace span
