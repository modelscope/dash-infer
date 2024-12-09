/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    launcher.cuh
 */

#pragma once

#include "attn/common/config.h"
#include "block_softmax.cuh"
#include "common/check_cuda.h"
#include "common/logger.h"
#include "tiled_softmax.cuh"

namespace span {

namespace detail {

template <int BLOCK_X, int BLOCK_Y, int UNROLL, typename FType>
SaStatus LaunchSoftmaxKernel(FType* QK, const uint32_t* seqLengths,
                             uint32_t batchStride, uint32_t softmaxBatch,
                             int nHead, cudaStream_t stream) {
  using Config = SoftmaxConfig<FType>;
  using SoftmaxCompT = typename Config::ComputeType;

  uint32_t grid = U32DivRU(softmaxBatch, BLOCK_Y);
  uint32_t block = BLOCK_X * BLOCK_Y;
  U32DivMod nHeadDivMod(nHead);
  LOG(DEBUG) << "InplaceSoftmaxKernel with grid " << grid << " block " << block
             << std::endl;
  InplaceSoftmaxKernel<BLOCK_X, BLOCK_Y, UNROLL, FType, SoftmaxCompT>
      <<<grid, block, 0, stream>>>(QK, seqLengths, nHeadDivMod, softmaxBatch,
                                   batchStride);
  SA_CHECK_KERNEL_RET();
  return SaStatus::SUCCESS;
}

template <int BLOCK, int UNROLL, int MAX_NTILES_PER_TASK, typename FType>
SaStatus TiledSoftmaxDispatchTile(FType* QK, const uint32_t* seqLengths,
                                  const void* softmaxMappingWs, void* softmaxWs,
                                  size_t softmaxReduceWsSize,
                                  size_t softmaxReduceFinishCountSize,
                                  uint32_t totalTiles, uint32_t batchStride,
                                  uint32_t maxLength, int nHead, int smCount,
                                  cudaStream_t stream) {
  using Config = SoftmaxConfig<FType>;
  using SoftmaxCompT = typename Config::ComputeType;
  uint32_t maxNTilesPerTask = U32DivRU(maxLength, BLOCK * UNROLL);

  int maxBlocksPerSM;
  SA_CHECK_CUDA_RET(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSM,
      TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES_PER_TASK, FType,
                                SoftmaxCompT>,
      BLOCK, 0));

  // prevent dead lock
  if (maxNTilesPerTask > static_cast<uint32_t>(MAX_NTILES_PER_TASK) ||
      maxNTilesPerTask > static_cast<uint32_t>(maxBlocksPerSM * smCount)) {
    LOG(ERROR) << "too many tiles required by TiledInplaceSoftmaxKernel: "
               << maxNTilesPerTask << std::endl;
    return SaStatus::EXCEED_LIMIT_ERROR;
  }

  // workspace
  char* wsPtr = static_cast<char*>(softmaxWs);
  SoftmaxCompT* maxReduceWs = reinterpret_cast<SoftmaxCompT*>(wsPtr);
  SoftmaxCompT* sumReduceWs =
      reinterpret_cast<SoftmaxCompT*>(wsPtr + softmaxReduceWsSize);
  uint32_t* maxReduceFinishCount =
      reinterpret_cast<uint32_t*>(wsPtr + 2 * softmaxReduceWsSize);
  uint32_t* sumReduceFinishCount = reinterpret_cast<uint32_t*>(
      wsPtr + 2 * softmaxReduceWsSize + softmaxReduceFinishCountSize);

  // set the finish count workspace to 0
  SA_CHECK_CUDA_RET(cudaMemsetAsync(wsPtr + 2 * softmaxReduceWsSize, 0,
                                    2 * softmaxReduceFinishCountSize, stream));

  LOG(DEBUG) << "TiledInplaceSoftmaxKernel with grid (" << totalTiles << ", "
             << nHead << ")  block " << BLOCK << std::endl;
  TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES_PER_TASK, FType,
                            SoftmaxCompT>
      <<<dim3(totalTiles, nHead), BLOCK, 0, stream>>>(
          QK, maxReduceWs, maxReduceFinishCount, sumReduceWs,
          sumReduceFinishCount, seqLengths,
          static_cast<const span::TileMapping*>(softmaxMappingWs), batchStride,
          maxNTilesPerTask);
  SA_CHECK_KERNEL_RET();
  return SaStatus::SUCCESS;
}

template <typename FType>
void TiledSoftmaxWorkspaceBytes(size_t* reduceWsBytes,
                                size_t* reduceFinishCountBytes, int batch,
                                int nHead, int maxLength) {
  using SoftmaxConfig = SoftmaxConfig<FType>;
  using SoftmaxCompT = typename SoftmaxConfig::ComputeType;

  constexpr int SOFTMAX_TILE_SIZE = SoftmaxConfig::TILED_SOFTMAX_SEQ_TILE_SIZE;
  uint32_t softmaxNTiles = U32DivRU(maxLength, SOFTMAX_TILE_SIZE);
  uint32_t softmaxBatch = batch * nHead;

  *reduceWsBytes = AlignedMemSize<MEM_ALIGN>(softmaxBatch * softmaxNTiles *
                                             sizeof(SoftmaxCompT));
  *reduceFinishCountBytes =
      AlignedMemSize<MEM_ALIGN>(softmaxBatch * sizeof(uint32_t));
  return;
}

}  // namespace detail

template <typename FType>
size_t SoftmaxInplaceWorkspaceBytes(int batch, int nHead, int maxLength) {
  size_t reduceWsSize{0};
  size_t reduceFinishCountSize{0};
  using Config = SoftmaxConfig<FType>;
  if (maxLength > Config::SOFTMAX_MAX_BLOCK * Config::SOFTMAX_MAX_UNROLL) {
    detail::TiledSoftmaxWorkspaceBytes<FType>(
        &reduceWsSize, &reduceFinishCountSize, batch, nHead, maxLength);
  }
  return 2 * reduceWsSize + 2 * reduceFinishCountSize;
}

template <typename FType>
SaStatus SoftmaxInplace(FType* QK, const uint32_t* seqLengths,
                        const void* softmaxMappingWs, void* softmaxWs,
                        int batch, int nHead, int stride, int maxLength,
                        uint32_t totalTiles, int smCount, cudaStream_t stream) {
  using detail::LaunchSoftmaxKernel;
  using detail::TiledSoftmaxDispatchTile;
  using detail::TiledSoftmaxWorkspaceBytes;

  using Config = SoftmaxConfig<FType>;
  uint32_t softmaxBatch = batch * nHead;

  if (maxLength <= 16) {
    return LaunchSoftmaxKernel<4, 32, 4>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 32) {
    return LaunchSoftmaxKernel<8, 16, 4>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 64) {
    return LaunchSoftmaxKernel<16, 8, 4>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 128) {
    return LaunchSoftmaxKernel<32, 4, 4>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 256) {
    return LaunchSoftmaxKernel<32, 4, 8>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 512) {
    return LaunchSoftmaxKernel<64, 2, 8>(QK, seqLengths, stride, softmaxBatch,
                                         nHead, stream);
  } else if (maxLength <= 1024) {
    return LaunchSoftmaxKernel<128, 1, 8>(QK, seqLengths, stride, softmaxBatch,
                                          nHead, stream);
  } else if (maxLength <= 2048) {
    return LaunchSoftmaxKernel<256, 1, 8>(QK, seqLengths, stride, softmaxBatch,
                                          nHead, stream);
  } else if (maxLength <= 4096) {
    return LaunchSoftmaxKernel<512, 1, 8>(QK, seqLengths, stride, softmaxBatch,
                                          nHead, stream);
  } else if (maxLength <= 8192) {
    return LaunchSoftmaxKernel<512, 1, 16>(QK, seqLengths, stride, softmaxBatch,
                                           nHead, stream);
  } else if (maxLength <=
             Config::SOFTMAX_MAX_BLOCK * Config::SOFTMAX_MAX_UNROLL) {
    return LaunchSoftmaxKernel<Config::SOFTMAX_MAX_BLOCK, 1,
                               Config::SOFTMAX_MAX_UNROLL>(
        QK, seqLengths, stride, softmaxBatch, nHead, stream);
  } else {
    size_t softmaxReduceWsSize{0};
    size_t softmaxReduceFinishCountSize{0};
    TiledSoftmaxWorkspaceBytes<FType>(&softmaxReduceWsSize,
                                      &softmaxReduceFinishCountSize, batch,
                                      nHead, maxLength);
    return TiledSoftmaxDispatchTile<Config::TILED_SOFTMAX_BLOCK,
                                    Config::TILED_SOFTMAX_UNROLL,
                                    Config::TILED_SOFTMAX_MAX_NTILES_PER_TASK>(
        QK, seqLengths, softmaxMappingWs, softmaxWs, softmaxReduceWsSize,
        softmaxReduceFinishCountSize, totalTiles, stride, maxLength, nHead,
        smCount, stream);
  }
}

}  // namespace span
