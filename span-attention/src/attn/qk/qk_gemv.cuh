/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qk_gemv.cuh
 */

#pragma once

#include "attn/common/tile_mapping.hpp"
#include "attn/common/utils.hpp"
#include "attn/quant.cuh"
#include "cache_quant/qcache.cuh"
#include "common/check_cuda.h"
#include "utils/pack.cuh"
#include "utils/shuffle.cuh"

namespace span {

namespace detail {

// load the input by 8-half vectorized
template <int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE, int BLOCK_X,
          int BLOCK_Y, int PACK_SIZE,
          int INNER_UNROLL,  // unrolling inside a chunk
          int OUTER_UNROLL,  // unrolling cross chunks
          bool QUANTED, typename FT, typename QT, typename QParamT,
          typename ComputeT, typename QConfT>
__global__ __launch_bounds__(256) void QKGemvKernel(
    const FT* Q, const QT* const* KCachePtrs, FT* QK,
    const uint32_t* seqLengths, const TileMapping* mappings, ComputeT QKScale,
    uint32_t stride, uint32_t nChunk, uint32_t headsPerGroup) {
  static_assert(BLOCK_X <= 32, "");

  static_assert(PACK_SIZE % QConfT::UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_PACK_SIZE = PACK_SIZE / QConfT::UNDERLYING_SIZE;
  constexpr int QUANT_HEAD_SIZE = HEAD_SIZE / QConfT::UNDERLYING_SIZE;

  // shared memory buffer for coherent STG
  __shared__ ComputeT QKSmem[BLOCK_Y * OUTER_UNROLL * INNER_UNROLL];

  const uint32_t globalTileId = blockIdx.x;
  const uint32_t headId = blockIdx.y;
  const uint32_t nGroup = gridDim.y;

  uint32_t attnBatchId = mappings[globalTileId].batchId;
  uint32_t seqBlockId = mappings[globalTileId].tileId;
  uint32_t gemvBatchId = attnBatchId * nGroup + headId;

  uint32_t seqLength = seqLengths[attnBatchId];

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;

  uint32_t tileSeqIdx0 = seqBlockId * BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL);
  uint32_t seqIdx0 = tileSeqIdx0 + tidY;

  // K_cache ldg guard
  bool fullTile =
      tileSeqIdx0 + BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL) <= seqLength;
  bool seqIdxInBound[OUTER_UNROLL][INNER_UNROLL];
  if (!fullTile) {
    int nLdg = seqLength > seqIdx0
                   ? static_cast<int>(U32DivRU(seqLength - seqIdx0, BLOCK_Y))
                   : 0;
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        seqIdxInBound[i][j] = i * INNER_UNROLL + j < nLdg;
      }
    }
  }

  // K_cache addr
  const uint32_t CHUNK_STRIDE =
      BLOCK_Y <= CHUNK_SIZE ? 1 : BLOCK_Y / CHUNK_SIZE;
  uint32_t chunkId0 = seqIdx0 / CHUNK_SIZE;
  const QT* const* KCachePtr0 = KCachePtrs + attnBatchId * nChunk + chunkId0;
  const QT* KLdgPtr[OUTER_UNROLL];
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
      asm volatile("ld.global.nc.b64 %0, [%1];"
                   : "=l"(KLdgPtr[i])
                   : "l"(KCachePtr0 + i * CHUNK_STRIDE));
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
      if (seqIdxInBound[i][0]) {
        asm volatile("ld.global.nc.b64 %0, [%1];"
                     : "=l"(KLdgPtr[i])
                     : "l"(KCachePtr0 + i * CHUNK_STRIDE));
      }
    }
  }

  uint32_t chunkOffset = seqIdx0 % CHUNK_SIZE;
  const QParamT* qParamPtr[OUTER_UNROLL];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
    qParamPtr[i] = reinterpret_cast<const QParamT*>(
                       KLdgPtr[i] + nGroup * CHUNK_SIZE * QUANT_HEAD_SIZE) +
                   headId * CHUNK_SIZE + chunkOffset;
    KLdgPtr[i] += headId * CHUNK_SIZE * QUANT_HEAD_SIZE +
                  chunkOffset * QUANT_HEAD_SIZE + tidX * QUANT_PACK_SIZE;
  }

  // load K_cache
  using KPackT = WordPackT<QUANT_PACK_SIZE, QT>;
  KPackT KData[OUTER_UNROLL][INNER_UNROLL];
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        LdgCS(&KData[i][j], KLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        RegSet0(&KData[i][j]);
        if (seqIdxInBound[i][j]) {
          LdgCS(&KData[i][j], KLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
        }
      }
    }
  }

  // load quantize parameters
  QParamT qParam[OUTER_UNROLL][INNER_UNROLL];
  if constexpr (QUANTED) {
    LoadQuantParam<OUTER_UNROLL, INNER_UNROLL, BLOCK_Y>(
        qParam, qParamPtr, seqIdxInBound, fullTile);
  }

  // convert K_cache and parameters to ComputeT
  ComputeT KCompute[OUTER_UNROLL][INNER_UNROLL][PACK_SIZE];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      KData[i][j]
          .template Unpack<QConfT::UNDERLYING_SIZE,
                           typename QConfT::UnderlyingT>(
              KCompute[i][j], typename QConfT::ExtractorT());
    }
  }

  // K_cache dequantization
  if constexpr (QUANTED) {
    InplaceDequant<OUTER_UNROLL, INNER_UNROLL, PACK_SIZE>(KCompute, qParam);
  }

  // Q vector addr
  const FT* QLdgPtr =
      Q + gemvBatchId * headsPerGroup * HEAD_SIZE + tidX * PACK_SIZE;

#pragma unroll
  for (uint32_t qIdxInGroup = 0; qIdxInGroup < GROUP_SIZE; ++qIdxInGroup) {
    if (qIdxInGroup >= headsPerGroup) {
      break;
    }

    // load Q
    using QPackT = WordPackT<PACK_SIZE, FT>;
    QPackT QData;

    LdgNC(&QData, QLdgPtr + qIdxInGroup * HEAD_SIZE);

    ComputeT QCompute[PACK_SIZE];

    // convert Q vector to ComputeT
    QData.Unpack(QCompute);

    // dot product
    ComputeT dot[OUTER_UNROLL][INNER_UNROLL];
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        dot[i][j] = 0;
#pragma unroll
        for (int p = 0; p < PACK_SIZE; ++p) {
          dot[i][j] += QCompute[p] * KCompute[i][j][p];
        }
      }
    }

// warp reduction
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
#pragma unroll
        for (int p = BLOCK_X; p > 1; p /= 2) {
          dot[i][j] += ShflBfly(0xffffffff, dot[i][j], p / 2, BLOCK_X);
        }
      }
    }

    // stg
    //! NOTE: sync to avoid WAR due to loop over query heads
    __syncthreads();
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        QKSmem[tidY + (i * INNER_UNROLL + j) * BLOCK_Y] = dot[i][j];
      }
    }
    __syncthreads();

    bool doStg = BLOCK_X * BLOCK_Y > BLOCK_Y * OUTER_UNROLL * INNER_UNROLL
                     ? threadIdx.x < BLOCK_Y * OUTER_UNROLL * INNER_UNROLL
                     : true;
    constexpr int STG_ROUND =
        BLOCK_X * BLOCK_Y < BLOCK_Y * OUTER_UNROLL * INNER_UNROLL
            ? BLOCK_Y * OUTER_UNROLL * INNER_UNROLL / (BLOCK_X * BLOCK_Y)
            : 1;
    if (doStg) {
      ComputeT stgData[STG_ROUND];
#pragma unroll
      for (int i = 0; i < STG_ROUND; ++i) {
        stgData[i] = QKSmem[threadIdx.x + i * BLOCK_X * BLOCK_Y];
      }

      uint32_t qkRow = gemvBatchId * headsPerGroup + qIdxInGroup;
      uint32_t stgSeqIdx0 =
          seqBlockId * BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL) + threadIdx.x;
      FT* QKPtr = QK + qkRow * stride + stgSeqIdx0;
      if (fullTile) {
#pragma unroll
        for (int i = 0; i < STG_ROUND; ++i) {
          QKPtr[i * BLOCK_X * BLOCK_Y] = static_cast<FT>(stgData[i] * QKScale);
        }
      } else {
#pragma unroll
        for (int i = 0; i < STG_ROUND; ++i) {
          if (stgSeqIdx0 + i * BLOCK_X * BLOCK_Y < seqLength) {
            QKPtr[i * BLOCK_X * BLOCK_Y] =
                static_cast<FT>(stgData[i] * QKScale);
          }
        }
      }
    }  // if (doStg)
  }
  return;
}

template <QuantMode QMODE, int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE,
          typename FType>
struct QKGemv {
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      float QKScale, int stride, int nGroups, int nChunk,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    using QConfigT = qcache::QCacheConfig<QMODE, FType>;
    using QParamT = qcache::QuantParam<QMODE, FType>;
    using ComputeType = typename QConfigT::ComputeT;
    using QType = typename QConfigT::QuantT;
    static_assert(sizeof(FType) >= sizeof(QType), "");

    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");

    using Config = QKGemvConfig<HEAD_SIZE, FType>;
    constexpr int PACK_SIZE = Config::PACK_SIZE;
    constexpr int BLOCK = Config::BLOCK;
    constexpr int BLOCK_X = Config::BLOCK_X;
    constexpr int BLOCK_Y = Config::BLOCK_Y;

    static_assert(HEAD_SIZE <= 32 * PACK_SIZE && HEAD_SIZE >= PACK_SIZE,
                  "invalid HEAD_SIZE");

    static_assert(BLOCK >= BLOCK_X && BLOCK % BLOCK_X == 0, "");
    static_assert(CHUNK_SIZE % BLOCK_Y == 0 || BLOCK_Y % CHUNK_SIZE == 0, "");

    constexpr int UNROLL = Config::UNROLL;
    constexpr int INNER_UNROLL = CHUNK_SIZE <= BLOCK_Y ? 1
                                 : CHUNK_SIZE >= UNROLL * BLOCK_Y
                                     ? UNROLL
                                     : CHUNK_SIZE / BLOCK_Y;
    constexpr int OUTER_UNROLL =
        INNER_UNROLL < UNROLL ? UNROLL / INNER_UNROLL : 1;

    constexpr bool QUANTED = QMODE != QuantMode::NONE;

    const QType* const* const KCacheArray =
        reinterpret_cast<const QType* const*>(KCachePtrs);

    QKGemvKernel<CHUNK_SIZE, HEAD_SIZE, GROUP_SIZE, BLOCK_X, BLOCK_Y, PACK_SIZE,
                 INNER_UNROLL, OUTER_UNROLL, QUANTED, FType, QType, QParamT,
                 ComputeType, QConfigT>
        <<<dim3(seqBlocks, nGroups), BLOCK, 0, stream>>>(
            Q, KCacheArray, QK, seqLengths,
            static_cast<const TileMapping*>(qkMappingWs),
            static_cast<ComputeType>(QKScale), stride, nChunk, headsPerGroup);

    SA_CHECK_KERNEL_RET();
    return SaStatus::SUCCESS;
  }
};

}  // namespace detail

}  // namespace span
