/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qkv_gemv.cuh
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
          bool USE_WS, bool QUANTED, typename FT, typename QT, typename QParamT,
          typename ComputeT, typename QConfT>
__global__ __launch_bounds__(256) void QKVGemvKernel(
    const FT* QK, const QT* const* VCachePtrs, FT* O, FT* ws,
    const uint32_t* seqLengths, const TileMapping* mappings, uint32_t stride,
    uint32_t maxSeqBlocks, uint32_t nChunk, uint32_t headsPerGroup) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  static_assert(PACK_SIZE % QConfT::UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_PACK_SIZE = PACK_SIZE / QConfT::UNDERLYING_SIZE;
  constexpr int QUANT_HEAD_SIZE = HEAD_SIZE / QConfT::UNDERLYING_SIZE;

  __shared__ ComputeT smem[PACK_SIZE * BLOCK_Y * BLOCK_X];

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

  // V_cache ldg guard
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

  // V_cache addr
  const uint32_t CHUNK_STRIDE =
      BLOCK_Y <= CHUNK_SIZE ? 1 : BLOCK_Y / CHUNK_SIZE;
  uint32_t chunkId0 = seqIdx0 / CHUNK_SIZE;
  const QT* const* VCachePtr0 = VCachePtrs + attnBatchId * nChunk + chunkId0;
  const QT* VLdgPtr[OUTER_UNROLL];
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
      asm volatile("ld.global.nc.b64 %0, [%1];"
                   : "=l"(VLdgPtr[i])
                   : "l"(VCachePtr0 + i * CHUNK_STRIDE));
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
      if (seqIdxInBound[i][0]) {
        asm volatile("ld.global.nc.b64 %0, [%1];"
                     : "=l"(VLdgPtr[i])
                     : "l"(VCachePtr0 + i * CHUNK_STRIDE));
      }
    }
  }

  uint32_t chunkOffset = seqIdx0 % CHUNK_SIZE;
  const QParamT* qParamPtr[OUTER_UNROLL];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
    qParamPtr[i] = reinterpret_cast<const QParamT*>(
                       VLdgPtr[i] + nGroup * CHUNK_SIZE * QUANT_HEAD_SIZE) +
                   headId * CHUNK_SIZE + chunkOffset;
    VLdgPtr[i] += headId * CHUNK_SIZE * QUANT_HEAD_SIZE +
                  chunkOffset * QUANT_HEAD_SIZE + tidX * QUANT_PACK_SIZE;
  }

  // load V_cache
  using VPackT = WordPackT<QUANT_PACK_SIZE, QT>;
  VPackT VData[OUTER_UNROLL][INNER_UNROLL];

  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        LdgCS(&VData[i][j], VLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        RegSet0(&VData[i][j]);
        if (seqIdxInBound[i][j]) {
          LdgCS(&VData[i][j], VLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
        }
      }
    }
  }

  QParamT qParam[OUTER_UNROLL][INNER_UNROLL];
  if constexpr (QUANTED) {
    LoadQuantParam<OUTER_UNROLL, INNER_UNROLL, BLOCK_Y>(
        qParam, qParamPtr, seqIdxInBound, fullTile);
  }

  // convert V_cache and parameters to ComputeT
  ComputeT VCompute[OUTER_UNROLL][INNER_UNROLL][PACK_SIZE];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      VData[i][j]
          .template Unpack<QConfT::UNDERLYING_SIZE,
                           typename QConfT::UnderlyingT>(
              VCompute[i][j], typename QConfT::ExtractorT());
    }
  }

  // V_cache dequantization
  if constexpr (QUANTED) {
    InplaceDequant<OUTER_UNROLL, INNER_UNROLL, PACK_SIZE>(VCompute, qParam);
  }

  // QK vector addr
  const FT* QKLdgPtr = QK + gemvBatchId * headsPerGroup * stride + seqIdx0;

#pragma unroll
  for (uint32_t qIdxInGroup = 0; qIdxInGroup < GROUP_SIZE; ++qIdxInGroup) {
    if (qIdxInGroup >= headsPerGroup) {
      break;
    }

    // load QK vector
    FT QKData[OUTER_UNROLL][INNER_UNROLL];

    if (fullTile) {
#pragma unroll
      for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
        for (int j = 0; j < INNER_UNROLL; ++j) {
          LdgCS(&QKData[i][j], QKLdgPtr + qIdxInGroup * stride +
                                   (i * INNER_UNROLL + j) * BLOCK_Y);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
        for (int j = 0; j < INNER_UNROLL; ++j) {
          RegSet0(&QKData[i][j]);
          if (seqIdxInBound[i][j]) {
            LdgCS(&QKData[i][j], QKLdgPtr + qIdxInGroup * stride +
                                     (i * INNER_UNROLL + j) * BLOCK_Y);
          }  // if in bound
        }
      }
    }

    // convert QK vector to ComputeT
    ComputeT QKCompute[OUTER_UNROLL][INNER_UNROLL];

#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        QKCompute[i][j] = static_cast<ComputeT>(QKData[i][j]);
      }
    }

    // thread dot product
    ComputeT dot[PACK_SIZE];
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
      dot[i] = 0;
    }
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
#pragma unroll
        for (int p = 0; p < PACK_SIZE; ++p) {
          dot[p] += QKCompute[i][j] * VCompute[i][j][p];
        }
      }
    }

    // block reduce
    if (BLOCK_Y > 1) {
      //! NOTE: sync to avoid WAR due to loop over query heads
      __syncthreads();
#pragma unroll
      for (int i = 0; i < PACK_SIZE; ++i) {
        smem[threadIdx.x + i * BLOCK_Y * BLOCK_X] = dot[i];
      }
      __syncthreads();

      constexpr int REDUCE_THREADS_X = BLOCK_X;
      constexpr int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
      constexpr int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

      if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
          dot[i] = smem[threadIdx.x + i * BLOCK_Y * BLOCK_X];
        }
#pragma unroll
        for (int i = 1; i < BLOCK_Y / REDUCE_THREADS_Y; ++i) {
#pragma unroll
          for (int j = 0; j < PACK_SIZE; ++j) {
            dot[j] +=
                smem[threadIdx.x + j * BLOCK_Y * BLOCK_X + i * REDUCE_THREADS];
          }
        }

        if (REDUCE_THREADS_Y > 1) {
#pragma unroll
          for (int i = 0; i < PACK_SIZE; ++i) {
#pragma unroll
            for (int j = 32; j > REDUCE_THREADS_X; j /= 2) {
              dot[i] += ShflBfly(0xffffffff, dot[i], j / 2, 32);
            }
          }
        }
      }
    }

    // convert to FT and store the output
    if (threadIdx.x < BLOCK_X) {
      /**
       * packed STG with PackT is only optimized for FP-to-FP conversion,
       * for FP-to-INT16/INT8 or INT-to-INT, it's recommended to use
       * I2IP (cvt.pack.sat) instructions for better performance on sm_72+
       * devices.
       */
      using StgPackT = PackT<PACK_SIZE, FT>;
      StgPackT stgReg;
      for (int i = 0; i < PACK_SIZE; ++i) {
        stgReg.data[i] = static_cast<FT>(dot[i]);
      }

      uint32_t outRow = gemvBatchId * headsPerGroup + qIdxInGroup;
      FT* stgPtr = USE_WS
                       ? ws + (outRow * maxSeqBlocks + seqBlockId) * HEAD_SIZE +
                             threadIdx.x * PACK_SIZE
                       : O + outRow * HEAD_SIZE + threadIdx.x * PACK_SIZE;
      *reinterpret_cast<StgPackT*>(stgPtr) = stgReg;
    }
  }
  return;
}

// load the input by 8-half vectorized
template <int HEAD_SIZE, int BLOCK_X, int BLOCK_Y, int PACK_SIZE, int UNROLL,
          int SEQ_TILE_SIZE, typename FT, typename ComputeT>
__global__ __launch_bounds__(256) void QKVReduceKernel(
    const FT* ws, FT* O, const uint32_t* seqLengths, uint32_t maxSeqBlocks,
    U32DivMod nHeadDivMod) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  __shared__ ComputeT smem[PACK_SIZE * BLOCK_Y * BLOCK_X];

  uint32_t batchId = nHeadDivMod.Div(blockIdx.x);
  uint32_t seqLength = seqLengths[batchId];
  uint32_t seqBlocks = U32DivRU(seqLength, SEQ_TILE_SIZE);

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;

  const FT* wsPtr = ws + blockIdx.x * maxSeqBlocks * HEAD_SIZE +
                    tidY * HEAD_SIZE + tidX * PACK_SIZE;
  uint32_t nFullTile = seqBlocks / (BLOCK_Y * UNROLL);
  uint32_t lastTile = seqBlocks % (BLOCK_Y * UNROLL);

  using PT = PackT<PACK_SIZE, FT>;
  constexpr int PACK_STRIDE = BLOCK_Y * (HEAD_SIZE / PACK_SIZE);
  const PT* ldgPtr = reinterpret_cast<const PT*>(wsPtr);
  PT ldData[UNROLL];
  ComputeT CompData[UNROLL][PACK_SIZE];
  ComputeT acc[PACK_SIZE];
#pragma unroll
  for (int i = 0; i < PACK_SIZE; ++i) {
    acc[i] = 0;
  }

  // fully unrolled tile loop
  for (; nFullTile > 0; --nFullTile) {
// load the input
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      ldData[i] = ldgPtr[i * PACK_STRIDE];
    }

// convert to ComputeT
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK_SIZE; ++j) {
        /**
         * only optimized for 2-Byte or 4-Byte FT type, for 1-Byte
         * or sub byte types, unpack the elements from registers
         * via bfe instruction (R.B1, R.B2 or SHF + SGXT in sass)
         * can be better.
         */
        CompData[i][j] = static_cast<ComputeT>(ldData[i].data[j]);
      }
    }

// reduce
#pragma unroll
    for (int i = 1; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK_SIZE; ++j) {
        CompData[0][j] += CompData[i][j];
      }
    }
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
      acc[i] += CompData[0][i];
    }

    ldgPtr += UNROLL * PACK_STRIDE;
  }

  // last tile loop
  if (lastTile != 0) {
// load the input
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      RegSet0(&ldData[i]);
      if (tidY + i * BLOCK_Y < lastTile) {
        ldData[i] = ldgPtr[i * PACK_STRIDE];
      }
    }

// convert to ComputeT
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK_SIZE; ++j) {
        CompData[i][j] = static_cast<ComputeT>(ldData[i].data[j]);
      }
    }

// reduce
#pragma unroll
    for (int i = 1; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK_SIZE; ++j) {
        CompData[0][j] += CompData[i][j];
      }
    }
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
      acc[i] += CompData[0][i];
    }
  }

  // block reduce
  if (BLOCK_Y > 1) {
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
      smem[threadIdx.x + i * BLOCK_Y * BLOCK_X] = acc[i];
    }
    __syncthreads();

    constexpr int REDUCE_THREADS_X = BLOCK_X;
    constexpr int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
    constexpr int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

    if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
      for (int i = 0; i < PACK_SIZE; ++i) {
        acc[i] = 0;
#pragma unroll
        for (int j = 0; j < BLOCK_Y / REDUCE_THREADS_Y; ++j) {
          acc[i] +=
              smem[threadIdx.x + j * REDUCE_THREADS + i * BLOCK_Y * BLOCK_X];
        }
      }

      if (REDUCE_THREADS_Y > 1) {
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma unroll
          for (int j = 32; j > REDUCE_THREADS_X; j /= 2) {
            acc[i] += ShflBfly(0xffffffff, acc[i], j / 2, 32);
          }
        }
      }
    }
  }

  // convert to FT and store the output
  if (threadIdx.x < BLOCK_X) {
    /**
     * packed STG with PackT is only optimized for FP-to-FP conversion,
     * for FP-to-INT16/INT8 or INT-to-INT, it's recommended to use
     * I2IP (cvt.pack.sat) instructions for better performance on sm_72+
     * devices.
     */
    using StgPackT = PackT<PACK_SIZE, FT>;
    StgPackT stgReg;
    for (int i = 0; i < PACK_SIZE; ++i) {
      stgReg.data[i] = static_cast<FT>(acc[i]);
    }

    FT* stgPtr = O + blockIdx.x * HEAD_SIZE + threadIdx.x * PACK_SIZE;
    *reinterpret_cast<StgPackT*>(stgPtr) = stgReg;
  }
}

template <QuantMode QMODE, int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE,
          typename FType>
struct QKVGemv {
  SaStatus operator()(const FType* QK, const void* const* VCachePtrs,
                      FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, int stride, int batch,
                      int nGroups, int nChunk, int headsPerGroup,
                      uint32_t seqBlocks, cudaStream_t stream) const {
    using QConfigT = qcache::QCacheConfig<QMODE, FType>;
    using QParamT = qcache::QuantParam<QMODE, FType>;
    using ComputeType = typename QConfigT::ComputeT;
    using QType = typename QConfigT::QuantT;
    static_assert(sizeof(FType) >= sizeof(QType), "");

    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");

    // tiled QKV GEMV
    using Config = QKVGemvConfig<HEAD_SIZE, FType>;
    constexpr int PACK_SIZE = Config::PACK_SIZE;
    constexpr int BLOCK = Config::BLOCK;
    constexpr int BLOCK_X = Config::BLOCK_X;
    constexpr int BLOCK_Y = Config::BLOCK_Y;
    constexpr int UNROLL = Config::UNROLL;
    constexpr int INNER_UNROLL = CHUNK_SIZE <= BLOCK_Y ? 1
                                 : CHUNK_SIZE >= UNROLL * BLOCK_Y
                                     ? UNROLL
                                     : CHUNK_SIZE / BLOCK_Y;
    constexpr int OUTER_UNROLL =
        INNER_UNROLL < UNROLL ? UNROLL / INNER_UNROLL : 1;
    constexpr int SEQ_TILE_SIZE = Config::SEQ_TILE_SIZE;

    constexpr bool QUANTED = QMODE != QuantMode::NONE;

    const QType* const* const VCacheArray =
        reinterpret_cast<const QType* const*>(VCachePtrs);

    const int nHead = nGroups * headsPerGroup;

    uint32_t maxQKVSeqBlocks = U32DivRU(stride, Config::SEQ_TILE_SIZE);

    if (maxQKVSeqBlocks == 1) {
      QKVGemvKernel<CHUNK_SIZE, HEAD_SIZE, GROUP_SIZE, BLOCK_X, BLOCK_Y,
                    PACK_SIZE, INNER_UNROLL, OUTER_UNROLL, false, QUANTED,
                    FType, QType, QParamT, ComputeType, QConfigT>
          <<<dim3(seqBlocks, nGroups), BLOCK, 0, stream>>>(
              QK, VCacheArray, O, QKVReduceWs, seqLengths,
              static_cast<const TileMapping*>(qkvMappingWs), stride,
              maxQKVSeqBlocks, nChunk, headsPerGroup);
    } else {
      QKVGemvKernel<CHUNK_SIZE, HEAD_SIZE, GROUP_SIZE, BLOCK_X, BLOCK_Y,
                    PACK_SIZE, INNER_UNROLL, OUTER_UNROLL, true, QUANTED, FType,
                    QType, QParamT, ComputeType, QConfigT>
          <<<dim3(seqBlocks, nGroups), BLOCK, 0, stream>>>(
              QK, VCacheArray, O, QKVReduceWs, seqLengths,
              static_cast<const TileMapping*>(qkvMappingWs), stride,
              maxQKVSeqBlocks, nChunk, headsPerGroup);
    }
    SA_CHECK_KERNEL_RET();

    // QKV GEMV reduction
    if (maxQKVSeqBlocks > 1) {
      constexpr int RED_PACK_SIZE = 16 / sizeof(FType);
      constexpr int BLOCK = 256;
      constexpr int RED_BLOCK_X = HEAD_SIZE / RED_PACK_SIZE;
      constexpr int RED_BLOCK_Y = BLOCK / RED_BLOCK_X;
      constexpr int RED_UNROLL = 4;

      QKVReduceKernel<HEAD_SIZE, RED_BLOCK_X, RED_BLOCK_Y, RED_PACK_SIZE,
                      RED_UNROLL, SEQ_TILE_SIZE, FType, ComputeType>
          <<<batch * nHead, RED_BLOCK_X * RED_BLOCK_Y, 0, stream>>>(
              QKVReduceWs, O, seqLengths, maxQKVSeqBlocks, U32DivMod(nHead));
    }
    SA_CHECK_KERNEL_RET();
    return SaStatus::SUCCESS;
  }
};

}  // namespace detail
}  // namespace span
