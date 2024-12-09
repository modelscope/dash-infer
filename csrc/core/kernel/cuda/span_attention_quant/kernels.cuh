/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernels.cuh
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "../cache_quant/qcache.cuh"
#include "../utils/pack.cuh"
#include "../utils/shuffle.cuh"
#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace span_attention_quant {

struct RetConfig {
  static const int Success = 0;
  static const int ProgramError = -1;
  static const int CudaRuntimeError = -2;
};

template <int CHUNK_SIZE, int HEAD_SIZE, int BLOCK_X, int BLOCK_Y,
          int PACK_SIZE,
          int INNER_UNROLL,  // unrolling inside a chunk
          int OUTER_UNROLL,  // unrolling cross chunks
          typename FT, typename QT, typename QParamT, typename ComputeT,
          typename QConfT>
__global__ void QuantizedQKGemvKernel(const FT* Q, const QT* const* KCachePtrs,
                                      FT* QK, ComputeT QKScale,
                                      U32DivMod seqBlockDivMod,
                                      U32DivMod nHeadDivMod, uint32_t seqLength,
                                      uint32_t nChunk) {
  static_assert(BLOCK_X <= 32, "");

  static_assert(PACK_SIZE % QConfT::UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_PACK_SIZE = PACK_SIZE / QConfT::UNDERLYING_SIZE;
  constexpr int QUANT_HEAD_SIZE = HEAD_SIZE / QConfT::UNDERLYING_SIZE;

  // shared memory buffer for coherent STG
  __shared__ ComputeT QKSmem[BLOCK_Y * OUTER_UNROLL * INNER_UNROLL];

  auto seqBlockDM = seqBlockDivMod.DivMod(blockIdx.x);
  uint32_t seqBlockId = seqBlockDM.mod;
  uint32_t gemvBatchId = seqBlockDM.div;

  auto nHeadDM = nHeadDivMod.DivMod(gemvBatchId);
  uint32_t headId = nHeadDM.mod;
  uint32_t attnBatchId = nHeadDM.div;

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;
  uint32_t tileSeqIdx0 = seqBlockId * BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL);
  uint32_t seqIdx0 = tileSeqIdx0 + tidY;

  // K_cache ldg guard
  bool fullTile =
      tileSeqIdx0 + BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL) <= seqLength;
  bool seqIdxInBound[OUTER_UNROLL][INNER_UNROLL];
  if (!fullTile) {
    uint32_t nLdg =
        seqLength > seqIdx0 ? U32DivRU(seqLength - seqIdx0, BLOCK_Y) : 0;
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
    const auto& nHead = nHeadDivMod.d_;
    qParamPtr[i] = reinterpret_cast<const QParamT*>(
                       KLdgPtr[i] + nHead * CHUNK_SIZE * QUANT_HEAD_SIZE) +
                   headId * CHUNK_SIZE + chunkOffset;
    KLdgPtr[i] += headId * CHUNK_SIZE * QUANT_HEAD_SIZE +
                  chunkOffset * QUANT_HEAD_SIZE + tidX * QUANT_PACK_SIZE;
  }

  // Q vector addr
  const FT* QLdgPtr = Q + gemvBatchId * HEAD_SIZE + tidX * PACK_SIZE;

  // load Q
  using QPackT = WordPackT<PACK_SIZE, FT>;
  QPackT QData;
  LdgNC(&QData, QLdgPtr);

  // load K_cache and quantize parameters
  using KPackT = WordPackT<QUANT_PACK_SIZE, QT>;
  KPackT KData[OUTER_UNROLL][INNER_UNROLL];
  QParamT qParam[OUTER_UNROLL][INNER_UNROLL];
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        LdgCS(&KData[i][j], KLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
        LdgCS(&qParam[i][j], qParamPtr[i] + j * BLOCK_Y);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        RegSet0(&KData[i][j]);
        RegSet0(&qParam[i][j]);
        if (seqIdxInBound[i][j]) {
          LdgCS(&KData[i][j], KLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
          LdgCS(&qParam[i][j], qParamPtr[i] + j * BLOCK_Y);
        }
      }
    }
  }

  // convert Q vector to ComputeT
  ComputeT QCompute[PACK_SIZE];
  QData.Unpack(QCompute);

  // convert K_cache and parameters to ComputeT
  ComputeT KCompute[OUTER_UNROLL][INNER_UNROLL][PACK_SIZE];
  ComputeT zero[OUTER_UNROLL][INNER_UNROLL];
  ComputeT scale[OUTER_UNROLL][INNER_UNROLL];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      KData[i][j].template Unpack<QConfT::UNDERLYING_SIZE, QConfT::UnderlyingT>(
          KCompute[i][j], QConfT::ExtractorT());
      zero[i][j] = static_cast<ComputeT>(qParam[i][j].zero);
      scale[i][j] = static_cast<ComputeT>(qParam[i][j].scale);
    }
  }

// K_cache dequantization
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      /**
       * transform '(KCompute - zero) * scale' to 'KCompute * scale -
       * (zero * scale)' to utilize FMA and prune instructions
       */
      zero[i][j] *= scale[i][j];
#pragma unroll
      for (int p = 0; p < PACK_SIZE; ++p) {
        KCompute[i][j][p] = KCompute[i][j][p] * scale[i][j] - zero[i][j];
      }
    }
  }

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
  const int STG_ROUND =
      BLOCK_X * BLOCK_Y < BLOCK_Y * OUTER_UNROLL * INNER_UNROLL
          ? BLOCK_Y * OUTER_UNROLL * INNER_UNROLL / (BLOCK_X * BLOCK_Y)
          : 1;
  if (doStg) {
    ComputeT stgData[STG_ROUND];
#pragma unroll
    for (int i = 0; i < STG_ROUND; ++i) {
      stgData[i] = QKSmem[threadIdx.x + i * BLOCK_X * BLOCK_Y];
    }

    uint32_t stgSeqIdx0 =
        seqBlockId * BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL) + threadIdx.x;
    FT* QKPtr = QK + gemvBatchId * seqLength + stgSeqIdx0;
    if (fullTile) {
#pragma unroll
      for (int i = 0; i < STG_ROUND; ++i) {
        QKPtr[i * BLOCK_X * BLOCK_Y] = static_cast<FT>(stgData[i] * QKScale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < STG_ROUND; ++i) {
        if (stgSeqIdx0 + i * BLOCK_X * BLOCK_Y < seqLength) {
          QKPtr[i * BLOCK_X * BLOCK_Y] = static_cast<FT>(stgData[i] * QKScale);
        }
      }
    }
  }
}

template <int CHUNK_SIZE, int HEAD_SIZE, int BLOCK_X, int BLOCK_Y,
          int PACK_SIZE,
          int INNER_UNROLL,  // unrolling inside a chunk
          int OUTER_UNROLL,  // unrolling cross chunks
          bool USE_WS, typename FT, typename QT, typename QParamT,
          typename ComputeT, typename QConfT>
__global__ void QuantizedQKVGemvKernel(const FT* QK,
                                       const QT* const* VCachePtrs, FT* O,
                                       FT* ws, U32DivMod seqBlockDivMod,
                                       U32DivMod nHeadDivMod,
                                       uint32_t seqLength, uint32_t nChunk) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  static_assert(PACK_SIZE % QConfT::UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_PACK_SIZE = PACK_SIZE / QConfT::UNDERLYING_SIZE;
  constexpr int QUANT_HEAD_SIZE = HEAD_SIZE / QConfT::UNDERLYING_SIZE;

  __shared__ ComputeT smem[PACK_SIZE * BLOCK_Y * BLOCK_X];

  auto seqBlockDM = seqBlockDivMod.DivMod(blockIdx.x);
  uint32_t seqBlockId = seqBlockDM.mod;
  uint32_t gemvBatchId = seqBlockDM.div;

  auto nHeadDM = nHeadDivMod.DivMod(gemvBatchId);
  uint32_t headId = nHeadDM.mod;
  uint32_t attnBatchId = nHeadDM.div;

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;
  uint32_t tileSeqIdx0 = seqBlockId * BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL);
  uint32_t seqIdx0 = tileSeqIdx0 + tidY;

  // V_cache ldg guard
  bool fullTile =
      tileSeqIdx0 + BLOCK_Y * (OUTER_UNROLL * INNER_UNROLL) <= seqLength;
  bool seqIdxInBound[OUTER_UNROLL][INNER_UNROLL];
  if (!fullTile) {
    uint32_t nLdg =
        seqLength > seqIdx0 ? U32DivRU(seqLength - seqIdx0, BLOCK_Y) : 0;
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
    const auto& nHead = nHeadDivMod.d_;
    qParamPtr[i] = reinterpret_cast<const QParamT*>(
                       VLdgPtr[i] + nHead * CHUNK_SIZE * QUANT_HEAD_SIZE) +
                   headId * CHUNK_SIZE + chunkOffset;
    VLdgPtr[i] += headId * CHUNK_SIZE * QUANT_HEAD_SIZE +
                  chunkOffset * QUANT_HEAD_SIZE + tidX * QUANT_PACK_SIZE;
  }

  // QK vector addr
  const FT* QKLdgPtr = QK + gemvBatchId * seqLength + seqIdx0;

  // load V_cache, quantize parameters and QK vector
  using VPackT = WordPackT<QUANT_PACK_SIZE, QT>;
  VPackT VData[OUTER_UNROLL][INNER_UNROLL];
  QParamT qParam[OUTER_UNROLL][INNER_UNROLL];
  FT QKData[OUTER_UNROLL][INNER_UNROLL];

  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        LdgCS(&QKData[i][j], QKLdgPtr + (i * INNER_UNROLL + j) * BLOCK_Y);
        LdgCS(&VData[i][j], VLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
        LdgCS(&qParam[i][j], qParamPtr[i] + j * BLOCK_Y);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        RegSet0(&QKData[i][j]);
        RegSet0(&VData[i][j]);
        RegSet0(&qParam[i][j]);
        if (seqIdxInBound[i][j]) {
          LdgCS(&QKData[i][j], QKLdgPtr + (i * INNER_UNROLL + j) * BLOCK_Y);
          LdgCS(&VData[i][j], VLdgPtr[i] + j * BLOCK_Y * QUANT_HEAD_SIZE);
          LdgCS(&qParam[i][j], qParamPtr[i] + j * BLOCK_Y);
        }
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

  // convert V_cache and parameters to ComputeT
  ComputeT VCompute[OUTER_UNROLL][INNER_UNROLL][PACK_SIZE];
  ComputeT zero[OUTER_UNROLL][INNER_UNROLL];
  ComputeT scale[OUTER_UNROLL][INNER_UNROLL];
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      VData[i][j].template Unpack<QConfT::UNDERLYING_SIZE, QConfT::UnderlyingT>(
          VCompute[i][j], QConfT::ExtractorT());
      zero[i][j] = static_cast<ComputeT>(qParam[i][j].zero);
      scale[i][j] = static_cast<ComputeT>(qParam[i][j].scale);
    }
  }

// V_cache dequantization
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      /**
       * transform '(KCompute - zero) * scale' to 'KCompute * scale -
       * (zero * scale)' to utilize FMA and prune instructions
       */
      zero[i][j] *= scale[i][j];
#pragma unroll
      for (int p = 0; p < PACK_SIZE; ++p) {
        VCompute[i][j][p] = VCompute[i][j][p] * scale[i][j] - zero[i][j];
      }
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
#pragma unroll
    for (int i = 0; i < PACK_SIZE; ++i) {
      smem[threadIdx.x + i * BLOCK_Y * BLOCK_X] = dot[i];
    }
    __syncthreads();

    const int REDUCE_THREADS_X = BLOCK_X;
    const int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
    const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

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

    uint32_t seqBlocks = seqBlockDivMod.d_;
    FT* stgPtr = USE_WS
                     ? ws + (gemvBatchId * seqBlocks + seqBlockId) * HEAD_SIZE +
                           threadIdx.x * PACK_SIZE
                     : O + gemvBatchId * HEAD_SIZE + threadIdx.x * PACK_SIZE;
    *reinterpret_cast<StgPackT*>(stgPtr) = stgReg;
  }
}

template <int HEAD_SIZE, int BLOCK_X, int BLOCK_Y, int PACK_SIZE, int UNROLL,
          typename FT, typename ComputeT>
__global__ void QKVReduceKernel(const FT* ws, FT* O, uint32_t length) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  __shared__ ComputeT smem[PACK_SIZE * BLOCK_Y * BLOCK_X];

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;

  const FT* wsPtr = ws + blockIdx.x * length * HEAD_SIZE + tidY * HEAD_SIZE +
                    tidX * PACK_SIZE;
  uint32_t nFullTile = length / (BLOCK_Y * UNROLL);
  uint32_t lastTile = length % (BLOCK_Y * UNROLL);

  using PT = PackT<PACK_SIZE, FT>;
  const int PACK_STRIDE = BLOCK_Y * (HEAD_SIZE / PACK_SIZE);
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

    const int REDUCE_THREADS_X = BLOCK_X;
    const int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
    const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

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

__device__ __forceinline__ float Expf(const float& x) { return __expf(x); }

__device__ __forceinline__ float Rcpf(const float& x) {
  float ret;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

/**
 * fmax for FP16 or BF16 is only supported on sm_80+ devices, so the
 * template struct MaxReduceFunctor si specialized for each type.
 */
template <typename T>
struct MaxReduceFunctor;

template <>
struct MaxReduceFunctor<float> {
  __device__ __forceinline__ float Init() {
    uint32_t RAW_NEG_INF = 0xff800000;
    return reinterpret_cast<const float&>(RAW_NEG_INF);
  };

  __device__ __forceinline__ float Reduce(const float& x, const float& y) {
    return fmaxf(x, y);
  }
};

template <typename T>
struct SumReduceFunctor {
  __device__ __forceinline__ T Init() { return T(0); }

  __device__ __forceinline__ T Reduce(const T& x, const T& y) { return x + y; }
};

template <int BLOCK_X, int BLOCK_Y, int UNROLL,
          template <typename> class ReduceFunc, typename T>
__device__ __forceinline__ T BlockReduce(const T (&x)[UNROLL]) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");
  __shared__ T smem[32];
  ReduceFunc<T> f;

  // thread reduction
  T ret = x[0];
#pragma unroll
  for (int i = 1; i < UNROLL; ++i) {
    ret = f.Reduce(ret, x[i]);
  }

  // warp reduction
  const int SHFL_WIDTH_0 = BLOCK_X > 32 ? 32 : BLOCK_X;
#pragma unroll
  for (int i = SHFL_WIDTH_0; i > 1; i /= 2) {
    ret = f.Reduce(ret, ShflBfly(0xffffffff, ret, i / 2, SHFL_WIDTH_0));
  }

  // block reduction
  if (BLOCK_X > 32) {
    uint32_t laneId = threadIdx.x % 32;
    uint32_t warpId = threadIdx.x / 32;

    smem[warpId] = ret;
    __syncthreads();

    if (warpId == 0) {
      // set SHFL_WIDTH_1 to 1 in case of BLOCK_X<=32 to avoid compiler
      // warning
      const int SHFL_WIDTH_1 = BLOCK_X > 32 ? BLOCK_X / 32 : 1;
      ret = smem[laneId];
#pragma unroll
      for (int i = SHFL_WIDTH_1; i > 1; i /= 2) {
        ret = f.Reduce(ret, ShflBfly(0xffffffff, ret, i / 2, SHFL_WIDTH_1));
      }
      smem[laneId / SHFL_WIDTH_1] = ret;
    }
    __syncthreads();

    ret = smem[threadIdx.x / BLOCK_X];
  }

  return ret;
}

// length <= BLOCK * UNROLL
template <int BLOCK_X, int BLOCK_Y, int UNROLL, typename FT, typename ComputeT>
__global__ void InplaceSoftmaxKernel(FT* x, uint32_t length, uint32_t batch) {
  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;
  uint32_t batchId = blockIdx.x * BLOCK_Y + tidY;
  FT* xPtr = x + batchId * length + tidX;

  FT xData[UNROLL];
  uint32_t nLdg = length > tidX ? U32DivRU(length - tidX, BLOCK_X) : 0;
  if (batchId < batch) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const double NEG_INF = -1. / 0.;
      xData[i] = i < nLdg ? xPtr[i * BLOCK_X] : static_cast<FT>(NEG_INF);
    }
  }

  ComputeT xCompute[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = static_cast<ComputeT>(xData[i]);
  }

  ComputeT max =
      BlockReduce<BLOCK_X, BLOCK_Y, UNROLL, MaxReduceFunctor, ComputeT>(
          xCompute);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = Expf(xCompute[i] - max);
  }

  ComputeT expSum =
      BlockReduce<BLOCK_X, BLOCK_Y, UNROLL, SumReduceFunctor, ComputeT>(
          xCompute);
  ComputeT expSumRcp = Rcpf(expSum);

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] *= expSumRcp;
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xData[i] = static_cast<FT>(xCompute[i]);
  }

  if (batchId < batch) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < nLdg) {
        xPtr[i * BLOCK_X] = xData[i];
      }
    }
  }
}

/**
 * reduceWs: workspace for reduction, should not be reused for multiple
 *           reductions of a kernel
 * finishCount: finished tile counter, the counter should set to 0 before
 *              kernel launch, and should not be reused for multiple
 *              reductions of a kernel
 */
template <int BLOCK, int UNROLL, int MAX_NTILES,
          template <typename> class ReduceFunc, typename T>
__device__ __forceinline__ T GridXReduce(const T (&x)[UNROLL], T* reduceWs,
                                         uint32_t* finishCount,
                                         uint32_t batchId, uint32_t tileId,
                                         uint32_t nTiles) {
  static_assert((BLOCK & (BLOCK - 1)) == 0, "BLOCK must be power of 2");
  static_assert(BLOCK >= 32, "");
  static_assert(MAX_NTILES >= 32 && MAX_NTILES % 32 == 0, "");

  __shared__ T reduceSmem[32];
  __shared__ T globalAcc;
  __shared__ bool isLastTileSmem;
  ReduceFunc<T> f;
  uint32_t laneId = threadIdx.x % 32;
  uint32_t warpId = threadIdx.x / 32;

  // ---------------------------------------
  // thread block reduction
  // ---------------------------------------
  T acc = x[0];
#pragma unroll
  for (int i = 1; i < UNROLL; ++i) {
    acc = f.Reduce(acc, x[i]);
  }

// warp reduction
#pragma unroll
  for (int i = 32; i > 1; i /= 2) {
    acc = f.Reduce(acc, ShflBfly(0xffffffff, acc, i / 2, 32));
  }

  // block reduction
  if (BLOCK > 32) {
    reduceSmem[warpId] = acc;
    __syncthreads();

    if (warpId == 0) {
      // set SHFL_WIDTH_1 to 1 in case of BLOCK<=32 to avoid compiler
      // warning
      const int SHFL_WIDTH_1 = BLOCK > 32 ? BLOCK / 32 : 1;
      acc = reduceSmem[laneId];
#pragma unroll
      for (int i = SHFL_WIDTH_1; i > 1; i /= 2) {
        acc = f.Reduce(acc, ShflBfly(0xffffffff, acc, i / 2, SHFL_WIDTH_1));
      }
    }
  }

  // ---------------------------------------
  // grid-x reduction
  // ---------------------------------------
  T* reduceWsPtr = reduceWs + batchId * nTiles;
  uint32_t* finishCountPtr = finishCount + batchId;

  // store the tile reduction to workspace
  if (threadIdx.x == 0) {
    StgCG(reduceWsPtr + tileId, acc);
    uint32_t finishRank;
    asm volatile(
#if __CUDA_ARCH__ >= 700
        "atom.release.gpu.global.inc.u32 %0, [%1], 0x7fffffffU;"
#else
        "membar.gl;\n"
        "atom.global.inc.u32 %0, [%1], 0x7fffffffU;"
#endif
        : "=r"(finishRank)
        : "l"(finishCountPtr));
    isLastTileSmem = finishRank == nTiles - 1;
  }
  __syncthreads();

  // usually there's not so many tiles, so we use only 1 warp for
  // inter-tile reduction to avoid inter-warp reduction cost
  if (isLastTileSmem && warpId == 0) {
    T tileAcc[MAX_NTILES / 32];
#pragma unroll
    for (int i = 0; i < MAX_NTILES / 32; ++i) {
      if (laneId + i * 32 < nTiles) {
        LdgCG(&tileAcc[i], reduceWsPtr + laneId + i * 32);
      } else {
        tileAcc[i] = f.Init();
      }
    }

    acc = tileAcc[0];
#pragma unroll
    for (int i = 1; i < MAX_NTILES / 32; ++i) {
      acc = f.Reduce(acc, tileAcc[i]);
    }
#pragma unroll
    for (int i = 32; i > 1; i /= 2) {
      acc = f.Reduce(acc, ShflBfly(0xffffffff, acc, i / 2, 32));
    }

    StgCG(reduceWsPtr, acc);
    asm volatile("membar.gl;");
    // store 0xffffffff to finishCount as a inter-tile reduce complete flag
    StgCG(finishCountPtr, 0xffffffffU);
  }

  // ---------------------------------------
  // load and broadcast the reduction output
  // ---------------------------------------
  if (threadIdx.x == 0) {
    uint32_t globalAccFlag;
    do {
#if __CUDA_ARCH__ >= 700
      // yield SM for aboue L2 cache latency cycles
      asm volatile("nanosleep.u32 200;");
#else
      // yield SM
      asm volatile("membar.cta;");
#endif
      LdgCG(&globalAccFlag, finishCountPtr);
    } while (globalAccFlag != 0xffffffffU);

    LdgCG(&acc, reduceWsPtr);
    globalAcc = acc;
  }
  __syncthreads();

  return globalAcc;
}

/**
 * maxReduceFinishCount and sumReduceFinishCount should set to 0
 * before kernel launch
 *
 * gridDim.x: number of tiles
 * gridDim.y: softmax batch size (batch * nHead)
 * softmax batch size <= 65535 (limited by gridDim.y)
 */
template <int BLOCK, int UNROLL, int MAX_NTILES, typename FT, typename ComputeT>
__global__ void TiledInplaceSoftmaxKernel(FT* x, ComputeT* maxReduceWs,
                                          uint32_t* maxReduceFinishCount,
                                          ComputeT* sumReduceWs,
                                          uint32_t* sumReduceFinishCount,
                                          uint32_t length, uint32_t batch) {
  uint32_t batchId = blockIdx.y;
  uint32_t tileId = blockIdx.x;
  uint32_t tileOffset = tileId * BLOCK * UNROLL;
  FT* xPtr = x + batchId * length + tileOffset + threadIdx.x;
  FT xData[UNROLL];
  bool fullTile = tileOffset + BLOCK * UNROLL <= length;

  // load the input
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      xData[i] = xPtr[i * BLOCK];
    }
  } else {
    uint32_t tileSize = length - tileOffset;
    uint32_t nLdg =
        tileSize > threadIdx.x ? U32DivRU(tileSize - threadIdx.x, BLOCK) : 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const double NEG_INF = -1. / 0.;
      xData[i] = i < nLdg ? xPtr[i * BLOCK] : static_cast<FT>(NEG_INF);
    }
  }

  // convert to ComputeT
  ComputeT xCompute[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = static_cast<ComputeT>(xData[i]);
  }

  // max reduction
  ComputeT max =
      GridXReduce<BLOCK, UNROLL, MAX_NTILES, MaxReduceFunctor, ComputeT>(
          xCompute, maxReduceWs, maxReduceFinishCount, batchId, tileId,
          gridDim.x);

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = Expf(xCompute[i] - max);
  }

  // expsum reduction
  ComputeT expSum =
      GridXReduce<BLOCK, UNROLL, MAX_NTILES, SumReduceFunctor, ComputeT>(
          xCompute, sumReduceWs, sumReduceFinishCount, batchId, tileId,
          gridDim.x);

  ComputeT expSumRcp = Rcpf(expSum);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] *= expSumRcp;
  }

// convert to FT
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xData[i] = static_cast<FT>(xCompute[i]);
  }

  // store the output
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      xPtr[i * BLOCK] = xData[i];
    }
  } else {
    uint32_t tileSize = length - tileOffset;
    uint32_t nLdg =
        tileSize > threadIdx.x ? U32DivRU(tileSize - threadIdx.x, BLOCK) : 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < nLdg) {
        xPtr[i * BLOCK] = xData[i];
      }
    }
  }
}

}  // namespace span_attention_quant
}  // namespace cuda
}  // namespace allspark
