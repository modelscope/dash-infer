/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention_bf16.cuh
 */

#pragma once
#include <core/kernel/cuda/cuda_kernel.h>
#include <utility/check_cuda.h>

#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#include "softmax.cuh"
#include "span_attention.cuh"
#include "utils.cuh"

namespace allspark {
namespace cuda {

#define SPAN_ATTN_CHECK_FUNC(EXPR)                   \
  {                                                  \
    auto err = (EXPR);                               \
    if (err != span_attention::RetConfig::Success) { \
      return err;                                    \
    }                                                \
  }

namespace span_attention {

using bfloat16 = __nv_bfloat16;

// load the input by 8-half vectorized
template <int CHUNK_SIZE, int HEAD_SIZE, int BLOCK_X, int BLOCK_Y>
__global__ void QKGemvLdg8Kernel(const bfloat16* Q,
                                 const bfloat16* const* KCachePtrs,
                                 bfloat16* QK, float QKScale,
                                 U32DivMod seqBlockDivMod,
                                 U32DivMod nGroupDivMod, uint32_t seqLength,
                                 uint32_t nChunk, uint32_t headsPerGroup) {
  static_assert(BLOCK_X <= 32, "");

  auto seqBlockDM = seqBlockDivMod.DivMod(blockIdx.x);
  uint32_t seqBlockId = seqBlockDM.mod;
  uint32_t gemvBatchId = seqBlockDM.div;

  auto nHeadDM = nGroupDivMod.DivMod(gemvBatchId);
  uint32_t headId = nHeadDM.mod;
  uint32_t attnBatchId = nHeadDM.div;

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;

  uint32_t seqIdx = seqBlockId * BLOCK_Y + tidY;

  // K_cache ldg guard
  bool seqIdxInBound = seqIdx < seqLength;

  // K_cache addr
  uint32_t chunkId = seqIdx / CHUNK_SIZE;
  uint32_t chunkOffset = seqIdx % CHUNK_SIZE;
  const bfloat16* KLdgPtr;
  if (seqIdxInBound) {
    asm volatile("ld.global.nc.b64 %0, [%1];"
                 : "=l"(KLdgPtr)
                 : "l"(KCachePtrs + attnBatchId * nChunk + chunkId));
  }
  KLdgPtr +=
      headId * CHUNK_SIZE * HEAD_SIZE + chunkOffset * HEAD_SIZE + tidX * 8;

  // load Q and K_cache, and do 8-half dot product
  float kf[8];
  asm volatile(
      "{.reg.pred     p;\n"
      " .reg.b32      k0, k1, k2, k3;\n"
      " .reg.b16      kl0, kh0, kl1, kh1, kl2, kh2, kl3, kh3;\n"
      // load K_cache
      " setp.ne.b32               p, %9, 0;\n"
      " @p ld.global.cs.v4.b32    {k0, k1, k2, k3}, [%8];\n"
      // convert to FP32
      " mov.b32       {kl0, kh0}, k0;\n"
      " mov.b32       {kl1, kh1}, k1;\n"
      " mov.b32       {kl2, kh2}, k2;\n"
      " mov.b32       {kl3, kh3}, k3;\n"
      " mov.b32       %0, {0, kl0};\n"
      " mov.b32       %1, {0, kh0};\n"
      " mov.b32       %2, {0, kl1};\n"
      " mov.b32       %3, {0, kh1};\n"
      " mov.b32       %4, {0, kl2};\n"
      " mov.b32       %5, {0, kh2};\n"
      " mov.b32       %6, {0, kl3};\n"
      " mov.b32       %7, {0, kh3};}"
      : "=f"(kf[0]), "=f"(kf[1]), "=f"(kf[2]), "=f"(kf[3]), "=f"(kf[4]),
        "=f"(kf[5]), "=f"(kf[6]), "=f"(kf[7])
      : "l"(KLdgPtr), "r"(static_cast<int>(seqIdxInBound)));

  // Q vector addr
  const bfloat16* QLdgPtr =
      Q + gemvBatchId * headsPerGroup * HEAD_SIZE + tidX * 8;
  for (uint32_t qIdxInGroup = 0; qIdxInGroup < headsPerGroup; ++qIdxInGroup) {
    float dot;
    asm volatile(
        "{.reg.pred     p;\n"
        " .reg.b32      q0, q1, q2, q3;\n"
        " .reg.b16      ql0, qh0, ql1, qh1, ql2, qh2, ql3, qh3;\n"
        " .reg.f32      qf0, qf1, qf2, qf3, qf4, qf5, qf6, qf7;\n"
        " .reg.f32      d;\n"
        // load Q
        " setp.ne.b32               p, %10, 0;\n"
        " @p ld.global.nc.v4.b32    {q0, q1, q2, q3}, [%9];\n"
        // convert to FP32
        " mov.b32       {ql0, qh0}, q0;\n"
        " mov.b32       {ql1, qh1}, q1;\n"
        " mov.b32       {ql2, qh2}, q2;\n"
        " mov.b32       {ql3, qh3}, q3;\n"
        " mov.b32       qf0, {0, ql0};\n"
        " mov.b32       qf1, {0, qh0};\n"
        " mov.b32       qf2, {0, ql1};\n"
        " mov.b32       qf3, {0, qh1};\n"
        " mov.b32       qf4, {0, ql2};\n"
        " mov.b32       qf5, {0, qh2};\n"
        " mov.b32       qf6, {0, ql3};\n"
        " mov.b32       qf7, {0, qh3};\n"
        // dot product
        " mul.f32       d, qf0, %1;\n"
        " fma.rn.f32    d, qf1, %2, d;\n"
        " fma.rn.f32    d, qf2, %3, d;\n"
        " fma.rn.f32    d, qf3, %4, d;\n"
        " fma.rn.f32    d, qf4, %5, d;\n"
        " fma.rn.f32    d, qf5, %6, d;\n"
        " fma.rn.f32    d, qf6, %7, d;\n"
        " fma.rn.f32    d, qf7, %8, d;\n"
        " mov.b32       %0, d;}"
        : "=f"(dot)
        : "f"(kf[0]), "f"(kf[1]), "f"(kf[2]), "f"(kf[3]), "f"(kf[4]),
          "f"(kf[5]), "f"(kf[6]), "f"(kf[7]),
          "l"(QLdgPtr + qIdxInGroup * HEAD_SIZE),
          "r"(static_cast<int>(seqIdxInBound)));

// warp reduce
#pragma unroll
    for (int i = BLOCK_X; i > 1; i /= 2) {
      dot += __shfl_xor_sync(0xffffffff, dot, i / 2, BLOCK_X);
    }

    // stg
    uint32_t qkRow = gemvBatchId * headsPerGroup + qIdxInGroup;
    if (tidX == 0 && seqIdxInBound) {
      QK[qkRow * seqLength + seqBlockId * BLOCK_Y + tidY] =
          __float2bfloat16(dot * QKScale);
    }
  }
}

// load the input by 8-half vectorized
template <int CHUNK_SIZE, int HEAD_SIZE, int BLOCK_X, int BLOCK_Y, int UNROLL,
          bool USE_WS>
__global__ __launch_bounds__(BLOCK_X* BLOCK_Y) void QKVGemvLdg8Kernel(
    const bfloat16* QK, const bfloat16* const* VCachePtrs, bfloat16* O,
    bfloat16* ws, U32DivMod seqBlockDivMod, U32DivMod nGroupDivMod,
    uint32_t seqLength, uint32_t nChunk, uint32_t headsPerGroup) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  __shared__ float smem[8 * BLOCK_Y * BLOCK_X];

  auto seqBlockDM = seqBlockDivMod.DivMod(blockIdx.x);
  uint32_t seqBlockId = seqBlockDM.mod;
  uint32_t gemvBatchId = seqBlockDM.div;

  auto nHeadDM = nGroupDivMod.DivMod(gemvBatchId);
  uint32_t headId = nHeadDM.mod;
  uint32_t attnBatchId = nHeadDM.div;

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;
  uint32_t seqIdxBase = seqBlockId * BLOCK_Y * UNROLL + tidY;

  uint32_t seqIdx[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    seqIdx[i] = seqIdxBase + i * BLOCK_Y;
  }

  // V_cache ptr index
  const bfloat16* const* VCachePtrAddr[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    VCachePtrAddr[i] =
        VCachePtrs + attnBatchId * nChunk + seqIdx[i] / CHUNK_SIZE;
  }

  // load V_cache chunk ptr
  const bfloat16* VLdgPtr[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (seqIdx[i] < seqLength) {
      asm volatile("ld.global.nc.b64 %0, [%1];"
                   : "=l"(VLdgPtr[i])
                   : "l"(VCachePtrAddr[i]));
    }
  }

// V_cahce LDG ptr
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    VLdgPtr[i] += headId * CHUNK_SIZE * HEAD_SIZE +
                  seqIdx[i] % CHUNK_SIZE * HEAD_SIZE + tidX * 8;
  }

  // load V_cache
  uint32_t VData[UNROLL][4];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int InBound = seqIdx[i] < seqLength;
    asm volatile(
        "{.reg.pred                 p;\n"
        " setp.ne.b32               p, %5, 0;\n"
        " @!p mov.b64               {%0, %1}, 0;\n"
        " @!p mov.b64               {%2, %3}, 0;\n"
        " @p ld.global.cs.v4.b32    {%0, %1, %2, %3}, [%4];}"
        : "=r"(VData[i][0]), "=r"(VData[i][1]), "=r"(VData[i][2]),
          "=r"(VData[i][3])
        : "l"(VLdgPtr[i]), "r"(InBound));
  }

  // convert VData to FP32
  float VF32[UNROLL][8];
  const bfloat16* VDataBfPtr = reinterpret_cast<bfloat16*>(VData);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      VF32[i][j] = __bfloat162float(VDataBfPtr[i * 8 + j]);
    }
  }

  // QK vector LDG ptr
  const bfloat16* QKLdgPtr =
      QK + gemvBatchId * headsPerGroup * seqLength + seqIdxBase;
  for (uint32_t qIdxInGroup = 0; qIdxInGroup < headsPerGroup; ++qIdxInGroup) {
    // load QK vector
    bfloat16 QKF16[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      int InBound = seqIdx[i] < seqLength;
      asm volatile(
          "{.reg.pred                 p;\n"
          " setp.ne.b32               p, %2, 0;\n"
          " @!p mov.b16               %0, 0;\n"
          " @p ld.global.cs.b16       %0, [%1];}"
          : "=h"(reinterpret_cast<uint16_t&>(QKF16[i]))
          : "l"(QKLdgPtr + qIdxInGroup * seqLength + i * BLOCK_Y),
            "r"(InBound));
    }

    // convert QKF16 to FP32
    float QKF32[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      QKF32[i] = __bfloat162float(QKF16[i]);
    }

    // thread dot product
    float dot[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      dot[i] = QKF32[0] * VF32[0][i];
    }
#pragma unroll
    for (int i = 1; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dot[j] += QKF32[i] * VF32[i][j];
      }
    }

    // block reduce
    if (BLOCK_Y > 1) {
      //! NOTE: sync to avoid RAW due to loop over query heads
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        smem[threadIdx.x + i * BLOCK_Y * BLOCK_X] = dot[i];
      }
      __syncthreads();

      const int REDUCE_THREADS_X = BLOCK_X;
      const int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
      const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

      if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
        for (int packIt = 0; packIt < 8; ++packIt) {
          dot[packIt] = 0;
#pragma unroll
          for (int i = 0; i < BLOCK_Y / REDUCE_THREADS_Y; ++i) {
            dot[packIt] += smem[threadIdx.x + i * REDUCE_THREADS +
                                packIt * BLOCK_Y * BLOCK_X];
          }
        }

        if (REDUCE_THREADS_Y > 1) {
#pragma unroll
          for (int packIt = 0; packIt < 8; ++packIt) {
#pragma unroll
            for (int i = 32; i > REDUCE_THREADS_X; i /= 2) {
              dot[packIt] +=
                  __shfl_xor_sync(0xffffffff, dot[packIt], i / 2, 32);
            }
          }
        }
      }
    }

    // convert to BF16 and stg
    if (threadIdx.x < BLOCK_X) {
      uint32_t seqBlocks = seqBlockDivMod.d_;
      uint32_t outRow = gemvBatchId * headsPerGroup + qIdxInGroup;
      bfloat16* stgPtr =
          USE_WS ? ws + (outRow * seqBlocks + seqBlockId) * HEAD_SIZE +
                       threadIdx.x * 8
                 : O + outRow * HEAD_SIZE + threadIdx.x * 8;

      bfloat16 bfRegs[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        bfRegs[i] = __float2bfloat16(dot[i]);
      }
      const uint16_t* u16Regs = reinterpret_cast<uint16_t*>(&bfRegs);
      asm volatile(
          "{.reg.b32 r0, r1, r2, r3;\n"
          " mov.b32           r0, {%0, %1};\n"
          " mov.b32           r1, {%2, %3};\n"
          " mov.b32           r2, {%4, %5};\n"
          " mov.b32           r3, {%6, %7};\n"
          " st.global.v4.b32  [%8], {r0, r1, r2, r3};}"
          :
          : "h"(u16Regs[0]), "h"(u16Regs[1]), "h"(u16Regs[2]), "h"(u16Regs[3]),
            "h"(u16Regs[4]), "h"(u16Regs[5]), "h"(u16Regs[6]), "h"(u16Regs[7]),
            "l"(stgPtr));
    }
  }
}

// load the input by 8-half vectorized
template <int HEAD_SIZE, int BLOCK_X, int BLOCK_Y, int UNROLL>
__global__ void QKVReduceLdg8Kernel(const bfloat16* ws, bfloat16* O,
                                    uint32_t length) {
  static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be power of 2");
  static_assert((BLOCK_Y & (BLOCK_Y - 1)) == 0, "BLOCK_Y must be power of 2");
  static_assert(BLOCK_X * BLOCK_Y >= 32, "");

  __shared__ float smem[8 * BLOCK_Y * BLOCK_X];

  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;

  const bfloat16* wsPtr =
      ws + blockIdx.x * length * HEAD_SIZE + tidY * HEAD_SIZE + tidX * 8;
  uint32_t nFullTile = length / (BLOCK_Y * UNROLL);
  uint32_t lastTile = length % (BLOCK_Y * UNROLL);

  uint32_t IData[UNROLL][4];
  const bfloat16* IDataBfPtr = reinterpret_cast<bfloat16*>(IData);
  float FData[UNROLL][8];
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0;
  }

  // fully unrolled tile loop
  for (; nFullTile > 0; --nFullTile) {
// load data
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(IData[i][0]), "=r"(IData[i][1]), "=r"(IData[i][2]),
                     "=r"(IData[i][3])
                   : "l"(wsPtr + i * BLOCK_Y * HEAD_SIZE));
    }

// convert to FP32
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        FData[i][j] = __bfloat162float(IDataBfPtr[i * 8 + j]);
      }
    }

// reduce
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j] += FData[i][j];
      }
    }

    wsPtr += BLOCK_Y * UNROLL * HEAD_SIZE;
  }

  // last tile loop
  if (lastTile != 0) {
// load data
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      int inBound = tidY + i * BLOCK_Y < lastTile;
      asm volatile(
          "{.reg.pred             p;\n"
          " setp.ne.b32           p, %5, 0;\n"
          " @!p mov.b64           {%0, %1}, 0;\n"
          " @!p mov.b64           {%2, %3}, 0;\n"
          " @p ld.global.v4.b32   {%0, %1, %2, %3}, [%4];}"
          : "=r"(IData[i][0]), "=r"(IData[i][1]), "=r"(IData[i][2]),
            "=r"(IData[i][3])
          : "l"(wsPtr + i * BLOCK_Y * HEAD_SIZE), "r"(inBound));
    }

// convert to FP32
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        FData[i][j] = __bfloat162float(IDataBfPtr[i * 8 + j]);
      }
    }

// reduce
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j] += FData[i][j];
      }
    }
  }

  // block reduce
  if (BLOCK_Y > 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      smem[threadIdx.x + i * BLOCK_Y * BLOCK_X] = acc[i];
    }
    __syncthreads();

    const int REDUCE_THREADS_X = BLOCK_X;
    const int REDUCE_THREADS_Y = BLOCK_X < 32 ? 32 / BLOCK_X : 1;
    const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

    if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
      for (int packIt = 0; packIt < 8; ++packIt) {
        acc[packIt] = 0;
#pragma unroll
        for (int i = 0; i < BLOCK_Y / REDUCE_THREADS_Y; ++i) {
          acc[packIt] += smem[threadIdx.x + i * REDUCE_THREADS +
                              packIt * BLOCK_Y * BLOCK_X];
        }
      }

      if (REDUCE_THREADS_Y > 1) {
#pragma unroll
        for (int packIt = 0; packIt < 8; ++packIt) {
#pragma unroll
          for (int i = 32; i > REDUCE_THREADS_X; i /= 2) {
            acc[packIt] += __shfl_xor_sync(0xffffffff, acc[packIt], i / 2, 32);
          }
        }
      }
    }
  }

  // convert to BF16 and stg
  if (threadIdx.x < BLOCK_X) {
    bfloat16 bfRegs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      bfRegs[i] = __float2bfloat16(acc[i]);
    }
    const uint16_t* u16Regs = reinterpret_cast<uint16_t*>(&bfRegs);
    asm volatile(
        "{.reg.b32 r0, r1, r2, r3;\n"
        " mov.b32           r0, {%0, %1};\n"
        " mov.b32           r1, {%2, %3};\n"
        " mov.b32           r2, {%4, %5};\n"
        " mov.b32           r3, {%6, %7};\n"
        " st.global.v4.b32  [%8], {r0, r1, r2, r3};}"
        :
        : "h"(u16Regs[0]), "h"(u16Regs[1]), "h"(u16Regs[2]), "h"(u16Regs[3]),
          "h"(u16Regs[4]), "h"(u16Regs[5]), "h"(u16Regs[6]), "h"(u16Regs[7]),
          "l"(O + blockIdx.x * HEAD_SIZE + threadIdx.x * 8));
  }
}

}  // namespace span_attention

template <>
class SpanAttention<hie::bfloat16> {
  using nvbf16 = span_attention::bfloat16;
  using FType = hie::bfloat16;
  using ComputeType = float;

 public:
  SpanAttention(int headSize, int nHead, int nGroups, int seqLength,
                int chunkSize, int nChunk, int batch, int deviceId)
      : headSize_(headSize),
        nHead_(nHead),
        nGroups_(nGroups),
        headsPerGroup_(nHead / nGroups),
        seqLength_(seqLength),
        chunkSize_(chunkSize),
        nChunk_(nChunk),
        batch_(batch) {
    const int MEM_ALIGN = 128;

    // QK
    QKSize_ = AlignedMemSize<MEM_ALIGN>(batch_ * nHead_ * seqLength_ *
                                        sizeof(hie::bfloat16));
    // QKVGemv reduction workspace
    uint32_t seqBlocks;
    switch (headSize_) {
      case 64:
        seqBlocks = span_attention::U32DivRU(seqLength_,
                                             QKVGemvConfig<64>::SEQ_TILE_SIZE);
        break;
      case 128:
        seqBlocks = span_attention::U32DivRU(seqLength_,
                                             QKVGemvConfig<128>::SEQ_TILE_SIZE);
        break;
      default:
        seqBlocks = 0;
        break;
    }
    QKVReduceWsSize_ = AlignedMemSize<MEM_ALIGN>(
        batch_ * nHead_ * seqBlocks * headSize_ * sizeof(hie::bfloat16));

    // tiled softmax workspace
    if (seqLength_ >
        SoftmaxConfig::SOFTMAX_MAX_BLOCK * SoftmaxConfig::SOFTMAX_MAX_UNROLL) {
      uint32_t softmaxTileSize = SoftmaxConfig::TILED_SOFTMAX_BLOCK *
                                 SoftmaxConfig::TILED_SOFTMAX_UNROLL;
      uint32_t softmaxNTiles =
          span_attention::U32DivRU(seqLength_, softmaxTileSize);
      uint32_t softmaxBatch = batch_ * nHead_;

      softmaxReduceWsSize_ = AlignedMemSize<MEM_ALIGN>(
          softmaxBatch * softmaxNTiles * sizeof(ComputeType));
      softmaxReduceFinishCountSize_ =
          AlignedMemSize<MEM_ALIGN>(softmaxBatch * sizeof(uint32_t));
    } else {
      softmaxReduceWsSize_ = 0;
      softmaxReduceFinishCountSize_ = 0;
    }

    AS_CHECK_CUDA(cudaDeviceGetAttribute(
        &smCount_, cudaDevAttrMultiProcessorCount, deviceId));
  }

  size_t GetWorkspaceSize() const {
    return QKSize_ + 2 * softmaxReduceWsSize_ +
           2 * softmaxReduceFinishCountSize_ + QKVReduceWsSize_;
  }

  /**
   *  Q: device memory
   *  KCachePtrs, VCachePtrs: device memory
   *  O: device memory
   *  ws: device memory
   *
   * return value:
   *     0: success
   *    -1: program error
   *    -2: cuda runtime error
   */
  int Run(const hie::bfloat16* Q, const hie::bfloat16* const* KCachePtrs,
          const hie::bfloat16* const* VCachePtrs, hie::bfloat16* O,
          float QKScale, void* ws, size_t wsSize, cudaStream_t stream) const {
    if (wsSize < GetWorkspaceSize()) {
      return span_attention::RetConfig::ProgramError;
    }

    // workspace
    char* wsPool = static_cast<char*>(ws);
    hie::bfloat16* QK = reinterpret_cast<hie::bfloat16*>(wsPool);
    void* softmaxWs = wsPool + QKSize_;
    hie::bfloat16* QKVReduceWs = reinterpret_cast<hie::bfloat16*>(
        wsPool + QKSize_ + 2 * softmaxReduceWsSize_ +
        2 * softmaxReduceFinishCountSize_);

    // dispatch chunkSize
    switch (chunkSize_) {
      case 16:
        return DispatchHeadSize<16>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 32:
        return DispatchHeadSize<32>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 64:
        return DispatchHeadSize<64>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 128:
        return DispatchHeadSize<128>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                     QKVReduceWs, O, QKScale, stream);
      default:
        return span_attention::RetConfig::ProgramError;
    }
  }

 private:
  template <int ALIGN>
  size_t AlignedMemSize(size_t s) const {
    return (s + ALIGN - 1) / ALIGN * ALIGN;
  }

  template <int CHUNK_SIZE>
  int DispatchHeadSize(const hie::bfloat16* Q,
                       const hie::bfloat16* const* KCachePtrs,
                       const hie::bfloat16* const* VCachePtrs,
                       hie::bfloat16* QK, void* softmaxWs,
                       hie::bfloat16* QKVReduceWs, hie::bfloat16* O,
                       float QKScale, cudaStream_t stream) const {
    switch (headSize_) {
      case 64:
        return LaunchKernel<CHUNK_SIZE, 64>(Q, KCachePtrs, VCachePtrs, QK,
                                            softmaxWs, QKVReduceWs, O, QKScale,
                                            stream);
      case 128:
        return LaunchKernel<CHUNK_SIZE, 128>(Q, KCachePtrs, VCachePtrs, QK,
                                             softmaxWs, QKVReduceWs, O, QKScale,
                                             stream);
      default:
        return span_attention::RetConfig::ProgramError;
    }
  }

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int LaunchKernel(const hie::bfloat16* Q,
                   const hie::bfloat16* const* KCachePtrs,
                   const hie::bfloat16* const* VCachePtrs, hie::bfloat16* QK,
                   void* softmaxWs, hie::bfloat16* QKVReduceWs,
                   hie::bfloat16* O, float QKScale, cudaStream_t stream) const {
    // QK GEMV
    SPAN_ATTN_CHECK_FUNC(
        (QKGemv<CHUNK_SIZE, HEAD_SIZE>(Q, KCachePtrs, QK, QKScale, stream)));

    // softmax
    SPAN_ATTN_CHECK_FUNC(SoftmaxInplace(QK, softmaxWs, stream));

    // QKV GEMV
    SPAN_ATTN_CHECK_FUNC((QKVGemv<CHUNK_SIZE, HEAD_SIZE>(
        QK, VCachePtrs, QKVReduceWs, O, stream)));

    return span_attention::RetConfig::Success;
  }

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int QKGemv(const hie::bfloat16* Q, const hie::bfloat16* const* KCachePtrs,
             hie::bfloat16* QK, float QKScale, cudaStream_t stream) const {
    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");
    static_assert(HEAD_SIZE <= 32 * 8 && HEAD_SIZE >= 8, "");

    const int BLOCK = 256;
    const int BLOCK_X = HEAD_SIZE / 8;
    const int BLOCK_Y = BLOCK / BLOCK_X;

    static_assert(BLOCK >= BLOCK_X && BLOCK % BLOCK_X == 0, "");
    static_assert(CHUNK_SIZE % BLOCK_Y == 0 || BLOCK_Y % CHUNK_SIZE == 0, "");

    uint32_t seqBlocks = span_attention::U32DivRU(seqLength_, BLOCK_Y);
    span_attention::QKGemvLdg8Kernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y>
        <<<batch_ * nGroups_ * seqBlocks, BLOCK, 0, stream>>>(
            reinterpret_cast<const nvbf16*>(Q),
            reinterpret_cast<const nvbf16* const*>(KCachePtrs),
            reinterpret_cast<nvbf16*>(QK), QKScale,
            span_attention::U32DivMod(seqBlocks),
            span_attention::U32DivMod(nGroups_), seqLength_, nChunk_,
            headsPerGroup_);

    AS_CHECK_CUDA(cudaGetLastError());
    return span_attention::RetConfig::Success;
  }

  template <int HEAD_SIZE>
  struct QKVGemvConfig {
    static const int BLOCK_X = HEAD_SIZE / 8;  // 8-half vectorized
    static const int BLOCK_Y = 512 / BLOCK_X;
    static const int UNROLL = 8;
    static const int SEQ_TILE_SIZE = BLOCK_Y * UNROLL;
  };

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int QKVGemv(const hie::bfloat16* QK, const hie::bfloat16* const* VCachePtrs,
              hie::bfloat16* QKVReduceWs, hie::bfloat16* O,
              cudaStream_t stream) const {
    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");

    // tiled QKV GEMV
    const int BLOCK_X = QKVGemvConfig<HEAD_SIZE>::BLOCK_X;
    const int BLOCK_Y = QKVGemvConfig<HEAD_SIZE>::BLOCK_Y;
    const int UNROLL = QKVGemvConfig<HEAD_SIZE>::UNROLL;

    uint32_t seqBlocks = span_attention::U32DivRU(seqLength_, BLOCK_Y * UNROLL);
    if (seqBlocks == 1) {
      span_attention::QKVGemvLdg8Kernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y,
                                        UNROLL, false>
          <<<batch_ * nGroups_ * seqBlocks, BLOCK_X * BLOCK_Y, 0, stream>>>(
              reinterpret_cast<const nvbf16*>(QK),
              reinterpret_cast<const nvbf16* const*>(VCachePtrs),
              reinterpret_cast<nvbf16*>(O),
              reinterpret_cast<nvbf16*>(QKVReduceWs),
              span_attention::U32DivMod(seqBlocks),
              span_attention::U32DivMod(nGroups_), seqLength_, nChunk_,
              headsPerGroup_);
    } else {
      span_attention::QKVGemvLdg8Kernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y,
                                        UNROLL, true>
          <<<batch_ * nGroups_ * seqBlocks, BLOCK_X * BLOCK_Y, 0, stream>>>(
              reinterpret_cast<const nvbf16*>(QK),
              reinterpret_cast<const nvbf16* const*>(VCachePtrs),
              reinterpret_cast<nvbf16*>(O),
              reinterpret_cast<nvbf16*>(QKVReduceWs),
              span_attention::U32DivMod(seqBlocks),
              span_attention::U32DivMod(nGroups_), seqLength_, nChunk_,
              headsPerGroup_);
    }
    AS_CHECK_CUDA(cudaGetLastError());

    // QKV GEMV reduction
    if (seqBlocks > 1) {
      const int BLOCK = 256;
      const int RED_BLOCK_X = HEAD_SIZE / 8;
      const int RED_BLOCK_Y = BLOCK / RED_BLOCK_X;
      const int RED_UNROLL = 4;

      span_attention::QKVReduceLdg8Kernel<HEAD_SIZE, RED_BLOCK_X, RED_BLOCK_Y,
                                          RED_UNROLL>
          <<<batch_ * nHead_, RED_BLOCK_X * RED_BLOCK_Y, 0, stream>>>(
              reinterpret_cast<nvbf16*>(QKVReduceWs),
              reinterpret_cast<nvbf16*>(O), seqBlocks);
    }
    AS_CHECK_CUDA(cudaGetLastError());
    return span_attention::RetConfig::Success;
  }

  struct SoftmaxConfig {
    // 1-CTA single pass softmax configuration
    static const int SOFTMAX_MAX_BLOCK = 1024;
    static const int SOFTMAX_MAX_UNROLL = 16;

    // multi-CTA single pass softmax configuration
    static const int TILED_SOFTMAX_BLOCK = 256;
    static const int TILED_SOFTMAX_UNROLL = 32;
    static const int TILED_SOFTMAX_MAX_NTILES = 32;
  };

  int SoftmaxInplace(FType* QK, void* softmaxWs, cudaStream_t stream) const {
    using Config = SoftmaxConfig;
    uint32_t softmaxBatch = batch_ * nHead_;
    uint32_t softmaxLength = seqLength_;

    if (softmaxLength <= 16) {
      return LaunchSoftmaxKernel<4, 32, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 32) {
      return LaunchSoftmaxKernel<8, 16, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 64) {
      return LaunchSoftmaxKernel<16, 8, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 128) {
      return LaunchSoftmaxKernel<32, 4, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 256) {
      return LaunchSoftmaxKernel<32, 4, 8>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 512) {
      return LaunchSoftmaxKernel<64, 2, 8>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 1024) {
      return LaunchSoftmaxKernel<128, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 2048) {
      return LaunchSoftmaxKernel<256, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 4096) {
      return LaunchSoftmaxKernel<512, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 8192) {
      return LaunchSoftmaxKernel<512, 1, 16>(QK, softmaxLength, softmaxBatch,
                                             stream);
    } else if (softmaxLength <=
               Config::SOFTMAX_MAX_BLOCK * Config::SOFTMAX_MAX_UNROLL) {
      return LaunchSoftmaxKernel<Config::SOFTMAX_MAX_BLOCK, 1,
                                 Config::SOFTMAX_MAX_UNROLL>(
          QK, softmaxLength, softmaxBatch, stream);
    } else {
      return LaunchTiledSoftmaxKernel<Config::TILED_SOFTMAX_BLOCK,
                                      Config::TILED_SOFTMAX_UNROLL,
                                      Config::TILED_SOFTMAX_MAX_NTILES>(
          QK, softmaxWs, softmaxLength, softmaxBatch, stream);
    }
  }

  template <int BLOCK_X, int BLOCK_Y, int UNROLL>
  int LaunchSoftmaxKernel(FType* QK, uint32_t length, uint32_t batch,
                          cudaStream_t stream) const {
    uint32_t grid = span_attention::U32DivRU(batch, BLOCK_Y);
    uint32_t block = BLOCK_X * BLOCK_Y;
    span_attention::InplaceSoftmaxKernel<BLOCK_X, BLOCK_Y, UNROLL, FType,
                                         ComputeType>
        <<<grid, block, 0, stream>>>(QK, length, batch);
    AS_CHECK_CUDA(cudaGetLastError());
    return span_attention::RetConfig::Success;
  }

  template <int BLOCK, int UNROLL, int MAX_NTILES>
  int LaunchTiledSoftmaxKernel(FType* QK, void* softmaxWs, uint32_t length,
                               uint32_t batch, cudaStream_t stream) const {
    uint32_t nTiles = span_attention::U32DivRU(length, BLOCK * UNROLL);
    int maxBlocksPerSM;
    AS_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        span_attention::TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES,
                                                  FType, ComputeType>,
        BLOCK, 0));

    if (nTiles > MAX_NTILES || nTiles > maxBlocksPerSM * smCount_) {
      return span_attention::RetConfig::ProgramError;
    }

    // workspace
    char* wsPtr = static_cast<char*>(softmaxWs);
    ComputeType* maxReduceWs = reinterpret_cast<ComputeType*>(wsPtr);
    ComputeType* sumReduceWs =
        reinterpret_cast<ComputeType*>(wsPtr + softmaxReduceWsSize_);
    uint32_t* maxReduceFinishCount =
        reinterpret_cast<uint32_t*>(wsPtr + 2 * softmaxReduceWsSize_);
    uint32_t* sumReduceFinishCount = reinterpret_cast<uint32_t*>(
        wsPtr + 2 * softmaxReduceWsSize_ + softmaxReduceFinishCountSize_);

    // set the finish count workspace to 0
    AS_CHECK_CUDA(cudaMemsetAsync(wsPtr + 2 * softmaxReduceWsSize_, 0,
                                  2 * softmaxReduceFinishCountSize_, stream));

    span_attention::TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES, FType,
                                              ComputeType>
        <<<dim3(nTiles, batch), BLOCK, 0, stream>>>(
            QK, maxReduceWs, maxReduceFinishCount, sumReduceWs,
            sumReduceFinishCount, length, batch);
    AS_CHECK_CUDA(cudaGetLastError());
    return span_attention::RetConfig::Success;
  }

  int headSize_;
  int nHead_;
  int nGroups_;
  int headsPerGroup_;
  int seqLength_;
  int chunkSize_;
  int nChunk_;
  int batch_;

  // workspace size in byte
  size_t QKSize_;
  size_t softmaxReduceWsSize_;
  size_t softmaxReduceFinishCountSize_;
  size_t QKVReduceWsSize_;

  int smCount_;
};

#undef SPAN_ATTN_CHECK_FUNC

}  // namespace cuda
}  // namespace allspark
