/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "attn/common/utils.hpp"
#include "utils/shuffle.cuh"

namespace span {

DEVICE_FUNC float Expf(const float& x) { return __expf(x); }

DEVICE_FUNC float Rcpf(const float& x) {
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
  DEVICE_FUNC float Init() {
    uint32_t RAW_NEG_INF = 0xff800000;
    return reinterpret_cast<const float&>(RAW_NEG_INF);
  };

  DEVICE_FUNC float Reduce(const float& x, const float& y) {
    return fmaxf(x, y);
  }
};

template <typename T>
struct SumReduceFunctor {
  DEVICE_FUNC T Init() { return T(0); }

  DEVICE_FUNC T Reduce(const T& x, const T& y) { return x + y; }
};

template <int BLOCK_X, int BLOCK_Y, int UNROLL,
          template <typename> class ReduceFunc, typename T>
DEVICE_FUNC T BlockReduce(const T (&x)[UNROLL]) {
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
  constexpr int SHFL_WIDTH_0 = BLOCK_X > 32 ? 32 : BLOCK_X;
#pragma unroll
  for (int i = SHFL_WIDTH_0; i > 1; i /= 2) {
    ret = f.Reduce(ret, ShflBfly(0xffffffff, ret, i / 2, SHFL_WIDTH_0));
  }

  // block reduction
  if constexpr (BLOCK_X > 32) {
    uint32_t laneId = threadIdx.x % 32;
    uint32_t warpId = threadIdx.x / 32;

    smem[warpId] = ret;
    __syncthreads();

    if (warpId == 0) {
      // set SHFL_WIDTH_1 to 1 in case of BLOCK_X<=32 to avoid compiler
      // warning
      constexpr int SHFL_WIDTH_1 = BLOCK_X > 32 ? BLOCK_X / 32 : 1;
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

// ===========================

/**
 * reduceWs: workspace for reduction, should not be reused for multiple
 *           reductions of a kernel
 * finishCount: finished tile counter, the counter should set to 0 before
 *              kernel launch, and should not be reused for multiple reductions
 *              of a kernel
 */
template <int BLOCK, int UNROLL, int MAX_NTILES,
          template <typename> class ReduceFunc, typename T>
DEVICE_FUNC T GridXReduce(const T (&x)[UNROLL], T* reduceWs,
                          uint32_t* finishCount, uint32_t batchId,
                          uint32_t tileId, uint32_t nTiles,
                          uint32_t maxNTiles) {
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
  if constexpr (BLOCK > 32) {
    reduceSmem[warpId] = acc;
    __syncthreads();

    if (warpId == 0) {
      // set SHFL_WIDTH_1 to 1 in case of BLOCK<=32 to avoid compiler
      // warning
      constexpr int SHFL_WIDTH_1 = BLOCK > 32 ? BLOCK / 32 : 1;
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
  T* reduceWsPtr = reduceWs + batchId * maxNTiles;
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

}  // namespace span
