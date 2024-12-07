/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax.cuh
 */

#pragma once

#include "../utils/shuffle.cuh"
#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace span_attention {

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
 *              kernel launch, and should not be reused for multiple reductions
 *              of a kernel
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

}  // namespace span_attention
}  // namespace cuda
}  // namespace allspark
