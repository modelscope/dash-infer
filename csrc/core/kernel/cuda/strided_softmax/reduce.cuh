/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "intrinsics.cuh"

namespace allspark {
namespace cuda {
namespace strided_softmax {
namespace reduce {

namespace functor {

/* ===== max ===== */
template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(const T a, const T b) const {
    return max(a, b);
  }
};

#ifdef ENABLE_FP16
template <>
struct Max<half> {
  __device__ __forceinline__ half operator()(const half a, const half b) const {
#if (__CUDA_ARCH__ >= 800) && (CUDA_VERSION >= 11000)
    return __hmax(a, b);
#else
    return a > b ? a : b;
#endif
  }
};
#endif

#ifdef ENABLE_BF16
template <>
struct Max<bfloat16> {
  __device__ __forceinline__ bfloat16 operator()(const bfloat16 a,
                                                 const bfloat16 b) const {
    return fmax_bf16(a, b);
  }
};
#endif

/* ===== sum ===== */
template <typename T>
struct Sum {
  __device__ __forceinline__ T operator()(const T a, const T b) const {
    return a + b;
  }
};

}  // namespace functor

namespace transform {

template <typename MathT>
struct Default {
  __device__ __forceinline__ MathT operator()(const MathT x) const { return x; }
};

}  // namespace transform

// ------------------------------------
// initialization
// ------------------------------------
template <template <typename> typename Func, typename T>
struct IdentityElem;

/* ===== max ===== */
template <typename T>
struct IdentityElem<functor::Max, T> {
  static constexpr T v = std::numeric_limits<T>::has_infinity
                             ? -std::numeric_limits<T>::infinity()
                             : std::numeric_limits<T>::lowest();
};

#ifdef ENABLE_FP16
template <>
struct IdentityElem<functor::Max, half> {
  static constexpr float v = -std::numeric_limits<float>::infinity();
};
#endif

#ifdef ENABLE_BF16
template <>
struct IdentityElem<functor::Max, bfloat16> {
  static constexpr float v = -std::numeric_limits<float>::infinity();
};
#endif

/* ===== sum ===== */
template <typename T>
struct IdentityElem<functor::Sum, T> {
  static constexpr T v = 0;
};

// ------------------------------------
// Helpers
// ------------------------------------
template <typename ST, typename DT>
__device__ __forceinline__ void writeBack(DT* dstArray, const ST& val,
                                          const uint32_t stride) {
  DT* writePtr = dstArray + blockIdx.y * stride;
  writePtr[blockIdx.x] = static_cast<DT>(val);
  return;
}

/**
 * @brief Warp reduce, lane 0 will return the final result.
 */
template <int BLOCK, template <typename> typename Func, typename ReduceT>
__device__ __forceinline__ ReduceT warpReduce(const ReduceT& threadRes) {
  ReduceT warpRes = threadRes;
#pragma unroll
  for (int i = WARP_SIZE; i > 1; i >>= 1) {
    // reduce shfl width per iteration
    ReduceT var = ShflBfly(0xffffffff, warpRes, i >> 1, i);
    warpRes = Func<ReduceT>()(warpRes, var);
  }
  return warpRes;
}

/**
 * @brief Block reduce, thread 0 will return the final result.
 */
template <int BLOCK, template <typename> typename Func, typename ReduceT>
__device__ __forceinline__ ReduceT blockReduce(const ReduceT& warpRes) {
  static_assert(CTA_WARP_MAX == WARP_SIZE,
                "invalid CUDA architecture definition");
  __shared__ ReduceT smem[CTA_WARP_MAX];

  constexpr int N_WARPS = BLOCK / WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  const int laneId = threadIdx.x % WARP_SIZE;

  // each lane 0 holds warp result
  if (laneId == 0) {
    smem[warpId] = warpRes;
  }

  if (warpId == 0) {
    if (laneId >= N_WARPS) {
      smem[laneId] = IdentityElem<Func, ReduceT>::v;
    }
  }
  __syncthreads();

  ReduceT blockRes = 0;
  if (warpId == 0) {
    blockRes = smem[laneId];
#pragma unroll
    for (int i = WARP_SIZE; i > 1; i >>= 1) {
      // reduce shfl width per iteration
      ReduceT var = ShflBfly(0xffffffff, blockRes, i >> 1, i);
      blockRes = Func<ReduceT>()(blockRes, var);
    }
  }
  return blockRes;
}

// ------------------------------------
// Kernel
// ------------------------------------
template <int BLOCK, int UNROLL, bool WRITE_WS,
          template <typename> typename ReduceFunc,
          template <typename> typename PreFunc =
              allspark::cuda::strided_softmax::reduce::transform::Default,
          template <typename> typename PostFunc =
              allspark::cuda::strided_softmax::reduce::transform::Default,
          typename ST, typename DT, typename ReduceT, typename MathT = ReduceT>
__global__ void __launch_bounds__(512)
    reduceKernel(const ST* x, DT* y, ReduceT* globalBlockRes,
                 const int* taskLenPtr, const uint32_t stride,
                 PreFunc<MathT> preFunc = PreFunc<MathT>(),
                 PostFunc<ReduceT> postFunc = PostFunc<ReduceT>()) {
  static_assert(BLOCK >= WARP_SIZE && BLOCK % WARP_SIZE == 0,
                "reduceMaxKernel requires BLOCK >= WARP_SIZE "
                "and BLOCK % WARP_SIZE == 0");

  const uint32_t taskId = blockIdx.y;
  const uint32_t taskLen = taskLenPtr != nullptr ? taskLenPtr[taskId] : stride;
  const uint32_t stepSize = BLOCK * UNROLL * gridDim.x;
  const uint32_t nSteps = UIntDivUp<uint32_t>(taskLen, stepSize);

  ReduceT finalRes = IdentityElem<ReduceFunc, ReduceT>::v;

  for (uint32_t step = 0; step < nSteps; ++step) {
    const uint32_t offset0 =
        threadIdx.x + blockIdx.x * BLOCK * UNROLL + step * stepSize;
    const uint32_t xCount =
        offset0 < taskLen ? UIntDivUp<uint32_t>(taskLen - offset0, BLOCK) : 0;
    const ST* xPtr = x + taskId * stride + offset0;

    // ----------------------
    // phase 1: load data
    // ----------------------
    ST xRegs[UNROLL];
    loadRegs<UNROLL>(xRegs, xCount,
                     [xPtr](const int i) { return xPtr[i * BLOCK]; });

    // ----------------------
    // phase 2: preprocess
    // ----------------------
    ReduceT reduceRegs[UNROLL];
    loadRegs<UNROLL>(
        reduceRegs, xCount,
        [&preFunc, &xRegs](const int i) {
          MathT var = preFunc(static_cast<MathT>(xRegs[i]));
          return static_cast<ReduceT>(var);
        },
        [&xRegs](const int i) { return IdentityElem<ReduceFunc, ReduceT>::v; });

    // ----------------------
    // phase 3: thread reduce
    // ----------------------
    ReduceT threadRes = IdentityElem<ReduceFunc, ReduceT>::v;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      threadRes = ReduceFunc<ReduceT>()(threadRes, reduceRegs[i]);
    }

    // ----------------------
    // phase 4: reduce
    // ----------------------
    ReduceT warpRes = warpReduce<BLOCK, ReduceFunc>(threadRes);

    ReduceT blockRes = warpRes;
    constexpr int N_WARPS = BLOCK / WARP_SIZE;
    if (N_WARPS > 1) {
      blockRes = blockReduce<BLOCK, ReduceFunc>(blockRes);
    }

    // ----------------------
    // phase 5: update res
    // ----------------------
    finalRes = ReduceFunc<ReduceT>()(finalRes, blockRes);
  }

  // final write back
  if (threadIdx.x == 0) {
    finalRes = postFunc(finalRes);
    if (WRITE_WS) {
      writeBack(globalBlockRes, finalRes, gridDim.x);
    } else {
      writeBack(y, finalRes, 1);
    }
  }
  return;
}

template <int BLOCK, int UNROLL, typename ReduceT>
void getReduceWorkspaceSize(size_t* wsInBytes, const int taskNum,
                            const int length) {
  uint32_t nBlocks = UIntDivUp<uint32_t>(length, BLOCK * UNROLL);
  *wsInBytes = sizeof(ReduceT) * taskNum * nBlocks;
  return;
}

}  // namespace reduce
}  // namespace strided_softmax
}  // namespace cuda
}  // namespace allspark
