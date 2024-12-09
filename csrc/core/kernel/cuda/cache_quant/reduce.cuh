/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "../utils/shuffle.cuh"
#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace qcache {
namespace impl {

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
struct Max<hie::bfloat16> {
  __device__ __forceinline__ hie::bfloat16 operator()(
      const hie::bfloat16 a, const hie::bfloat16 b) const {
    return fmax_bf16(a, b);
  }
};
#endif

/* ===== min ===== */
template <typename T>
struct Min {
  __device__ __forceinline__ T operator()(const T a, const T b) const {
    return min(a, b);
  }
};

#ifdef ENABLE_FP16
template <>
struct Min<half> {
  __device__ __forceinline__ half operator()(const half a, const half b) const {
#if (__CUDA_ARCH__ >= 800) && (CUDA_VERSION >= 11000)
    return __hmin(a, b);
#else
    return a > b ? a : b;
#endif
  }
};
#endif

#ifdef ENABLE_BF16
template <>
struct Min<hie::bfloat16> {
  __device__ __forceinline__ hie::bfloat16 operator()(
      const hie::bfloat16 a, const hie::bfloat16 b) const {
    return fmin_bf16(a, b);
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
struct IdentityElem<functor::Max, hie::bfloat16> {
  static constexpr float v = -std::numeric_limits<float>::infinity();
};
#endif

/* ===== min ===== */
template <typename T>
struct IdentityElem<functor::Min, T> {
  static constexpr T v = std::numeric_limits<T>::has_infinity
                             ? std::numeric_limits<T>::infinity()
                             : std::numeric_limits<T>::max();
};

#ifdef ENABLE_FP16
template <>
struct IdentityElem<functor::Min, half> {
  static constexpr float v = std::numeric_limits<float>::infinity();
};
#endif

#ifdef ENABLE_BF16
template <>
struct IdentityElem<functor::Min, hie::bfloat16> {
  static constexpr float v = std::numeric_limits<float>::infinity();
};
#endif

/* ===== sum ===== */
template <typename T>
struct IdentityElem<functor::Sum, T> {
  static constexpr T v = 0;
};

/**
 * @brief Warp reduce, lane 0 will return the final result.
 */
template <template <typename> typename Func, typename ReduceT>
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

}  // namespace impl
}  // namespace qcache
}  // namespace cuda
}  // namespace allspark
