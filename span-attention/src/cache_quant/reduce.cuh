/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "common/data_type.h"
#include "common/fp_math.cuh"
#include "utils.cuh"
#include "utils/shuffle.cuh"

namespace span {

namespace qcache {
namespace impl {

namespace functor {

/* ===== max ===== */
template <typename T>
struct Max {
  DEVICE_FUNC T operator()(const T a, const T b) const { return max(a, b); }
};

/* ===== min ===== */
template <typename T>
struct Min {
  DEVICE_FUNC T operator()(const T a, const T b) const { return min(a, b); }
};

/* ===== sum ===== */
template <typename T>
struct Sum {
  DEVICE_FUNC T operator()(const T a, const T b) const { return a + b; }
};

}  // namespace functor

namespace transform {

template <typename MathT>
struct Default {
  DEVICE_FUNC MathT operator()(const MathT x) const { return x; }
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

/* ===== min ===== */
template <typename T>
struct IdentityElem<functor::Min, T> {
  static constexpr T v = std::numeric_limits<T>::has_infinity
                             ? std::numeric_limits<T>::infinity()
                             : std::numeric_limits<T>::max();
};

/* ===== sum ===== */
template <typename T>
struct IdentityElem<functor::Sum, T> {
  static constexpr T v = 0;
};

/**
 * @brief Warp reduce, lane 0 will return the final result.
 */
template <template <typename> typename Func, typename ReduceT>
DEVICE_FUNC ReduceT warpReduce(const ReduceT& threadRes) {
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
DEVICE_FUNC ReduceT blockReduce(const ReduceT& warpRes) {
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

}  // namespace span
