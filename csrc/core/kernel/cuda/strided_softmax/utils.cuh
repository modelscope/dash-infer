/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include <cstdint>
#if __cplusplus >= 201703L
#include <type_traits>
#endif

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_BF16
#include <hie_bfloat16.hpp>
#endif

namespace allspark {
namespace cuda {
namespace strided_softmax {

#ifdef ENABLE_BF16
using bfloat16 = hie::bfloat16;
#endif

constexpr uint32_t MAX_ALIGNMENT = 16;

constexpr int WARP_SIZE = 32;
constexpr int CTA_MAX_SIZE = 1024;
constexpr int CTA_WARP_MAX = CTA_MAX_SIZE / WARP_SIZE;

template <typename T>
__host__ __device__ inline constexpr T UIntDivUp(const T& x, const T& y) {
  return (x + y - 1) / y;
}

template <typename T>
__host__ __device__ inline constexpr bool IsPowOf2(const T& x) {
  return (x & (x - 1)) == 0;
}

template <uint32_t ALIGNMENT, typename U>
__host__ __device__ inline U AlignUpPow2(U x) {
  static_assert(ALIGNMENT > 0 && ((ALIGNMENT & (ALIGNMENT - 1)) == 0),
                "ALIGNMENT must be a positive power of 2");
  return (x + ALIGNMENT - 1) & (~(ALIGNMENT - 1));
}

// ------------------------------------
// Load & store
// ------------------------------------
template <typename T>
struct EmptyLoadFunctor {
#if __cplusplus < 201703L
  __device__ __forceinline__ T operator()(const int) const { return T(); }
#endif  // __cplusplus < 201703L
};

template <int UNROLL, typename T, typename LoadFunc,
          typename DefaultFunc = EmptyLoadFunctor<T>>
__device__ __forceinline__ void loadRegs(
    T (&regs)[UNROLL], const uint32_t count, LoadFunc loadFunc,
    DefaultFunc defaultFunc = DefaultFunc()) {
  if (count >= UNROLL) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      regs[i] = loadFunc(i);
    }
  } else {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < count) {
        regs[i] = loadFunc(i);
      } else {
#if __cplusplus >= 201703L
        // C++17 helps reduce empty constructor, e.g. PRMT in SASS
        if constexpr (!std::is_same_v<DefaultFunc, EmptyLoadFunctor<T>>)
#endif  // __cplusplus >= 201703L
          regs[i] = defaultFunc(i);
      }
    }
  }
  return;
}

template <int UNROLL, typename ST, typename DT, typename IdxFunc>
__device__ __forceinline__ void storeRegs(DT* dst, const ST (&regs)[UNROLL],
                                          const uint32_t count,
                                          IdxFunc idxFunc) {
  if (count >= UNROLL) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      dst[idxFunc(i)] = static_cast<DT>(regs[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < count) {
        dst[idxFunc(i)] = static_cast<DT>(regs[i]);
      }
    }
  }
  return;
}

}  // namespace strided_softmax
}  // namespace cuda
}  // namespace allspark
