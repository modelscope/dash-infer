/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    shuffle.cuh
 */

#pragma once

#include <cassert>
#include <cstdint>

namespace allspark {
namespace cuda {
namespace intrinsic {

__device__ __forceinline__ uint32_t ShflBflyImpl(uint32_t mask, uint32_t var,
                                                 uint32_t laneMask,
                                                 uint32_t width) {
  constexpr int WARP_SIZE = 32;
  const uint32_t SHFL_C = ((WARP_SIZE - width) << 8) | (WARP_SIZE - 1);
  uint32_t ret;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(ret)
               : "r"(var), "r"(laneMask), "r"(SHFL_C), "r"(mask));
  return ret;
}

__device__ __forceinline__ uint32_t ShflIdxImpl(uint32_t mask, uint32_t var,
                                                uint32_t srcLane,
                                                uint32_t width) {
  constexpr int WARP_SIZE = 32;
  const uint32_t SHFL_C = ((WARP_SIZE - width) << 8) | (WARP_SIZE - 1);

  uint32_t ret;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(ret)
               : "r"(var), "r"(srcLane), "r"(SHFL_C), "r"(mask));
  return ret;
}

}  // namespace intrinsic

template <int SIZE>
struct ShflUIntT {
  using Type = uint32_t;
};

template <>
struct ShflUIntT<1> {
  using Type = uint8_t;
};

template <>
struct ShflUIntT<2> {
  using Type = uint16_t;
};

template <typename T>
__device__ __forceinline__ T ShflBfly(uint32_t mask, const T& var,
                                      uint32_t laneMask, uint32_t width) {
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                "sizeof(T) must be power of 2");

  T ret;
  using ShflT = typename ShflUIntT<sizeof(T)>::Type;
  const ShflT* x = reinterpret_cast<const ShflT*>(&var);
  ShflT* y = reinterpret_cast<ShflT*>(&ret);
#pragma unroll
  for (int i = 0; i < static_cast<int>(sizeof(T) / sizeof(ShflT)); ++i) {
    y[i] = intrinsic::ShflBflyImpl(mask, static_cast<uint32_t>(x[i]), laneMask,
                                   width);
  }
  return ret;
}

template <typename T>
__device__ __forceinline__ T ShflIdx(uint32_t mask, const T& var,
                                     uint32_t srcLane, uint32_t width) {
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                "sizeof(T) must be power of 2");

  T ret;
  using ShflT = typename ShflUIntT<sizeof(T)>::Type;
  const ShflT* x = reinterpret_cast<const ShflT*>(&var);
  ShflT* y = reinterpret_cast<ShflT*>(&ret);

#pragma unroll
  for (int i = 0; i < static_cast<int>(sizeof(T) / sizeof(ShflT)); ++i) {
    y[i] = intrinsic::ShflIdxImpl(mask, static_cast<uint32_t>(x[i]), srcLane,
                                  width);
  }

  return ret;
}

}  // namespace cuda
}  // namespace allspark
