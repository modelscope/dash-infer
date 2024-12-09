/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_extension.hpp
 */

#ifndef CUDA_EXTENSION_HPP_
#define CUDA_EXTENSION_HPP_
#include <cstdint>

// #define warpSize 32
constexpr int cuda_warp_thread = 32;
// #ifdef HIE_ENABLE_FLOAT16
// #include "hie_half.hpp"

// #if defined(__CUDA_ARCH__)
// __device__ __forceinline__ hie::float16 __ldg(const hie::float16* ptr) {
//     __half v = __ldg(reinterpret_cast<const __half*>(ptr));
//     return reinterpret_cast<const hie::float16&>(v);
// }

// __device__ __forceinline__ hie::float16 __shfl_down_sync(unsigned mask,
// hie::float16 var,
//                                                          unsigned int delta,
//                                                          int width =
//                                                          cuda_warp_thread) {
//     __half v = __shfl_down_sync(mask, var.__x.__x.f, delta, width);
//     return reinterpret_cast<const hie::float16&>(v);
// }
// #endif
// #endif

#ifdef ENABLE_BF16
#include "../../../common/hie_bfloat16.hpp"
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ hie::bfloat16 __ldg(const hie::bfloat16* ptr) {
  __nv_bfloat16 v = __ldg(reinterpret_cast<const __nv_bfloat16*>(ptr));
  return reinterpret_cast<const hie::bfloat16&>(v);
}

__device__ __forceinline__ hie::bfloat16 __shfl_down_sync(
    unsigned mask, hie::bfloat16 var, unsigned int delta,
    int width = cuda_warp_thread) {
  __nv_bfloat16 v = __shfl_down_sync(mask, var.__x.__x.f, delta, width);
  return reinterpret_cast<const hie::bfloat16&>(v);
}
#else
__device__ __forceinline__ hie::bfloat16 __ldg(const hie::bfloat16* ptr) {
  // BFLOAT16 TODO
  asm("trap;");
  return 0.0f;
}

__device__ __forceinline__ hie::bfloat16 __shfl_down_sync(
    unsigned mask, hie::bfloat16 var, unsigned int delta,
    int width = cuda_warp_thread) {
  // BFLOAT16 TODO
  asm("trap;");
  return 0.0f;
}
#endif
#endif

#endif  // CUDA_EXTENSION_HPP_
