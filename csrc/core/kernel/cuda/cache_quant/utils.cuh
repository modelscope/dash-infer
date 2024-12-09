/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include <cuda_runtime.h>

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif  // ENABLE_FP16

#ifdef ENABLE_BF16
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#endif  // ENABLE_BF16

#include "config.cuh"

namespace allspark {
namespace cuda {
namespace qcache {
namespace impl {

constexpr int WARP_SIZE = 32;
constexpr int CTA_MAX_SIZE = 1024;
constexpr int CTA_WARP_MAX = CTA_MAX_SIZE / WARP_SIZE;

// ------------------------------------
// Math: Div
// ------------------------------------

template <typename T>
__device__ __forceinline__ T Div(const T a, const T b) {
  return a / b;
}

template <>
__device__ __forceinline__ float Div(const float a, const float b) {
  return __fdividef(a, b);
}

#ifdef ENABLE_FP16
template <>
__device__ __forceinline__ __half Div(const __half a, const __half b) {
  return __hdiv(a, b);
}
#endif  // ENABLE_FP16

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ hie::bfloat16 Div(const hie::bfloat16 a,
                                             const hie::bfloat16 b) {
  return a / b;
}
#endif  // ENABLE_BF16

/**
 * @brief If CONFIG_CACHE_ROUND_RNI is defined, behaves as rintf; otherwise,
 * behaves as roundf. If T is not float, extra type conversion is implied.
 */
template <typename T>
__device__ __forceinline__ T Rounding(const T x) {
#ifdef CONFIG_CACHE_ROUND_RNI
  return rintf(x);
#else
  return roundf(x);
#endif
}

}  // namespace impl
}  // namespace qcache
}  // namespace cuda
}  // namespace allspark
