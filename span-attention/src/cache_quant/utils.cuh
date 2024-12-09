/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include "common/func_modifier.h"
#include "config.cuh"

namespace span {

namespace qcache {
namespace impl {

constexpr int WARP_SIZE = 32;
constexpr int CTA_MAX_SIZE = 1024;
constexpr int CTA_WARP_MAX = CTA_MAX_SIZE / WARP_SIZE;

// ------------------------------------
// Math: Div
// ------------------------------------

template <typename T>
DEVICE_FUNC T Div(const T a, const T b) {
  return a / b;
}

template <>
DEVICE_FUNC float Div(const float a, const float b) {
  return __fdividef(a, b);
}

/**
 * @brief If CONFIG_CACHE_ROUND_RNI is defined, behaves as rintf; otherwise,
 * behaves as roundf. If T is not float, extra type conversion is implied.
 */
template <typename T>
DEVICE_FUNC T Rounding(const T x) {
#ifdef CONFIG_CACHE_ROUND_RNI
  return rintf(x);
#else
  return roundf(x);
#endif
}

}  // namespace impl
}  // namespace qcache

}  // namespace span
