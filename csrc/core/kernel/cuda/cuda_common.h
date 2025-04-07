/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_common.h
 */

#pragma once

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <stdint.h>

#include <iostream>
#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_BF16
#include "bfloat16_impl.hpp"
#include "hie_bfloat16.hpp"
#endif

#include <common.h>
#include <cublas_v2.h>
#include <hiednn_cuda.h>

#include "cuda/hie/enum.h"
#include "cuda_extension.hpp"
#ifdef ENABLE_FP16
__device__ __forceinline__ float operator*(const half& lh, const float& rh) {
  return (float)lh * rh;
}
__device__ __forceinline__ float operator*(const float& lh, const half& rh) {
  return lh * (float)rh;
}
__device__ __forceinline__ float operator+(const half& lh, const float& rh) {
  return (float)lh + rh;
}
__device__ __forceinline__ float operator+(const float& lh, const half& rh) {
  return lh + (float)rh;
}
#endif
#define THREAD_PER_BLOCK 256
#define MAX_DIMS 6

namespace hie {
template <typename T, int nbElems>
struct Array {
  __host__ __device__ inline T& operator[](unsigned i) { return data[i]; }
  __host__ __device__ inline const T& operator[](unsigned i) const {
    return data[i];
  }
  __host__ __device__ inline constexpr int size() const { return nbElems; }
  __host__ __device__ inline T mul() const {
    T value = 1;
    for (int i = 0; i < nbElems; ++i) {
      value *= data[i];
    }
    return value;
  }
  __host__ __device__ Array() {
#ifndef __CUDA_ARCH__
    for (int i = 0; i < nbElems; i++) {
      data[i] = T();
    }
#endif
  }

  // constructors with arguments only work for host
  template <typename S>
  __host__ Array(const S* src, int n) {
    for (int i = 0; i < n; ++i) {
      data[i] = static_cast<T>(src[i]);
    }
  }

  template <typename S>
  __host__ Array(const S* src, int n, const S& fill) {
    for (int i = 0; i < n; ++i) {
      data[i] = static_cast<T>(src[i]);
    }
    for (int i = n; i < nbElems; ++i) {
      data[i] = static_cast<T>(fill);
    }
  }

  T data[nbElems];
};

/**
 * @brief active
 *
 */

#define ACTIVE_FUNC(func, expr)                                          \
  template <typename T>                                                  \
  struct func {                                                          \
    __forceinline__ __device__ T operator()(const T& x) { return expr; } \
  };

ACTIVE_FUNC(NONE, x);
ACTIVE_FUNC(RELU, x < static_cast<T>(0.0f) ? static_cast<T>(0.0f) : x);
ACTIVE_FUNC(TANH, cmath_tanh(x));
ACTIVE_FUNC(GELU, x * 0.5f * (1.0f + erff(x * 0.70710678f)));
ACTIVE_FUNC(GELU_TANH, x * 0.5f *
                           (1.0f + tanhf((0.7978845608028654f *
                                          (x + 0.044715f * x * x * x)))));

#undef ACTIVE_FUNC

template <template <typename> class ActiveOp>
struct GetActiveOp {
  template <typename T, typename... Arg>
  ActiveOp<T> get(Arg&&... args) {
    return ActiveOp<T>(std::forward<Arg>(args)...);
  }
};
}  // namespace hie
