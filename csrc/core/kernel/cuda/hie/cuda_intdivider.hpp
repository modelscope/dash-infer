/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_intdivider.hpp
 */

#ifndef CUDA_INTDIVIDER_HPP_
#define CUDA_INTDIVIDER_HPP_

#include <limits>

// #include "check.h"

namespace hie {

namespace internal {

template <typename T>
struct IntDivider {
  T d_;
  explicit IntDivider(T d) { d_ = d; }

  __host__ __device__ __forceinline__ T div(T n) { return n / d_; }
};

template <>
struct IntDivider<uint32_t> {
  uint32_t magic_;
  uint32_t shift_;

  IntDivider() {}

  explicit IntDivider(uint32_t d) {
    // HIE_ENFORCE(d >= 1 && d <= INT32_MAX);

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    uint64_t tmp_magic = ((1llu << 32) * ((1llu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
                         // HIE_ENFORCE(magic_ != 0 && magic_ == tmp_magic);
  }

  __host__ __device__ __forceinline__ uint32_t div(uint32_t n) {
#if defined(__CUDA_ARCH__)
    return (__umulhi(n, magic_) + n) >> shift_;
#else
    uint32_t t = (static_cast<uint64_t>(n) * magic_) >> 32;
    return (t + n) >> shift_;
#endif  // defined(__CUDA_ARCH__)
  }
};

template <typename T>
struct DivMod {
  T div;
  T mod;
  __host__ __device__ DivMod(T d, T m) : div(d), mod(m) {}
};

template <typename T>
struct IntDivModer {
  uint32_t d_;

  IntDivModer() {}
  explicit IntDivModer(T d) : d_(d) {}

  __host__ __device__ __forceinline__ T div(T n) { return n / d_; }

  __host__ __device__ __forceinline__ T mod(T n) { return n % d_; }

  __host__ __device__ __forceinline__ DivMod<T> divmod(T n) {
    return DivMod<T>(n / d_, n % d_);
  }
};

template <>
struct IntDivModer<uint32_t> {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  IntDivModer() {}

  explicit IntDivModer(uint32_t d) {
    // HIE_ENFORCE(d >= 1 && d <= INT32_MAX);
    d_ = d;

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    // xunle : in MSVC sizeof(long) is 4 bytes , which causes overflow
    uint64_t tmp_magic = ((1llu << 32) * ((1llu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
                         // HIE_ENFORCE(magic_ != 0 && magic_ == tmp_magic);
  }

  __host__ __device__ __forceinline__ uint32_t div(uint32_t n) {
#if defined(__CUDA_ARCH__)
    return (__umulhi(n, magic_) + n) >> shift_;
#else
    uint32_t t = (static_cast<uint64_t>(n) * magic_) >> 32;
    return (t + n) >> shift_;
#endif  // defined(__CUDA_ARCH__)
  }

  __host__ __device__ __forceinline__ uint32_t mod(uint32_t n) {
    return n - div(n) * d_;
  }

  __host__ __device__ __forceinline__ DivMod<uint32_t> divmod(uint32_t n) {
    uint32_t d = div(n);
    return DivMod<uint32_t>(d, n - d_ * d);
  }
};

using U32DivModer = IntDivModer<uint32_t>;

}  // namespace internal

}  // namespace hie

#endif  // CUDA_INTDIVIDER_HPP_
