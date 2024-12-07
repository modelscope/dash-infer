/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "../utils/intrinsic.cuh"

namespace allspark {
namespace cuda {
namespace span_attention {

__host__ __device__ inline uint32_t U32DivRU(uint32_t x, uint32_t y) {
  return (x + y - 1) / y;
}

template <typename T>
struct DivModT {
  T div;
  T mod;
  __host__ __device__ DivModT(T d, T m) : div(d), mod(m) {}
};

struct U32DivMod {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  U32DivMod() {}

  explicit U32DivMod(uint32_t d) : d_(d) {
    assert(d >= 1 && d <= INT32_MAX);

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit

    assert(magic_ != 0 && magic_ == tmp_magic);
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) {
    return (__umulhi(n, magic_) + n) >> shift_;
  }

  __device__ __forceinline__ DivModT<uint32_t> DivMod(uint32_t n) {
    uint32_t d = Div(n);
    return DivModT<uint32_t>(d, n - d_ * d);
  }
};

}  // namespace span_attention
}  // namespace cuda
}  // namespace allspark
