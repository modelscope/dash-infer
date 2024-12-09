/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.hpp
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "common/func_modifier.h"
#include "utils/intrinsic.cuh"

namespace span {

template <typename A, typename B>
struct CheckSameType {
  static_assert(std::is_same<A, B>::value, "A and B must be the same type");
};

template <int ALIGN>
HOST_DEVICE_FUNC constexpr size_t AlignedMemSize(const size_t s) {
  return (s + ALIGN - 1) / ALIGN * ALIGN;
}
HOST_DEVICE_FUNC constexpr bool IsPowerOf2(const uint32_t x) {
  return (x & (x - 1)) == 0;
}
HOST_DEVICE_FUNC constexpr uint32_t RoundUpPower2(const uint32_t x) {
  uint32_t v = x;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
HOST_DEVICE_FUNC constexpr uint32_t U32DivRU(const uint32_t x,
                                             const uint32_t y) {
  return (x + y - 1) / y;
}

template <typename T>
struct DivModT {
  T div;
  T mod;
  HOST_DEVICE_FUNC DivModT(T d, T m) : div(d), mod(m) {}
};

struct U32DivMod {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  U32DivMod() = default;

  explicit U32DivMod(uint32_t d) : d_(d) {
    assert(d >= 1 && d <= INT32_MAX);

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
    assert(magic_ != 0 && magic_ == tmp_magic);
  }

  DEVICE_FUNC uint32_t Div(uint32_t n) {
#ifdef __CUDACC__
    return (__umulhi(n, magic_) + n) >> shift_;
#else
    uint64_t tmp = (static_cast<uint64_t>(n) * magic_) >> 32;
    return (static_cast<uint32_t>(tmp) + n) >> shift_;
#endif
  }

  DEVICE_FUNC DivModT<uint32_t> DivMod(uint32_t n) {
    uint32_t d = Div(n);
    return DivModT<uint32_t>(d, n - d_ * d);
  }
};

}  // namespace span
