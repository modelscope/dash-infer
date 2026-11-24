// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "core.hpp"
#include <type_traits>

namespace r2sk {
// Storage for raw IEEE BF16 bits; no BF16 arithmetic or SM80 instructions.
// Finite inputs only. This type does NOT round arbitrary FP32 inputs to BF16.
struct BFloat16 {
  uint16_t bits = 0;
  R2SK_HD operator float() const {
    uint32_t word = uint32_t(bits) << 16;
#ifdef __CUDA_ARCH__
    return __uint_as_float(word);
#else
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
#endif
  }
  // Internal sorting/staging stores only values loaded from BF16, or an
  // exactly representable padding sentinel. Their low 16 FP32 bits are zero.
  R2SK_HD BFloat16& operator=(float exact) {
    bits = uint16_t(float_bits(exact) >> 16);
    return *this;
  }
};
static_assert(sizeof(BFloat16) == 2 && alignof(BFloat16) == 2,
              "BF16 storage layout");
static_assert(std::is_trivially_copyable<BFloat16>::value, "BF16 raw copies");
}  // namespace r2sk
