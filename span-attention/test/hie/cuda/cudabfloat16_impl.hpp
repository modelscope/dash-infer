/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudabfloat16_impl.hpp
 */

#ifndef UTILS_GPU_CUDABFLOAT16_IMPL_HPP_
#define UTILS_GPU_CUDABFLOAT16_IMPL_HPP_

#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#include <cstdint>
#include <limits>

#if defined(__CUDA_ARCH__)

#define __BF16_IMPL
#define __BF16_DEVICE_FUNC __device__ __forceinline__
#define __BF16_DEVICE_FUNC_DECL __device__
#define __CUBF16_IMPL static __device__ __forceinline__
#define __CUBF16_DECL static __device__W
#define __BF16_NATIVE (__CUDA_ARCH__ >= 800) && (CUDA_VERSION >= 11000)

namespace __hie_buildin {

// #ifdef HIE_ENABLE_FLOAT16
// struct half;
// #endif

struct __attribute__((aligned(2))) __Bf16Impl {
  union F16 {
    __BF16_DEVICE_FUNC F16() {}
    __BF16_DEVICE_FUNC F16(uint16_t val) : i{val} {}
    uint16_t i;
#if __BF16_NATIVE
    __BF16_DEVICE_FUNC F16(__nv_bfloat16 val) : f{val} {}
    __nv_bfloat16 f;  // bfloat16 type in cuda_bp16.hpp
#endif
  };
  F16 __x;

  __BF16_DEVICE_FUNC __Bf16Impl() {}

  __BF16_DEVICE_FUNC explicit __Bf16Impl(uint16_t v) { __x.i = v; }

  __CUBF16_IMPL __Bf16Impl from_bits(uint16_t bits) { return __Bf16Impl(bits); }

  // #ifdef HIE_ENABLE_FLOAT16
  //     __CUBF16_DECL __Bf16Impl half2bfloat16(half v);
  // #endif

#if __BF16_NATIVE

  __BF16_DEVICE_FUNC explicit __Bf16Impl(__nv_bfloat16 v) { __x.f = v; }

  __CUBF16_IMPL __Bf16Impl float2bfloat16(float v) {
    return __Bf16Impl(__float2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl double2bfloat16(double v) {
    return __Bf16Impl(__float2bfloat16_rn(static_cast<float>(v)));
  }

  __CUBF16_IMPL __Bf16Impl ll2bfloat16(long long v) {
    return __Bf16Impl(__ll2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl ull2bfloat16(unsigned long long v) {
    return __Bf16Impl(__ull2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl int2bfloat16(int v) {
    return __Bf16Impl(__int2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl uint2bfloat16(unsigned int v) {
    return __Bf16Impl(__uint2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl short2bfloat16(short v) {
    return __Bf16Impl(__short2bfloat16_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl ushort2bfloat16(unsigned short v) {
    return __Bf16Impl(__ushort2bfloat16_rn(v));
  }

  __CUBF16_IMPL float bfloat162float(__Bf16Impl v) {
    return __bfloat162float(v.__x.f);
  }

  __CUBF16_IMPL double bfloat162double(__Bf16Impl v) {
    return static_cast<double>(__bfloat162float(v.__x.f));
  }

  __CUBF16_IMPL long long bfloat162ll(__Bf16Impl v) {
    return __bfloat162ll_rz(v.__x.f);
  }

  __CUBF16_IMPL unsigned long long bfloat162ull(__Bf16Impl v) {
    return __bfloat162ull_rz(v.__x.f);
  }

  __CUBF16_IMPL int bfloat162int(__Bf16Impl v) {
    return __bfloat162int_rz(v.__x.f);
  }

  __CUBF16_IMPL unsigned int bfloat162uint(__Bf16Impl v) {
    return __bfloat162uint_rz(v.__x.f);
  }

  __CUBF16_IMPL short bfloat162short(__Bf16Impl v) {
    return __bfloat162short_rz(v.__x.f);
  }

  __CUBF16_IMPL unsigned short bfloat162ushort(__Bf16Impl v) {
    return __bfloat162ushort_rz(v.__x.f);
  }

  // + - * /
  __CUBF16_IMPL __Bf16Impl bf16add(__Bf16Impl a, __Bf16Impl b) {
    return __Bf16Impl(__hadd(a.__x.f, b.__x.f));
  }
  __CUBF16_IMPL __Bf16Impl bf16sub(__Bf16Impl a, __Bf16Impl b) {
    return __Bf16Impl(__hsub(a.__x.f, b.__x.f));
  }
  __CUBF16_IMPL __Bf16Impl bf16mul(__Bf16Impl a, __Bf16Impl b) {
    return __Bf16Impl(__hmul(a.__x.f, b.__x.f));
  }
  __CUBF16_IMPL __Bf16Impl bf16div(__Bf16Impl a, __Bf16Impl b) {
    return __Bf16Impl(__hdiv(a.__x.f, b.__x.f));
  }

  // == != > < >= <=
  __CUBF16_IMPL bool bf16eq(__Bf16Impl a, __Bf16Impl b) {
    return __heq(a.__x.f, b.__x.f);
  }
  __CUBF16_IMPL bool bf16ne(__Bf16Impl a, __Bf16Impl b) {
    return __hne(a.__x.f, b.__x.f);
  }
  __CUBF16_IMPL bool bf16gt(__Bf16Impl a, __Bf16Impl b) {
    return __hgt(a.__x.f, b.__x.f);
  }
  __CUBF16_IMPL bool bf16lt(__Bf16Impl a, __Bf16Impl b) {
    return __hlt(a.__x.f, b.__x.f);
  }
  __CUBF16_IMPL bool bf16ge(__Bf16Impl a, __Bf16Impl b) {
    return __hge(a.__x.f, b.__x.f);
  }
  __CUBF16_IMPL bool bf16le(__Bf16Impl a, __Bf16Impl b) {
    return __hle(a.__x.f, b.__x.f);
  }
#else

  __CUBF16_IMPL __Bf16Impl float2bfloat16(float v) {
    uint32_t bits = reinterpret_cast<uint32_t&>(v);
    if ((bits & 0x7fffffff) > 0x7f800000) {
      return __Bf16Impl::from_bits(0x7fffU);
    } else {
      uint32_t lsb = (bits >> 16) & 1;
      uint32_t rounding_bias = 0x7fffU + lsb;
      bits += rounding_bias;
      uint16_t value = static_cast<uint16_t>(bits >> 16);
      return __Bf16Impl::from_bits(value);
    }
  }

  __CUBF16_IMPL __Bf16Impl double2bfloat16(double v) {
    return float2bfloat16(__double2float_rn((v)));
  }

  __CUBF16_IMPL __Bf16Impl ll2bfloat16(long long v) {
    return float2bfloat16(__ll2float_rn((v)));
  }

  __CUBF16_IMPL __Bf16Impl ull2bfloat16(unsigned long long v) {
    return float2bfloat16(__ull2float_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl int2bfloat16(int v) {
    return float2bfloat16(__int2float_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl uint2bfloat16(unsigned int v) {
    return float2bfloat16(__uint2float_rn(v));
  }

  __CUBF16_IMPL __Bf16Impl short2bfloat16(short v) {
    return float2bfloat16(static_cast<float>(v));
  }

  __CUBF16_IMPL __Bf16Impl ushort2bfloat16(unsigned short v) {
    return float2bfloat16(static_cast<float>(v));
  }

  __CUBF16_IMPL float bfloat162float(__Bf16Impl v) {
    uint32_t reti = static_cast<uint32_t>(v.__x.i) << 16;
    const float* retptr = reinterpret_cast<const float*>(&reti);
    return *retptr;
  }

  __CUBF16_IMPL double bfloat162double(__Bf16Impl v) {
    return static_cast<double>(bfloat162float(v));
  }

  __CUBF16_IMPL long long bfloat162ll(__Bf16Impl v) {
    return static_cast<long long>(bfloat162float(v));
  }

  __CUBF16_IMPL unsigned long long bfloat162ull(__Bf16Impl v) {
    return static_cast<unsigned long long>(bfloat162float(v));
  }

  __CUBF16_IMPL int bfloat162int(__Bf16Impl v) {
    return static_cast<int>(bfloat162float(v));
  }

  __CUBF16_IMPL unsigned int bfloat162uint(__Bf16Impl v) {
    return static_cast<unsigned int>(bfloat162float(v));
  }

  __CUBF16_IMPL short bfloat162short(__Bf16Impl v) {
    return static_cast<short>(bfloat162float(v));
  }

  __CUBF16_IMPL unsigned short bfloat162ushort(__Bf16Impl v) {
    return static_cast<unsigned short>(bfloat162float(v));
  }

  // + - * /
  __CUBF16_IMPL __Bf16Impl bf16add(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return float2bfloat16(a_f32 + b_f32);
  }
  __CUBF16_IMPL __Bf16Impl bf16sub(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return float2bfloat16(a_f32 - b_f32);
  }
  __CUBF16_IMPL __Bf16Impl bf16mul(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return float2bfloat16(a_f32 * b_f32);
  }
  __CUBF16_IMPL __Bf16Impl bf16div(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return float2bfloat16(a_f32 / b_f32);
  }

  // == != > < >= <=
  __CUBF16_IMPL bool bf16eq(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 == b_f32;
  }
  __CUBF16_IMPL bool bf16ne(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 != b_f32;
  }
  __CUBF16_IMPL bool bf16gt(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 > b_f32;
  }
  __CUBF16_IMPL bool bf16lt(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 < b_f32;
  }
  __CUBF16_IMPL bool bf16ge(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 >= b_f32;
  }
  __CUBF16_IMPL bool bf16le(__Bf16Impl a, __Bf16Impl b) {
    float a_f32 = bfloat162float(a);
    float b_f32 = bfloat162float(b);
    return a_f32 <= b_f32;
  }
#endif  // __BF16_NATIVE
};
}  // namespace __hie_buildin

namespace std {

/// Numeric limits
template <>
struct numeric_limits<__hie_buildin::__Bf16Impl> {
  using bf16_impl_t = __hie_buildin::__Bf16Impl;

 public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const is_modulo = false;
  static bool const is_bounded = true;
  static bool const is_iec559 = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static const std::float_denorm_style has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static const bool traps = false;
  static const bool tinyness_before = false;
  static std::float_round_style const round_style = std::round_to_nearest;
  static const int digits = 8;
  static const int digits10 = 2;
  static const int max_digits10 = 4;
  static const int radix = 2;
  static const int min_exponent = -125;
  static const int min_exponent10 = -37;
  static const int max_exponent = 128;
  static const int max_exponent10 = 38;

  /// Smallest positive normal value.
  __BF16_DEVICE_FUNC
  static bf16_impl_t min() { return bf16_impl_t::from_bits(0x0080); }

  /// Smallest finite value.
  __BF16_DEVICE_FUNC
  static bf16_impl_t lowest() { return bf16_impl_t::from_bits(0xff7f); }

  /// Largest finite value.
  __BF16_DEVICE_FUNC
  static bf16_impl_t max() { return bf16_impl_t::from_bits(0x7f7f); }

  /// Difference between 1 and next representable value.
  __BF16_DEVICE_FUNC
  static bf16_impl_t epsilon() { return bf16_impl_t::from_bits(0x3c00); }

  /// Maximum rounding error in ULP (units in the last place).
  __BF16_DEVICE_FUNC
  static bf16_impl_t round_error() { return bf16_impl_t::from_bits(0x3f00); }

  /// Positive infinity.
  __BF16_DEVICE_FUNC
  static bf16_impl_t infinity() { return bf16_impl_t::from_bits(0x7f80); }

  /// Quiet NaN.
  __BF16_DEVICE_FUNC
  static bf16_impl_t quiet_NaN() { return bf16_impl_t::from_bits(0x7fff); }

  /// Signaling NaN.
  __BF16_DEVICE_FUNC
  static bf16_impl_t signaling_NaN() { return bf16_impl_t::from_bits(0x7fff); }

  /// Smallest positive subnormal value.
  __BF16_DEVICE_FUNC
  static bf16_impl_t denorm_min() { return bf16_impl_t::from_bits(0x0001); }
};
}  // namespace std

#undef __CUBF16_IMPL
#undef __CUBF16_DECL
#undef __BF16_NATIVE

#endif  // defined(__CUDA_ARCH__)

#endif  // UTILS_GPU_CUDABFLOAT16_IMPL_HPP_
