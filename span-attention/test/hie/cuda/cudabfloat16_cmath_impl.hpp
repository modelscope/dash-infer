/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudabfloat16_cmath_impl.hpp
 */

#ifndef UTILS_GPU_CUDABFLOAT16_CMATH_IMPL_HPP_
#define UTILS_GPU_CUDABFLOAT16_CMATH_IMPL_HPP_

#include <cuda.h>

#include "cudabfloat16_impl.hpp"

namespace __hie_buildin {

namespace __bf16_cmath_impl {

#if defined(__CUDA_ARCH__)

#define __BF16_CMATH_IMPL

#define __CMATH_DEVICE_FUNC __device__ __forceinline__
#define __BF16_NATIVE (__CUDA_ARCH__ >= 800) && (CUDA_VERSION >= 11000)

#define __F32_FALLBACK(func) \
  __Bf16Impl::float2bfloat16(func(__Bf16Impl::bfloat162float(x)))

#define __H2F(x) __Bf16Impl::bfloat162float(x)
#define __F2H(x) __Bf16Impl::float2bfloat16(x)

//------------------------------------------
// basic operators
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl fabs(__Bf16Impl x) {
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("abs.bf16 %0, %1;\n" : "=h"(ret.__x.i) : "h"(x.__x.i));
#else
  ret.__x.i = x.__x.i & 0x7fffU;
#endif
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl fmod(__Bf16Impl x, __Bf16Impl y) {
  float xf = __H2F(x);
  float yf = __H2F(y);
  float ret;
  asm("{.reg.b32         d;\n"
      " div.approx.f32   d, %1, %2;\n"
      " cvt.rzi.f32.f32  d, d;\n"
      " neg.f32          d, d;\n"
      " fma.rn.f32       %0, %2, d, %1;}\n"
      : "=f"(ret)
      : "f"(xf), "f"(yf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl remainder(__Bf16Impl x, __Bf16Impl y) {
  float xf = __H2F(x);
  float yf = __H2F(y);
  float ret;
  asm("{.reg.b32         d;\n"
      " div.approx.f32   d, %1, %2;\n"
      " cvt.rni.f32.f32  d, d;\n"
      " neg.f32          d, d;\n"
      " fma.rn.f32       %0, %2, d, %1;}\n"
      : "=f"(ret)
      : "f"(xf), "f"(yf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl fmax(__Bf16Impl x, __Bf16Impl y) {
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("max.bf16 %0, %1, %2;\n" : "=h"(ret.__x.i) : "h"(x.__x.i), "h"(y.__x.i));
#else
  ret = (y.__x.i & 0x7fffU) > 0x7f80U || __Bf16Impl::bf16gt(x, y) ? x : y;
#endif
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl fmin(__Bf16Impl x, __Bf16Impl y) {
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("min.bf16 %0, %1, %2;\n" : "=h"(ret.__x.i) : "h"(x.__x.i), "h"(y.__x.i));
#else
  ret = (y.__x.i & 0x7fffU) > 0x7f80U || __Bf16Impl::bf16lt(x, y) ? x : y;
#endif
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl fdim(__Bf16Impl x, __Bf16Impl y) {
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("{.reg.b16     neg;\n"
      " .reg.b32     subf;\n"
      " .reg.pred    p;\n"
      " mov.b16      neg, 0xbf80U;\n"
      " fma.rn.bf16  %0, %2, neg, %1;\n"
      " mov.b32      subf, {0, %0};\n"
      " setp.lt.f32  p, subf, 0.0;\n"
      " @p mov.b16   %0, 0;}\n"
      : "=h"(ret.__x.i)
      : "h"(x.__x.i), "h"(y.__x.i));
#else
  float sub = __H2F(x) - __H2F(y);
  if (sub < 0.f) {
    sub = 0.f;
  }
  ret = __F2H(sub);
#endif
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl fma(__Bf16Impl x, __Bf16Impl y, __Bf16Impl z) {
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("fma.rn.bf16 %0, %1, %2, %3;\n"
      : "=h"(ret.__x.i)
      : "h"(x.__x.i), "h"(y.__x.i), "h"(z.__x.i));
#else
  ret = __F2H(fmaf(__H2F(x), __H2F(y), __H2F(z)));
#endif
  return ret;
}

//------------------------------------------
// exponential functions
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl exp(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         expo;\n"
      " mov.b32          expo, 0x3fb8aa3bU;\n"
      " mul.f32          expo, expo, %1;\n"
      " ex2.approx.f32   %0, expo;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl exp2(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl expm1(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         expo;\n"
      " mov.b32          expo, 0x3fb8aa3bU;\n"
      " mul.f32          expo, expo, %1;\n"
      " ex2.approx.f32   %0, expo;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret - 1.f);
}

__CMATH_DEVICE_FUNC
__Bf16Impl log(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         logx, logercp;\n"
      " mov.b32          logercp, 0x3f317218U;\n"
      " lg2.approx.f32   logx, %1;\n"
      " mul.f32          %0, logx, logercp;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl log10(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         logx, log10rcp;\n"
      " mov.b32          log10rcp, 0x3e9a209bU;\n"
      " lg2.approx.f32   logx, %1;\n"
      " mul.f32          %0, logx, log10rcp;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl log2(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("lg2.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl log1p(__Bf16Impl x) {
  float xf = __H2F(x) + 1.f;
  float ret;
  asm("{.reg.b32         logx, logercp;\n"
      " mov.b32          logercp, 0x3f317218U;\n"
      " lg2.approx.f32   logx, %1;\n"
      " mul.f32          %0, logx, logercp;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

//------------------------------------------
// power functions
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl pow(__Bf16Impl x, __Bf16Impl y) {
  float xf = __H2F(x);
  float yf = __H2F(y);
  float ret;
  asm("{.reg.b32         expo;\n"
      " lg2.approx.f32   expo, %1;\n"
      " mul.f32          expo, expo, %2;\n"
      " ex2.approx.f32   %0, expo;}\n"
      : "=f"(ret)
      : "f"(xf), "f"(yf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl sqrt(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("sqrt.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl rsqrt(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("rsqrt.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl cbrt(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         rcp3, r;\n"
      " .reg.pred        p;\n"
      " mov.b32          rcp3, 0x3eaaaaabU;\n"
      " setp.lt.f32      p, %1, 0.0;\n"
      " abs.f32          r, %1;\n"
      " lg2.approx.f32   r, r;\n"
      " mul.f32          r, r, rcp3;\n"
      " ex2.approx.f32   r, r;\n"
      " @p neg.f32       r, r;\n"
      " mov.b32          %0, r;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl hypot(__Bf16Impl x, __Bf16Impl y) {
  float xf = __H2F(x);
  float yf = __H2F(y);
  float ret;
  asm("{.reg.b32         r;\n"
      " mul.f32          r, %1, %1;\n"
      " fma.rn.f32       r, %2, %2, r;\n"
      " sqrt.approx.f32  %0, r;}\n"
      : "=f"(ret)
      : "f"(xf), "f"(yf));
  return __F2H(ret);
}

//------------------------------------------
// trigonometric functions
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl sin(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("sin.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl cos(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("cos.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl tan(__Bf16Impl x) {
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32 s, c;\n"
      " sin.approx.f32 s, %1;\n"
      " cos.approx.f32 c, %1;\n"
      " div.approx.f32 %0, s, c;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl asin(__Bf16Impl x) { return __F32_FALLBACK(asinf); }

__CMATH_DEVICE_FUNC
__Bf16Impl acos(__Bf16Impl x) { return __F32_FALLBACK(acosf); }

__CMATH_DEVICE_FUNC
__Bf16Impl atan(__Bf16Impl x) { return __F32_FALLBACK(atanf); }

__CMATH_DEVICE_FUNC
__Bf16Impl atan2(__Bf16Impl x, __Bf16Impl y) {
  return __F2H(atan2f(__H2F(x), __H2F(y)));
}

//------------------------------------------
// hyperbolic functions
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl sinh(__Bf16Impl x) {
  // sinh = (exp(x) - exp(-x)) / 2
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         loge;\n"
      " .reg.b32         r0, r1;\n"
      " mov.b32          loge, 0x3fb8aa3bU;\n"
      " mul.f32          r0, %1, loge;\n"
      " ex2.approx.f32   r0, r0;\n"
      " neg.f32          r1, %1;\n"
      " mul.f32          r1, r1, loge;\n"
      " ex2.approx.f32   r1, r1;\n"
      " sub.f32          r0, r0, r1;\n"
      " mul.f32          %0, r0, 0.5;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl cosh(__Bf16Impl x) {
  // cosh = (exp(x) + exp(-x)) / 2
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         loge;\n"
      " .reg.b32         r0, r1;\n"
      " mov.b32          loge, 0x3fb8aa3bU;\n"
      " mul.f32          r0, %1, loge;\n"
      " ex2.approx.f32   r0, r0;\n"
      " neg.f32          r1, %1;\n"
      " mul.f32          r1, r1, loge;\n"
      " ex2.approx.f32   r1, r1;\n"
      " add.f32          r0, r0, r1;\n"
      " mul.f32          %0, r0, 0.5;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl tanh(__Bf16Impl x) {
  // tanh = (exp(2x) - 1) / (exp(2x) + 1)
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         loge, r0, r1;\n"
      " mov.b32          loge, 0x3fb8aa3bU;\n"
      " mul.f32          r0, %1, 2.0;\n"
      " mul.f32          r0, r0, loge;\n"
      " ex2.approx.f32   r0, r0;\n"
      " add.f32          r1, r0, 1.0;\n"
      " sub.f32          r0, r0, 1.0;\n"
      " div.approx.f32   %0, r0, r1;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl asinh(__Bf16Impl x) {
  // asinh = ln(x + sqrt(x^2 + 1))
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         logercp, r;\n"
      " mov.b32          logercp, 0x3f317218U;\n"
      " fma.rn.f32       r, %1, %1, 1.0;\n"
      " sqrt.approx.f32  r, r;\n"
      " add.f32          r, r, %1;\n"
      " lg2.approx.f32   r, r;\n"
      " mul.f32          %0, r, logercp;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl acosh(__Bf16Impl x) {
  // acosh = ln(x + sqrt(x^2 - 1))
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         logercp, r;\n"
      " mov.b32          logercp, 0x3f317218U;\n"
      " fma.rn.f32       r, %1, %1, -1.0;\n"
      " sqrt.approx.f32  r, r;\n"
      " add.f32          r, r, %1;\n"
      " lg2.approx.f32   r, r;\n"
      " mul.f32          %0, r, logercp;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

__CMATH_DEVICE_FUNC
__Bf16Impl atanh(__Bf16Impl x) {
  // atanh = ln((1 + x) / (1 - x)) / 2
  //       = (ln(1 + x) - ln(1 - x)) / 2
  float xf = __H2F(x);
  float ret;
  asm("{.reg.b32         logercp, r0, r1;\n"
      " mov.b32          logercp, 0x3f317218U;\n"
      " add.f32          r0, 1.0, %1;\n"
      " sub.f32          r1, 1.0, %1;\n"
      " lg2.approx.f32   r1, r1;\n"
      " mul.f32          r1, r1, logercp;\n"
      " neg.f32          r1, r1;\n"
      " lg2.approx.f32   r0, r0;\n"
      " fma.rn.f32       r0, r0, logercp, r1;\n"
      " mul.f32          %0, r0, 0.5;}\n"
      : "=f"(ret)
      : "f"(xf));
  return __F2H(ret);
}

//------------------------------------------
// error and gamma functions
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl erf(__Bf16Impl x) { return __F32_FALLBACK(erff); }

__CMATH_DEVICE_FUNC
__Bf16Impl erfc(__Bf16Impl x) { return __F32_FALLBACK(erfcf); }

__CMATH_DEVICE_FUNC
__Bf16Impl tgamma(__Bf16Impl x) { return __F32_FALLBACK(tgammaf); }

__CMATH_DEVICE_FUNC
__Bf16Impl lgamma(__Bf16Impl x) { return __F32_FALLBACK(lgammaf); }

//------------------------------------------
// nearest integer floating point operations
//------------------------------------------
__CMATH_DEVICE_FUNC
__Bf16Impl ceil(__Bf16Impl x) {
  float retf = ceilf(__H2F(x));
  __Bf16Impl ret;

  const uint32_t& reti = reinterpret_cast<const uint32_t&>(retf);
  if ((reti & 0x7fffffffU) > 0x7f800000U) {
    ret.__x.i = 0x7fffU;
  } else {
    ret.__x.i = reti >> 16;
    // round up
    if ((reti & 0xffffU) != 0 && (reti & 0x80000000U) == 0) {
      ++ret.__x.i;
    }
  }
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl floor(__Bf16Impl x) {
  float retf = floorf(__H2F(x));
  __Bf16Impl ret;

  const uint32_t& reti = reinterpret_cast<const uint32_t&>(retf);
  if ((reti & 0x7fffffffU) > 0x7f800000U) {
    ret.__x.i = 0x7fffU;
  } else {
    ret.__x.i = reti >> 16;
    // round down
    if ((reti & 0xffffU) != 0 && (reti & 0x80000000U) != 0) {
      ++ret.__x.i;
    }
  }
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl trunc(__Bf16Impl x) {
  float retf = truncf(__H2F(x));
  __Bf16Impl ret;
#if __BF16_NATIVE
  asm("cvt.rz.bf16.f32 %0, %1;\n" : "=h"(ret.__x.i) : "f"(retf));
#else
  const uint32_t& reti = reinterpret_cast<const uint32_t&>(retf);
  ret.__x.i = (reti & 0x7fffffffU) > 0x7f800000U ? 0x7fffU : (reti >> 16);
#endif
  return ret;
}

__CMATH_DEVICE_FUNC
__Bf16Impl nearbyint(__Bf16Impl x) { return __F32_FALLBACK(nearbyintf); }

__CMATH_DEVICE_FUNC
__Bf16Impl round(__Bf16Impl x) { return __F32_FALLBACK(roundf); }

__CMATH_DEVICE_FUNC
__Bf16Impl rint(__Bf16Impl x) { return __F32_FALLBACK(rintf); }

//------------------------------------------
// classification
//------------------------------------------
__CMATH_DEVICE_FUNC
bool isinf(__Bf16Impl x) { return (x.__x.i & 0x7fffU) == 0x7f80U; }

__CMATH_DEVICE_FUNC
bool isnan(__Bf16Impl x) { return (x.__x.i & 0x7fffU) > 0x7f80U; }

__CMATH_DEVICE_FUNC
bool isfinite(__Bf16Impl x) { return (x.__x.i & 0x7fffU) < 0x7f80U; }

__CMATH_DEVICE_FUNC
bool isnormal(__Bf16Impl x) {
  uint16_t expo = x.__x.i & 0x7f80U;
  return expo < 0x7f80U && expo != 0;
}

__CMATH_DEVICE_FUNC
bool signbit(__Bf16Impl x) { return (x.__x.i & 0x8000) != 0; }

#undef __BF16_NATIVE
#undef __F32_FALLBACK
#undef __H2F
#undef __F2H

#endif  // __CUDA_ARCH__

}  // namespace __bf16_cmath_impl

}  // namespace __hie_buildin

#endif  // UTILS_GPU_CUDABFLOAT16_CMATH_IMPL_HPP_
