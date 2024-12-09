/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudahalf_cmath_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_CMATH_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_CMATH_IMPL_HPP_

#include <cuda.h>
#include "cudahalf_impl.hpp"

namespace __hiednn_buildin {

namespace __half_cmath_impl {

#if defined(__CUDA_ARCH__)

#define __HALF_CMATH_IMPL

#define __CMATH_DEVICE_FUNC __device__ __forceinline__
#define __FP16_NATIVE(_SM_CHECK, _CUDA_VERSION_CHECK) \
    (__CUDA_ARCH__ >= _SM_CHECK) && (CUDA_VERSION >= _CUDA_VERSION_CHECK)

#define __F32_FALLBACK(func) \
    _HalfImpl::float2half(func(_HalfImpl::half2float(x)))

#define __H2F(x) _HalfImpl::half2float(x)
#define __F2H(x) _HalfImpl::float2half(x)

//------------------------------------------
// basic operators
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl fabs(_HalfImpl x) {
    _HalfImpl ret;
#if __FP16_NATIVE(530, 10020)
    asm ("abs.f16 %0, %1;\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
#else
    ret._x = x._x & 0x7fff;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl fmod(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
    asm ("{.reg.b32         d, xf, yf;\n"
         " cvt.f32.f16      xf, %1;\n"
         " cvt.f32.f16      yf, %2;\n"
         " div.approx.f32   d, xf, yf;\n"
         " cvt.rzi.f32.f32  d, d;\n"
         " neg.f32          d, d;\n"
         " fma.rn.f32       d, yf, d, xf;\n"
         " cvt.rn.f16.f32   %0, d;}\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl remainder(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
    asm ("{.reg.b32         d, xf, yf;\n"
         " cvt.f32.f16      xf, %1;\n"
         " cvt.f32.f16      yf, %2;\n"
         " div.approx.f32   d, xf, yf;\n"
         " cvt.rni.f32.f32  d, d;\n"
         " neg.f32          d, d;\n"
         " fma.rn.f32       d, yf, d, xf;\n"
         " cvt.rn.f16.f32   %0, d;}\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl fmax(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
#if __FP16_NATIVE(800, 11000)
    asm ("max.f16 %0, %1, %2;\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
#else
    ret = (y._x & 0x7fffU) > 0x7c00U || _HalfImpl::hgt(x, y) ? x : y;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl fmin(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
#if __FP16_NATIVE(800, 11000)
    asm ("min.f16 %0, %1, %2;\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
#else
    ret = (y._x & 0x7fffU) > 0x7c00U || _HalfImpl::hlt(x, y) ? x : y;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl fdim(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
#if __FP16_NATIVE(530, 7000)
    asm ("{.reg.pred    p;\n"
         " .reg.b16     zero;\n"
         " mov.b16      zero, 0;\n"
         " sub.f16      %0, %1, %2;\n"
         " setp.lt.f16  p, %0, zero;\n"
         " @p mov.b16   %0, 0;}\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
#else
    float sub = __H2F(x) - __H2F(y);
    if (sub < 0.f) { sub = 0.f; }
    ret = __F2H(sub);
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl fma(_HalfImpl x, _HalfImpl y, _HalfImpl z) {
    _HalfImpl ret;
#if __FP16_NATIVE(530, 7000)
    asm ("fma.rn.f16 %0, %1, %2, %3;\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x), "h"(z._x)
    );
#else
    ret = __F2H(fmaf(__H2F(x), __H2F(y), __H2F(z)));
#endif
    return ret;
}

//------------------------------------------
// exponential functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl exp(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         log2_e, f;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl exp2(_HalfImpl x) {
    // the max relative error of ex2.approx.f16 is 2^-9.9,
    // so convert it to ex2.approx.f32 for higher precision
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " ex2.approx.f32   f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl expm1(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         log2_e, f;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " sub.f32          f, f, 1.0;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl log(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_e, f;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " cvt.f32.f16      f, %1;\n"
         " lg2.approx.f32   f, f;\n"
         " mul.f32          f, f, rcp_log2_e;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl log10(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_10, f;\n"
         " mov.b32          rcp_log2_10, 0x3e9a209bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " lg2.approx.f32   f, f;\n"
         " mul.f32          f, f, rcp_log2_10;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl log2(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " lg2.approx.f32   f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl log1p(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_e, f;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " cvt.f32.f16      f, %1;\n"
         " add.f32          f, f, 1.0;\n"
         " lg2.approx.f32   f, f;\n"
         " mul.f32          f, f, rcp_log2_e;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

//------------------------------------------
// power functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl pow(_HalfImpl x, _HalfImpl y) {
    /**
     * pseudo code:
     *
     * yr = round_to_nearest_integer(y);
     * y_is_int = abs(y - yr) < (1 / 2048);
     * xcomp = y_is_int ? abs(x) : x;
     * z = pow(xcomp, y);
     * yi = round_to_nearest_integer(y);
     * y_is_odd = yi % 2 != 0;
     * if (y_is_int && y_is_odd) {
     *     copy sign bit of x to z
     * }
     * return z;
     */
    _HalfImpl ret;
    asm ("{.reg.b32             xf, xcomp, yf, yr, yi, zf, rcp2048;\n"
         " .reg.pred            p0, p1;\n"
         " mov.b32              rcp2048, 0x3a000000U;\n"
         " cvt.f32.f16          xf, %1;\n"
         " cvt.f32.f16          yf, %2;\n"
         " mov.b32              xcomp, xf;\n"
         " cvt.rni.f32.f32      yr, yf;\n"
         " sub.f32              yr, yf, yr;\n"
         " abs.f32              yr, yr;\n"
         " setp.lt.f32          p0, yr, rcp2048;\n"
         " @p0 abs.f32          xcomp, xf;\n"
         " lg2.approx.f32       zf, xcomp;\n"
         " mul.f32              zf, zf, yf;\n"
         " ex2.approx.f32       zf, zf;\n"
         " @p0 cvt.rni.s32.f32  yi, yf;\n"
         " @p0 and.b32          yi, yi, 0x1;\n"
         " setp.ne.and.s32      p1, yi, 0x0, p0;\n"
         " @p1 copysign.f32     zf, xf, zf;\n"
         " cvt.rn.f16.f32       %0, zf;}\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl sqrt(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " sqrt.approx.f32  f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl rsqrt(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " rsqrt.approx.f32 f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl cbrt(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_3, f, fr, s;\n"
         " .reg.pred        p;\n"
         " mov.b32          rcp_3, 0x3eaaaaabU;\n"
         " cvt.f32.f16      f, %1;\n"
         " abs.f32          fr, f;\n"
         " lg2.approx.f32   fr, fr;\n"
         " mul.f32          fr, fr, rcp_3;\n"
         " ex2.approx.f32   fr, fr;\n"
         " copysign.f32     fr, f, fr;\n"
         " cvt.rn.f16.f32   %0, fr;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl hypot(_HalfImpl x, _HalfImpl y) {
    _HalfImpl ret;
    asm ("{.reg.b32         r, xf, yf;\n"
         " cvt.f32.f16      xf, %1;\n"
         " cvt.f32.f16      yf, %2;\n"
         " mul.f32          r, xf, xf;\n"
         " fma.rn.f32       r, yf, yf, r;\n"
         " sqrt.approx.f32  r, r;\n"
         " cvt.rn.f16.f32   %0, r;}\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
    return ret;
}

//------------------------------------------
// trigonometric functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl sin(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " sin.approx.f32   f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl cos(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " cos.approx.f32   f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl tan(_HalfImpl x) {
    _HalfImpl ret;
    asm ("{.reg.b32         f, s, c;\n"
         " cvt.f32.f16      f, %1;\n"
         " sin.approx.f32   s, f;\n"
         " cos.approx.f32   c, f;\n"
         " div.approx.f32   f, s, c;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl asin(_HalfImpl x) {
    return __F32_FALLBACK(asinf);
}

__CMATH_DEVICE_FUNC
_HalfImpl acos(_HalfImpl x) {
    return __F32_FALLBACK(acosf);
}

__CMATH_DEVICE_FUNC
_HalfImpl atan(_HalfImpl x) {
    return __F32_FALLBACK(atanf);
}

__CMATH_DEVICE_FUNC
_HalfImpl atan2(_HalfImpl x, _HalfImpl y) {
    return __F2H(atan2f(__H2F(x), __H2F(y)));
}

//------------------------------------------
// hyperbolic functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl sinh(_HalfImpl x) {
    // sinh =
    // for x > 0:  (1 - exp(-2x)) / (2 * exp(-x))
    // for x <= 0: (exp(2x) - 1) / (2 * exp(x))
    _HalfImpl ret;
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, r, s, t;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " and.b32          s, f, 0x80000000U;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " mul.f32          r, f, 2.0;\n"
         " rcp.approx.f32   r, r;\n"
         " neg.f32          t, f;\n"
         " fma.rn.f32       f, f, t, 1.0;\n"
         " xor.b32          f, f, s;\n"
         " mul.f32          f, f, r;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl cosh(_HalfImpl x) {
    // cosh =
    // for x > 0:  (1 + exp(-2x)) / (2 * exp(-x))
    // for x <= 0: (1 + exp(2x)) / (2 * exp(x))
    _HalfImpl ret;
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, r;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " mul.f32          r, f, 2.0;\n"
         " rcp.approx.f32   r, r;\n"
         " mul.f32          f, f, f;\n"
         " add.f32          f, f, 1.0;\n"
         " mul.f32          f, f, r;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl tanh(_HalfImpl x) {
    _HalfImpl ret;
#if __FP16_NATIVE(750, 11000)
    // the max absolute error of tanh.approx.f16 is 2^-10.987
    // so convert it to tanh.approx.f32 for higher precision
    asm ("{.reg.b32         f;\n"
         " cvt.f32.f16      f, %1;\n"
         " tanh.approx.f32  f, f;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x));
#else
    // tanh =
    // for x > 0: (1 - exp(-2x)) / (1 + exp(-2x))
    // for x <= 0: (exp(2x) - 1) / (1 + exp(2x))
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, t, r, s;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " cvt.f32.f16      f, %1;\n"
         " and.b32          s, f, 0x80000000U;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " fma.rn.f32       r, f, f, 1.0;\n"
         " rcp.approx.f32   r, r;\n"
         " neg.f32          t, f;\n"
         " fma.rn.f32       t, f, t, 1.0;\n"
         " mul.f32          f, t, r;\n"
         " xor.b32          f, f, s;\n"
         " cvt.rn.f16.f32   %0, f;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl asinh(_HalfImpl x) {
    // asinh = ln(x + sqrt(x^2 + 1))
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_e, r, f;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " cvt.f32.f16      f, %1;\n"
         " fma.rn.f32       r, f, f, 1.0;\n"
         " sqrt.approx.f32  r, r;\n"
         " add.f32          r, r, f;\n"
         " lg2.approx.f32   r, r;\n"
         " mul.f32          r, r, rcp_log2_e;\n"
         " cvt.rn.f16.f32   %0, r;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl acosh(_HalfImpl x) {
    // acosh = ln(x + sqrt(x^2 - 1))
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_e, r, f;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " cvt.f32.f16      f, %1;\n"
         " fma.rn.f32       r, f, f, -1.0;\n"
         " sqrt.approx.f32  r, r;\n"
         " add.f32          r, r, f;\n"
         " lg2.approx.f32   r, r;\n"
         " mul.f32          r, r, rcp_log2_e;\n"
         " cvt.rn.f16.f32   %0, r;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl atanh(_HalfImpl x) {
    // atanh = ln((1 + x) / (1 - x)) / 2
    //       = (ln(1 + x) - ln(1 - x)) / 2
    _HalfImpl ret;
    asm ("{.reg.b32         rcp_log2_e, r0, r1, f;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " cvt.f32.f16      f, %1;\n"
         " add.f32          r0, 1.0, f;\n"
         " sub.f32          r1, 1.0, f;\n"
         " lg2.approx.f32   r1, r1;\n"
         " mul.f32          r1, r1, rcp_log2_e;\n"
         " neg.f32          r1, r1;\n"
         " lg2.approx.f32   r0, r0;\n"
         " fma.rn.f32       r0, r0, rcp_log2_e, r1;\n"
         " mul.f32          r0, r0, 0.5;\n"
         " cvt.rn.f16.f32   %0, r0;}\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
    return ret;
}

//------------------------------------------
// error and gamma functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl erf(_HalfImpl x) {
    return __F32_FALLBACK(erff);
}

__CMATH_DEVICE_FUNC
_HalfImpl erfc(_HalfImpl x) {
    return __F32_FALLBACK(erfcf);
}

__CMATH_DEVICE_FUNC
_HalfImpl tgamma(_HalfImpl x) {
    return __F32_FALLBACK(tgammaf);
}

__CMATH_DEVICE_FUNC
_HalfImpl lgamma(_HalfImpl x) {
    return __F32_FALLBACK(lgammaf);
}

//------------------------------------------
// nearest integer floating point operations
//------------------------------------------
__CMATH_DEVICE_FUNC
_HalfImpl ceil(_HalfImpl x) {
    _HalfImpl ret;
    asm ("cvt.rpi.f16.f16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl floor(_HalfImpl x) {
    _HalfImpl ret;
    asm ("cvt.rmi.f16.f16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl trunc(_HalfImpl x) {
    _HalfImpl ret;
    asm ("cvt.rzi.f16.f16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl nearbyint(_HalfImpl x) {
    _HalfImpl ret;
    asm ("cvt.rni.f16.f16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_HalfImpl round(_HalfImpl x) {
    return __F32_FALLBACK(roundf);
}

__CMATH_DEVICE_FUNC
_HalfImpl rint(_HalfImpl x) {
    _HalfImpl ret;
    asm ("cvt.rni.f16.f16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

//------------------------------------------
// classification
//------------------------------------------
__CMATH_DEVICE_FUNC
bool isinf(_HalfImpl x) {
    return (x._x & 0x7fffU) == 0x7c00U;
}

__CMATH_DEVICE_FUNC
bool isnan(_HalfImpl x) {
    return (x._x & 0x7fffU) > 0x7c00U;
}

__CMATH_DEVICE_FUNC
bool isfinite(_HalfImpl x) {
    return (x._x & 0x7fffU) < 0x7c00U;
}

__CMATH_DEVICE_FUNC
bool isnormal(_HalfImpl x) {
    uint16_t expo = x._x & 0x7c00U;
    return expo < 0x7c00U && expo != 0;
}

__CMATH_DEVICE_FUNC
bool signbit(_HalfImpl x) {
    return (x._x & 0x8000) != 0;
}

#undef __FP16_NATIVE
#undef __F32_FALLBACK
#undef __H2F
#undef __F2H

#endif  // __CUDA_ARCH__

}  // namespace __half_cmath_impl

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_CMATH_IMPL_HPP_


