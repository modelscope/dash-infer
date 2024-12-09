/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudabfloat16_cmath_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_CMATH_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_CMATH_IMPL_HPP_

#include <cuda.h>
#include "cudabfloat16_impl.hpp"

namespace __hiednn_buildin {

namespace __bf16_cmath_impl {

#if defined(__CUDA_ARCH__)

#define __BF16_CMATH_IMPL

#define __CMATH_DEVICE_FUNC __device__ __forceinline__
#define __BF16_NATIVE(_SM_CHECK, _CUDA_VERSION_CHECK) \
    (__CUDA_ARCH__ >= _SM_CHECK) && (CUDA_VERSION >= _CUDA_VERSION_CHECK)

#define __F32_FALLBACK(func) \
    _Bf16Impl::float2bfloat16(func(_Bf16Impl::bfloat162float(x)))

#define __H2F(x) _Bf16Impl::bfloat162float(x)
#define __F2H(x) _Bf16Impl::float2bfloat16(x)

//------------------------------------------
// basic operators
//------------------------------------------
__CMATH_DEVICE_FUNC
_Bf16Impl fabs(_Bf16Impl x) {
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("abs.bf16 %0, %1;\n"
         : "=h"(ret._x)
         : "h"(x._x)
    );
#else
    ret._x = x._x & 0x7fffU;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl fmod(_Bf16Impl x, _Bf16Impl y) {
    float xf = __H2F(x);
    float yf = __H2F(y);
    float ret;
    asm ("{.reg.b32         d;\n"
         " div.approx.f32   d, %1, %2;\n"
         " cvt.rzi.f32.f32  d, d;\n"
         " neg.f32          d, d;\n"
         " fma.rn.f32       %0, %2, d, %1;}\n"
         : "=f"(ret)
         : "f"(xf), "f"(yf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl remainder(_Bf16Impl x, _Bf16Impl y) {
    float xf = __H2F(x);
    float yf = __H2F(y);
    float ret;
    asm ("{.reg.b32         d;\n"
         " div.approx.f32   d, %1, %2;\n"
         " cvt.rni.f32.f32  d, d;\n"
         " neg.f32          d, d;\n"
         " fma.rn.f32       %0, %2, d, %1;}\n"
         : "=f"(ret)
         : "f"(xf), "f"(yf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl fmax(_Bf16Impl x, _Bf16Impl y) {
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("max.bf16 %0, %1, %2;\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
#else
    ret = (y._x & 0x7fffU) > 0x7f80U || _Bf16Impl::bf16gt(x, y) ? x : y;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl fmin(_Bf16Impl x, _Bf16Impl y) {
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("min.bf16 %0, %1, %2;\n"
         : "=h"(ret._x)
         : "h"(x._x), "h"(y._x)
    );
#else
    ret = (y._x & 0x7fffU) > 0x7f80U || _Bf16Impl::bf16lt(x, y) ? x : y;
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl fdim(_Bf16Impl x, _Bf16Impl y) {
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("{.reg.b16     neg_1;\n"
         " .reg.b32     subf;\n"
         " .reg.pred    p;\n"
         " mov.b16      neg_1, 0xbf80U;\n"
         " fma.rn.bf16  %0, %2, neg_1, %1;\n"
         " mov.b32      subf, {0, %0};\n"
         " setp.lt.f32  p, subf, 0.0;\n"
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
_Bf16Impl fma(_Bf16Impl x, _Bf16Impl y, _Bf16Impl z) {
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
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
_Bf16Impl exp(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_e, r;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " mul.f32          r, log2_e, %1;\n"
         " ex2.approx.f32   %0, r;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl exp2(_Bf16Impl x) {
    // the max relative error of ex2.approx.bf16 is 2^-7,
    // so convert it to ex2.approx.f32 for higher precision
    float xf = __H2F(x);
    float retf;
    asm ("ex2.approx.f32 %0, %1;\n"
         : "=f"(retf)
         : "f"(xf)
    );
    return __F2H(retf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl expm1(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_e, r;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " mul.f32          r, log2_e, %1;\n"
         " ex2.approx.f32   %0, r;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret - 1.f);
}

__CMATH_DEVICE_FUNC
_Bf16Impl log(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_x, rcp_log2_e;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " lg2.approx.f32   log2_x, %1;\n"
         " mul.f32          %0, log2_x, rcp_log2_e;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl log10(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_x, rcp_log2_10;\n"
         " mov.b32          rcp_log2_10, 0x3e9a209bU;\n"
         " lg2.approx.f32   log2_x, %1;\n"
         " mul.f32          %0, log2_x, rcp_log2_10;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl log2(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("lg2.approx.f32 %0, %1;\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl log1p(_Bf16Impl x) {
    float xf = __H2F(x) + 1.f;
    float ret;
    asm ("{.reg.b32         log2_x, rcp_log2_e;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " lg2.approx.f32   log2_x, %1;\n"
         " mul.f32          %0, log2_x, rcp_log2_e;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

//------------------------------------------
// power functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_Bf16Impl pow(_Bf16Impl x, _Bf16Impl y) {
    /**
     * pseudo code:
     *
     * yr = round_to_nearest_integer(yf);
     * y_is_int = abs(yf - yr) < (1 / 256);
     * x = y_is_int ? abs(xf) : xf;
     * z = pow(x, yf);
     * yi = round_to_nearest_integer(yf);
     * y_is_odd = yi % 2 != 0;
     * if (y_is_int && y_is_odd) {
     *     copy sign bit of x to z
     * }
     * return z;
     */
    float xf = __H2F(x);
    float yf = __H2F(y);
    float ret;
    asm ("{.reg.b32             x, yr, yi, z, rcp256;\n"
         " .reg.pred            p0, p1;\n"
         " mov.b32              rcp256, 0x3b800000U;\n"
         " mov.b32              x, %1;\n"
         " cvt.rni.f32.f32      yr, %2;\n"
         " sub.f32              yr, %2, yr;\n"
         " abs.f32              yr, yr;\n"
         " setp.lt.f32          p0, yr, rcp256;\n"
         " @p0 abs.f32          x, %1;\n"
         " lg2.approx.f32       z, x;\n"
         " mul.f32              z, z, %2;\n"
         " ex2.approx.f32       z, z;\n"
         " @p0 cvt.rni.s32.f32  yi, %2;\n"
         " @p0 and.b32          yi, yi, 0x1;\n"
         " setp.ne.and.s32      p1, yi, 0x0, p0;\n"
         " @p1 copysign.f32     z, %1, z;\n"
         " mov.b32              %0, z;}\n"
         : "=f"(ret)
         : "f"(xf), "f"(yf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl sqrt(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("sqrt.approx.f32 %0, %1;\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl rsqrt(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("rsqrt.approx.f32 %0, %1;\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl cbrt(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         rcp_3, r, s;\n"
         " .reg.pred        p;\n"
         " mov.b32          rcp_3, 0x3eaaaaabU;\n"
         " abs.f32          r, %1;\n"
         " lg2.approx.f32   r, r;\n"
         " mul.f32          r, r, rcp_3;\n"
         " ex2.approx.f32   r, r;\n"
         " copysign.f32     r, %1, r;\n"
         " mov.b32          %0, r;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl hypot(_Bf16Impl x, _Bf16Impl y) {
    float xf = __H2F(x);
    float yf = __H2F(y);
    float ret;
    asm ("{.reg.b32         r;\n"
         " mul.f32          r, %1, %1;\n"
         " fma.rn.f32       r, %2, %2, r;\n"
         " sqrt.approx.f32  %0, r;}\n"
         : "=f"(ret)
         : "f"(xf), "f"(yf)
    );
    return __F2H(ret);
}

//------------------------------------------
// trigonometric functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_Bf16Impl sin(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("sin.approx.f32 %0, %1;\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl cos(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("cos.approx.f32 %0, %1;\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl tan(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32 s, c;\n"
         " sin.approx.f32 s, %1;\n"
         " cos.approx.f32 c, %1;\n"
         " div.approx.f32 %0, s, c;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl asin(_Bf16Impl x) {
    return __F32_FALLBACK(asinf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl acos(_Bf16Impl x) {
    return __F32_FALLBACK(acosf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl atan(_Bf16Impl x) {
    return __F32_FALLBACK(atanf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl atan2(_Bf16Impl x, _Bf16Impl y) {
    return __F2H(atan2f(__H2F(x), __H2F(y)));
}

//------------------------------------------
// hyperbolic functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_Bf16Impl sinh(_Bf16Impl x) {
    // sinh =
    // for x > 0:  (1 - exp(-2x)) / (2 * exp(-x))
    // for x <= 0: (exp(2x) - 1) / (2 * exp(x))
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, r, s, t;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " mov.b32          f, %1;\n"
         " and.b32          s, f, 0x80000000U;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " mul.f32          r, f, 2.0;\n"
         " rcp.approx.f32   r, r;\n"
         " neg.f32          t, f;\n"
         " fma.rn.f32       f, f, t, 1.0;\n"
         " xor.b32          f, f, s;\n"
         " mul.f32          %0, f, r;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl cosh(_Bf16Impl x) {
    // cosh =
    // for x > 0:  (1 + exp(-2x)) / (2 * exp(-x))
    // for x <= 0: (exp(2x) + 1) / (2 * exp(x))
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, r;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " mov.b32          f, %1;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " mul.f32          r, f, 2.0;\n"
         " rcp.approx.f32   r, r;\n"
         " mul.f32          f, f, f;\n"
         " add.f32          f, f, 1.0;\n"
         " mul.f32          %0, f, r;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl tanh(_Bf16Impl x) {
    float xf = __H2F(x);
    float ret;
#if __BF16_NATIVE(750, 11000)
    // the max absolute error of tanh.approx.bf16 is 2^-8,
    // so convert it to tanh.approx.f32 for higher precision
    asm ("tanh.approx.f32 %0, %1;" : "=f"(ret) : "f"(xf));
#else
    // tanh =
    // for x > 0: (1 - exp(-2x)) / (1 + exp(-2x))
    // for x <= 0: (exp(2x) - 1) / (exp(2x) + 1)
    asm ("{.reg.b32         log2_e;\n"
         " .reg.b32         f, t, r, s;\n"
         " mov.b32          log2_e, 0x3fb8aa3bU;\n"
         " mov.b32          f, %1;\n"
         " and.b32          s, f, 0x80000000U;\n"
         " or.b32           f, f, 0x80000000U;\n"
         " mul.f32          f, f, log2_e;\n"
         " ex2.approx.f32   f, f;\n"
         " fma.rn.f32       r, f, f, 1.0;\n"
         " rcp.approx.f32   r, r;\n"
         " neg.f32          t, f;\n"
         " fma.rn.f32       t, f, t, 1.0;\n"
         " mul.f32          f, t, r;\n"
         " xor.b32          %0, f, s;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
#endif
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl asinh(_Bf16Impl x) {
    // asinh = ln(x + sqrt(x^2 + 1))
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         rcp_log2_e, r;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " fma.rn.f32       r, %1, %1, 1.0;\n"
         " sqrt.approx.f32  r, r;\n"
         " add.f32          r, r, %1;\n"
         " lg2.approx.f32   r, r;\n"
         " mul.f32          %0, r, rcp_log2_e;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl acosh(_Bf16Impl x) {
    // acosh = ln(x + sqrt(x^2 - 1))
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         rcp_log2_e, r;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " fma.rn.f32       r, %1, %1, -1.0;\n"
         " sqrt.approx.f32  r, r;\n"
         " add.f32          r, r, %1;\n"
         " lg2.approx.f32   r, r;\n"
         " mul.f32          %0, r, rcp_log2_e;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

__CMATH_DEVICE_FUNC
_Bf16Impl atanh(_Bf16Impl x) {
    // atanh = ln((1 + x) / (1 - x)) / 2
    //       = (ln(1 + x) - ln(1 - x)) / 2
    float xf = __H2F(x);
    float ret;
    asm ("{.reg.b32         rcp_log2_e, r0, r1;\n"
         " mov.b32          rcp_log2_e, 0x3f317218U;\n"
         " add.f32          r0, 1.0, %1;\n"
         " sub.f32          r1, 1.0, %1;\n"
         " lg2.approx.f32   r1, r1;\n"
         " mul.f32          r1, r1, rcp_log2_e;\n"
         " neg.f32          r1, r1;\n"
         " lg2.approx.f32   r0, r0;\n"
         " fma.rn.f32       r0, r0, rcp_log2_e, r1;\n"
         " mul.f32          %0, r0, 0.5;}\n"
         : "=f"(ret)
         : "f"(xf)
    );
    return __F2H(ret);
}

//------------------------------------------
// error and gamma functions
//------------------------------------------
__CMATH_DEVICE_FUNC
_Bf16Impl erf(_Bf16Impl x) {
    return __F32_FALLBACK(erff);
}

__CMATH_DEVICE_FUNC
_Bf16Impl erfc(_Bf16Impl x) {
    return __F32_FALLBACK(erfcf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl tgamma(_Bf16Impl x) {
    return __F32_FALLBACK(tgammaf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl lgamma(_Bf16Impl x) {
    return __F32_FALLBACK(lgammaf);
}

//------------------------------------------
// nearest integer floating point operations
//------------------------------------------
#if __BF16_NATIVE(900, 11080)

__CMATH_DEVICE_FUNC
_Bf16Impl ceil(_Bf16Impl x) {
    _Bf16Impl ret;
    asm ("cvt.rpi.bf16.bf16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl floor(_Bf16Impl x) {
    _Bf16Impl ret;
    asm ("cvt.rmi.bf16.bf16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl trunc(_Bf16Impl x) {
    _Bf16Impl ret;
    asm ("cvt.rzi.bf16.bf16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl nearbyint(_Bf16Impl x) {
    _Bf16Impl ret;
    asm ("cvt.rni.bf16.bf16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl round(_Bf16Impl x) {
    return __F32_FALLBACK(roundf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl rint(_Bf16Impl x) {
    _Bf16Impl ret;
    asm ("cvt.rni.bf16.bf16 %0, %1;\n" : "=h"(ret._x) : "h"(x._x));
    return ret;
}

#else

__CMATH_DEVICE_FUNC
_Bf16Impl ceil(_Bf16Impl x) {
    float retf = ceilf(__H2F(x));
    _Bf16Impl ret;

    const uint32_t &reti = reinterpret_cast<const uint32_t &>(retf);
    if ((reti & 0x7fffffffU) > 0x7f800000U) {
        ret._x = 0x7fffU;
    } else {
        ret._x = reti >> 16;
        // round up
        if ((reti & 0xffffU) != 0 && (reti & 0x80000000U) == 0) {
            ++ret._x;
        }
    }
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl floor(_Bf16Impl x) {
    float retf = floorf(__H2F(x));
    _Bf16Impl ret;

    const uint32_t &reti = reinterpret_cast<const uint32_t &>(retf);
    if ((reti & 0x7fffffffU) > 0x7f800000U) {
        ret._x = 0x7fffU;
    } else {
        ret._x = reti >> 16;
        // round down
        if ((reti & 0xffffU) != 0 && (reti & 0x80000000U) != 0) {
            ++ret._x;
        }
    }
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl trunc(_Bf16Impl x) {
    float retf = truncf(__H2F(x));
    _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
    asm ("cvt.rz.bf16.f32 %0, %1;\n"
         : "=h"(ret._x)
         : "f"(retf)
    );
#else
    const uint32_t &reti = reinterpret_cast<const uint32_t &>(retf);
    ret._x = (reti & 0x7fffffffU) > 0x7f800000U ? 0x7fffU : (reti >> 16);
#endif
    return ret;
}

__CMATH_DEVICE_FUNC
_Bf16Impl nearbyint(_Bf16Impl x) {
    return __F32_FALLBACK(nearbyintf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl round(_Bf16Impl x) {
    return __F32_FALLBACK(roundf);
}

__CMATH_DEVICE_FUNC
_Bf16Impl rint(_Bf16Impl x) {
    return __F32_FALLBACK(rintf);
}

#endif

//------------------------------------------
// classification
//------------------------------------------
__CMATH_DEVICE_FUNC
bool isinf(_Bf16Impl x) {
    return (x._x & 0x7fffU) == 0x7f80U;
}

__CMATH_DEVICE_FUNC
bool isnan(_Bf16Impl x) {
    return (x._x & 0x7fffU) > 0x7f80U;
}

__CMATH_DEVICE_FUNC
bool isfinite(_Bf16Impl x) {
    return (x._x & 0x7fffU) < 0x7f80U;
}

__CMATH_DEVICE_FUNC
bool isnormal(_Bf16Impl x) {
    uint16_t expo = x._x & 0x7f80U;
    return expo < 0x7f80U && expo != 0;
}

__CMATH_DEVICE_FUNC
bool signbit(_Bf16Impl x) {
    return (x._x & 0x8000) != 0;
}

#undef __BF16_NATIVE
#undef __F32_FALLBACK
#undef __H2F
#undef __F2H

#endif  // __CUDA_ARCH__

}  // namespace __bf16_cmath_impl

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_CMATH_IMPL_HPP_


