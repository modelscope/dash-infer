/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cmath_wrapper.hpp
 */

#ifndef DNN_INCLUDE_CMATH_WRAPPER_HPP_
#define DNN_INCLUDE_CMATH_WRAPPER_HPP_

#include <cmath>
#ifdef HIEDNN_USE_FP16
#include <datatype_extension/half/hiednn_half.hpp>
#include <datatype_extension/half/hiednn_half_cmath.hpp>
#endif
#ifdef HIEDNN_USE_BF16
#include <datatype_extension/bfloat16/hiednn_bfloat16.hpp>
#include <datatype_extension/bfloat16/hiednn_bfloat16_cmath.hpp>
#endif
#include <device_function_modifier.hpp>

namespace hiednn {

/*
 * C++11 cmath funcions wrapper (in namespace ::hiednn)
 * support CUDA & CPP
 * including followed functions: (where T is half, bfloat16, float or double)
 *
 * for more information, refer to
 * https://en.cppreference.com/w/cpp/header/cmath
 *
 * --------------------------
 * basic operators
 * --------------------------
 * T cmath_fabs(T x)
 * T cmath_fmod(T x, T y)
 * T cmath_remainder(T x, T y)
 * T cmath_fma(T x, T y, T z)
 * T cmath_fmax(T x, T y)
 * T cmath_fmin(T x, T y)
 * T cmath_fdim(T x, T y)
 *
 * --------------------------
 * exponential functions
 * --------------------------
 * T cmath_exp(T x)
 * T cmath_exp2(T x)
 * T cmath_expm1(T x)
 * T cmath_log(T x)
 * T cmath_log10(T x)
 * T cmath_log2(T x)
 * T cmath_log1p(T x)
 *
 * --------------------------
 * power functions
 * --------------------------
 * T cmath_pow(T base, T exp)
 * T cmath_sqrt(T x)
 * T cmath_cbrt(T x)
 * T cmath_hypot(T x, T y)
 *
 * --------------------------
 * trigonometric functions
 * --------------------------
 * T cmath_sin(T x)
 * T cmath_cos(T x)
 * T cmath_tan(T x)
 * T cmath_asin(T x)
 * T cmath_acos(T x)
 * T cmath_atan(T x)
 * T cmath_atan2(T y, T x)
 *
 * --------------------------
 * hyperbolic functions
 * --------------------------
 * T cmath_sinh(T x)
 * T cmath_cosh(T x)
 * T cmath_tanh(T x)
 * T cmath_asinh(T x)
 * T cmath_acosh(T x)
 * T cmath_atanh(T x)
 *
 * --------------------------
 * error and gamma functions
 * --------------------------
 * T cmath_erf(T x)
 * T cmath_erfc(T x)
 * T cmath_tgamma(T x)
 * T cmath_lgamma(T x)
 *
 * --------------------------
 * nearest integer floating point operations
 * --------------------------
 * T cmath_ceil(T x)
 * T cmath_floor(T x)
 * T cmath_trunc(T x)
 * T cmath_round(T x)
 * T cmath_nearbyint(T x)
 * T cmath_rint(T x)
 *
 * --------------------------
 * nearest integer floating point operations
 * --------------------------
 * bool cmath_isfinite(T x)
 * bool cmath_isinf(T x)
 * bool cmath_isnan(T x)
 * bool cmath_signbit(T x)
 *
 */

#ifdef __CUDACC__
#define CMATH_NAMESPACE
#else
#define CMATH_NAMESPACE std
#endif

#ifdef HIEDNN_USE_FP16
#define CMATH_HALF_HH(func) \
DEVICE_FUNCTION half cmath_##func(half x) { \
    return func##_h(x); \
}
#define CMATH_HALF_HHH(func) \
DEVICE_FUNCTION half cmath_##func(half x, half y) { \
    return func##_h(x, y); \
}
#define CMATH_HALF_BH(func) \
DEVICE_FUNCTION bool cmath_##func(half x) { \
    return std::func(x); \
}
#else
#define CMATH_HALF_HH(func)
#define CMATH_HALF_HHH(func)
#define CMATH_HALF_BH(func)
#endif

#ifdef HIEDNN_USE_BF16
#define CMATH_BF16_BFBF(func) \
DEVICE_FUNCTION bfloat16 cmath_##func(bfloat16 x) { \
    return func##_bf16(x); \
}
#define CMATH_BF16_BFBFBF(func) \
DEVICE_FUNCTION bfloat16 cmath_##func(bfloat16 x, bfloat16 y) { \
    return func##_bf16(x, y); \
}
#define CMATH_BF16_BBF(func) \
DEVICE_FUNCTION bool cmath_##func(bfloat16 x) { \
    return std::func(x); \
}
#else
#define CMATH_BF16_BFBF(func)
#define CMATH_BF16_BFBFBF(func)
#define CMATH_BF16_BBF(func)
#endif

#define CMATH_WRAPPER_FF(func) \
CMATH_HALF_HH(func) \
CMATH_BF16_BFBF(func) \
DEVICE_FUNCTION float cmath_##func(float x) { \
    return func##f(x); \
} \
DEVICE_FUNCTION double cmath_##func(double x) { \
    return func(x); \
}

#define CMATH_WRAPPER_FFF(func) \
CMATH_HALF_HHH(func) \
CMATH_BF16_BFBFBF(func) \
DEVICE_FUNCTION float cmath_##func(float x, float y) { \
    return func##f(x, y); \
} \
DEVICE_FUNCTION double cmath_##func(double x, double y) { \
    return func(x, y); \
}

#define CMATH_WRAPPER_BF(func) \
CMATH_HALF_BH(func) \
CMATH_BF16_BBF(func) \
DEVICE_FUNCTION bool cmath_##func(float x) { \
    return CMATH_NAMESPACE::func(x); \
} \
DEVICE_FUNCTION bool cmath_##func(double x) { \
    return CMATH_NAMESPACE::func(x); \
}

// basic operators
CMATH_WRAPPER_FF(fabs)
CMATH_WRAPPER_FFF(fmod)
CMATH_WRAPPER_FFF(remainder)

DEVICE_FUNCTION float cmath_fma(float x, float y, float z) {
    return CMATH_NAMESPACE::fmaf(x, y, z);
}
DEVICE_FUNCTION double cmath_fma(double x, double y, double z) {
    return CMATH_NAMESPACE::fma(x, y, z);
}
#ifdef HIEDNN_USE_FP16
DEVICE_FUNCTION half cmath_fma(half x, half y, half z) {
    return fma_h(x, y, z);
}
#endif
#ifdef HIEDNN_USE_BF16
DEVICE_FUNCTION bfloat16 cmath_fma(bfloat16 x, bfloat16 y, bfloat16 z) {
    return fma_bf16(x, y, z);
}
#endif

CMATH_WRAPPER_FFF(fmax)
CMATH_WRAPPER_FFF(fmin)
CMATH_WRAPPER_FFF(fdim)

// exponential functions
CMATH_WRAPPER_FF(exp)
CMATH_WRAPPER_FF(exp2)
CMATH_WRAPPER_FF(expm1)
CMATH_WRAPPER_FF(log)
CMATH_WRAPPER_FF(log10)
CMATH_WRAPPER_FF(log2)
CMATH_WRAPPER_FF(log1p)

// power functions
CMATH_WRAPPER_FFF(pow)
CMATH_WRAPPER_FF(sqrt)
CMATH_WRAPPER_FF(cbrt)
CMATH_WRAPPER_FFF(hypot)

// trigonometric functions
CMATH_WRAPPER_FF(sin)
CMATH_WRAPPER_FF(cos)
CMATH_WRAPPER_FF(tan)
CMATH_WRAPPER_FF(asin)
CMATH_WRAPPER_FF(acos)
CMATH_WRAPPER_FF(atan)
CMATH_WRAPPER_FFF(atan2)

// hyperbolic functions
CMATH_WRAPPER_FF(sinh)
CMATH_WRAPPER_FF(cosh)
CMATH_WRAPPER_FF(tanh)
CMATH_WRAPPER_FF(asinh)
CMATH_WRAPPER_FF(acosh)
CMATH_WRAPPER_FF(atanh)

// error and gamma functions
CMATH_WRAPPER_FF(erf)
CMATH_WRAPPER_FF(erfc)
CMATH_WRAPPER_FF(tgamma)
CMATH_WRAPPER_FF(lgamma)

// nearest integer floating point operations
CMATH_WRAPPER_FF(ceil)
CMATH_WRAPPER_FF(floor)
CMATH_WRAPPER_FF(trunc)
CMATH_WRAPPER_FF(round)
CMATH_WRAPPER_FF(nearbyint)
CMATH_WRAPPER_FF(rint)

// classification
CMATH_WRAPPER_BF(isfinite)
CMATH_WRAPPER_BF(isinf)
CMATH_WRAPPER_BF(isnan)
CMATH_WRAPPER_BF(signbit)

#undef CMATH_HALF_HH
#undef CMATH_HALF_HHH
#undef CMATH_HALF_BH
#undef CMATH_BF16_BFBF
#undef CMATH_BF16_BFBFBF
#undef CMATH_BF16_BBF
#undef CMATH_WRAPPER_FF
#undef CMATH_WRAPPER_FFF
#undef CMATH_WRAPPER_BF
#undef CMATH_NAMESPACE

}  // namespace hiednn

#endif  // DNN_INCLUDE_CMATH_WRAPPER_HPP_
