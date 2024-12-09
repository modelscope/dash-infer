/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_half_cmath.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_CMATH_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_CMATH_HPP_

#include "hiednn_half.hpp"
#include "cpphalf_cmath_impl.hpp"

// ------------------------------------
//  C++11 cmath API for FP16
// ------------------------------------

#if !defined(__HALF_CMATH_IMPL)
#error __HALF_CMATH_IMPL not defined
#endif

#define __HIEDNN_CMATH_API_HH(func, expr) \
__CMATH_DEVICE_FUNC __hiednn_buildin::half func(__hiednn_buildin::half arg) { \
    return __hiednn_buildin::half( \
            __hiednn_buildin::__half_cmath_impl::expr(arg._x)); \
}

#define __HIEDNN_CMATH_API_HHH(func, expr) \
__CMATH_DEVICE_FUNC __hiednn_buildin::half \
func(__hiednn_buildin::half x, __hiednn_buildin::half y) { \
    return __hiednn_buildin::half( \
            __hiednn_buildin::__half_cmath_impl::expr(x._x, y._x)); \
}

#define __HIEDNN_CMATH_API_BH(func) \
__CMATH_DEVICE_FUNC bool func(::__hiednn_buildin::half arg) { \
    return ::__hiednn_buildin::__half_cmath_impl::func(arg._x); \
}

// basic operators
__HIEDNN_CMATH_API_HH(fabs_h, fabs)
__HIEDNN_CMATH_API_HHH(fmod_h, fmod)
__HIEDNN_CMATH_API_HHH(remainder_h, remainder)

__CMATH_DEVICE_FUNC __hiednn_buildin::half
fma_h(__hiednn_buildin::half x,
        __hiednn_buildin::half y, __hiednn_buildin::half z) {
    return __hiednn_buildin::half(
        __hiednn_buildin::__half_cmath_impl::fma(x._x, y._x, z._x));
}

__HIEDNN_CMATH_API_HHH(fmax_h, fmax)
__HIEDNN_CMATH_API_HHH(fmin_h, fmin)
__HIEDNN_CMATH_API_HHH(fdim_h, fdim)

// exponential functions
__HIEDNN_CMATH_API_HH(exp_h, exp)
__HIEDNN_CMATH_API_HH(exp2_h, exp2)
__HIEDNN_CMATH_API_HH(expm1_h, expm1)
__HIEDNN_CMATH_API_HH(log_h, log)
__HIEDNN_CMATH_API_HH(log10_h, log10)
__HIEDNN_CMATH_API_HH(log2_h, log2)
__HIEDNN_CMATH_API_HH(log1p_h, log1p)

// power functions
__HIEDNN_CMATH_API_HHH(pow_h, pow)
__HIEDNN_CMATH_API_HH(sqrt_h, sqrt)
__HIEDNN_CMATH_API_HH(rsqrt_h, rsqrt)
__HIEDNN_CMATH_API_HH(cbrt_h, cbrt)
__HIEDNN_CMATH_API_HHH(hypot_h, hypot)

// trigonometric functions
__HIEDNN_CMATH_API_HH(sin_h, sin)
__HIEDNN_CMATH_API_HH(cos_h, cos)
__HIEDNN_CMATH_API_HH(tan_h, tan)
__HIEDNN_CMATH_API_HH(asin_h, asin)
__HIEDNN_CMATH_API_HH(acos_h, acos)
__HIEDNN_CMATH_API_HH(atan_h, atan)
__HIEDNN_CMATH_API_HHH(atan2_h, atan2)

// hyperbolic functions
__HIEDNN_CMATH_API_HH(sinh_h, sinh)
__HIEDNN_CMATH_API_HH(cosh_h, cosh)
__HIEDNN_CMATH_API_HH(tanh_h, tanh)
__HIEDNN_CMATH_API_HH(asinh_h, asinh)
__HIEDNN_CMATH_API_HH(acosh_h, acosh)
__HIEDNN_CMATH_API_HH(atanh_h, atanh)

// error and gamma functions
__HIEDNN_CMATH_API_HH(erf_h, erf)
__HIEDNN_CMATH_API_HH(erfc_h, erfc)
__HIEDNN_CMATH_API_HH(tgamma_h, tgamma)
__HIEDNN_CMATH_API_HH(lgamma_h, lgamma)

// nearest integer floating point operations
__HIEDNN_CMATH_API_HH(ceil_h, ceil)
__HIEDNN_CMATH_API_HH(floor_h, floor)
__HIEDNN_CMATH_API_HH(trunc_h, trunc)
__HIEDNN_CMATH_API_HH(round_h, round)
__HIEDNN_CMATH_API_HH(nearbyint_h, nearbyint)
__HIEDNN_CMATH_API_HH(rint_h, rint)

// classification
namespace std {
__HIEDNN_CMATH_API_BH(isfinite)
__HIEDNN_CMATH_API_BH(isinf)
__HIEDNN_CMATH_API_BH(isnan)
__HIEDNN_CMATH_API_BH(isnormal)
__HIEDNN_CMATH_API_BH(signbit)
}  // namespace std

#undef __HIEDNN_CMATH_API_HH
#undef __HIEDNN_CMATH_API_HHH
#undef __HIEDNN_CMATH_API_BH

#undef __HALF_CMATH_IMPL
#undef __CMATH_DEVICE_FUNC

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_CMATH_HPP_
