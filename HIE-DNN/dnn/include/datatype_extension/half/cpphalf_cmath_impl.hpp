/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpphalf_cmath_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_CMATH_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_CMATH_IMPL_HPP_

#include "cpphalf_impl.hpp"

#if defined (__CUDACC__)
#include "cudahalf_cmath_impl.hpp"
#endif

namespace __hiednn_buildin {

namespace __half_cmath_impl {

#ifndef __HALF_CMATH_IMPL
#define __HALF_CMATH_IMPL

#define __CMATH_DEVICE_FUNC inline

#define __CMATH_IMPL1(func) \
__CMATH_DEVICE_FUNC _HalfImpl func(_HalfImpl arg) { \
    return _HalfImpl(func(arg._x)); \
}

#define __CMATH_IMPL2(func) \
__CMATH_DEVICE_FUNC _HalfImpl func(_HalfImpl x, _HalfImpl y) { \
    return _HalfImpl(func(x._x, y._x)); \
}

#define __CMATH_IMPL_BOOL(func) \
__CMATH_DEVICE_FUNC bool func(_HalfImpl arg) { \
    return func(arg._x); \
}

// basic operators
__CMATH_IMPL1(fabs)
__CMATH_IMPL2(fmod)
__CMATH_IMPL2(remainder)

__CMATH_DEVICE_FUNC _HalfImpl fma(_HalfImpl x, _HalfImpl y, _HalfImpl z) {
    return _HalfImpl(fma(x._x, y._x, z._x));
}

__CMATH_IMPL2(fmax)
__CMATH_IMPL2(fmin)
__CMATH_IMPL2(fdim)

// exponential functions
__CMATH_IMPL1(exp)
__CMATH_IMPL1(exp2)
__CMATH_IMPL1(expm1)
__CMATH_IMPL1(log)
__CMATH_IMPL1(log10)
__CMATH_IMPL1(log2)
__CMATH_IMPL1(log1p)

// power functions
__CMATH_IMPL2(pow)
__CMATH_IMPL1(sqrt)

__CMATH_DEVICE_FUNC _HalfImpl rsqrt(_HalfImpl arg) {
    std::uint16_t one = 0x3c00;
    return _HalfImpl(reinterpret_cast<_HalfImpl &>(one)._x / sqrt(arg._x));
}

__CMATH_IMPL1(cbrt)
__CMATH_IMPL2(hypot)

// trigonometric functions
__CMATH_IMPL1(sin)
__CMATH_IMPL1(cos)
__CMATH_IMPL1(tan)
__CMATH_IMPL1(asin)
__CMATH_IMPL1(acos)
__CMATH_IMPL1(atan)
__CMATH_IMPL2(atan2)

// hyperbolic functions
__CMATH_IMPL1(sinh)
__CMATH_IMPL1(cosh)
__CMATH_IMPL1(tanh)
__CMATH_IMPL1(asinh)
__CMATH_IMPL1(acosh)
__CMATH_IMPL1(atanh)

// error and gamma functions
__CMATH_IMPL1(erf)
__CMATH_IMPL1(erfc)
__CMATH_IMPL1(tgamma)
__CMATH_IMPL1(lgamma)

// nearest integer floating point operations
__CMATH_IMPL1(ceil)
__CMATH_IMPL1(floor)
__CMATH_IMPL1(trunc)
__CMATH_IMPL1(round)
__CMATH_IMPL1(nearbyint)
__CMATH_IMPL1(rint)

// classification
__CMATH_IMPL_BOOL(isfinite)
__CMATH_IMPL_BOOL(isinf)
__CMATH_IMPL_BOOL(isnan)
__CMATH_IMPL_BOOL(isnormal)
__CMATH_IMPL_BOOL(signbit)

#undef __CMATH_IMPL1
#undef __CMATH_IMPL2
#undef __CMATH_IMPL_BOOL

#endif  //  __HALF_CMATH_IMPL

}  // namespace __half_cmath_impl

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_CMATH_IMPL_HPP_
