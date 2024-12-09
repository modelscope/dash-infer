/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cppbfloat16_cmath_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_CMATH_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_CMATH_IMPL_HPP_

#include <cmath>
#include "cppbfloat16_impl.hpp"

#if defined(__CUDACC__)
#include "cudabfloat16_cmath_impl.hpp"
#endif

namespace __hiednn_buildin {

namespace __bf16_cmath_impl {

#ifndef __BF16_CMATH_IMPL
#define __BF16_CMATH_IMPL

#define __CMATH_DEVICE_FUNC inline

#define __FP32_FALLBACK1(func) \
__CMATH_DEVICE_FUNC _Bf16Impl func(_Bf16Impl arg) { \
    float val = _Bf16Impl::bfloat162float(arg); \
    return _Bf16Impl::float2bfloat16(std::func(val)); \
}

#define __FP32_FALLBACK2(func) \
__CMATH_DEVICE_FUNC _Bf16Impl func(_Bf16Impl x, _Bf16Impl y) { \
    float valx = _Bf16Impl::bfloat162float(x); \
    float valy = _Bf16Impl::bfloat162float(y); \
    return _Bf16Impl::float2bfloat16(std::func(valx, valy)); \
}

// basic operators
__FP32_FALLBACK1(fabs)
__FP32_FALLBACK2(fmod)
__FP32_FALLBACK2(remainder)

__CMATH_DEVICE_FUNC _Bf16Impl fma(_Bf16Impl x, _Bf16Impl y, _Bf16Impl z) {
    float valx = _Bf16Impl::bfloat162float(x);
    float valy = _Bf16Impl::bfloat162float(y);
    float valz = _Bf16Impl::bfloat162float(z);
    return _Bf16Impl::float2bfloat16(std::fma(valx, valy, valz));
}

__FP32_FALLBACK2(fmax)
__FP32_FALLBACK2(fmin)
__FP32_FALLBACK2(fdim)

// exponential functions
__FP32_FALLBACK1(exp)
__FP32_FALLBACK1(exp2)
__FP32_FALLBACK1(expm1)
__FP32_FALLBACK1(log)
__FP32_FALLBACK1(log10)
__FP32_FALLBACK1(log2)
__FP32_FALLBACK1(log1p)

// power functions
__FP32_FALLBACK2(pow)
__FP32_FALLBACK1(sqrt)

__CMATH_DEVICE_FUNC _Bf16Impl rsqrt(_Bf16Impl arg) {
    float val = _Bf16Impl::bfloat162float(arg);
    return _Bf16Impl::float2bfloat16(1.f / std::sqrt(val));
}

__FP32_FALLBACK1(cbrt)
__FP32_FALLBACK2(hypot)

// trigonometric functions
__FP32_FALLBACK1(sin)
__FP32_FALLBACK1(cos)
__FP32_FALLBACK1(tan)
__FP32_FALLBACK1(asin)
__FP32_FALLBACK1(acos)
__FP32_FALLBACK1(atan)
__FP32_FALLBACK2(atan2)

// hyperbolic functions
__FP32_FALLBACK1(sinh)
__FP32_FALLBACK1(cosh)
__FP32_FALLBACK1(tanh)
__FP32_FALLBACK1(asinh)
__FP32_FALLBACK1(acosh)
__FP32_FALLBACK1(atanh)

// error and gamma functions
__FP32_FALLBACK1(erf)
__FP32_FALLBACK1(erfc)
__FP32_FALLBACK1(tgamma)
__FP32_FALLBACK1(lgamma)

// nearest integer floating point operations
__FP32_FALLBACK1(ceil)
__FP32_FALLBACK1(floor)
__FP32_FALLBACK1(trunc)
__FP32_FALLBACK1(round)
__FP32_FALLBACK1(nearbyint)
__FP32_FALLBACK1(rint)

// classification
__CMATH_DEVICE_FUNC bool isinf(_Bf16Impl arg) {
    return (arg._x & 0x7fffU) == 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isnan(_Bf16Impl arg) {
    return (arg._x & 0x7fffU) > 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isfinite(_Bf16Impl arg) {
    return (arg._x & 0x7fffU) < 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isnormal(_Bf16Impl arg) {
    uint16_t expo = reinterpret_cast<uint16_t &>(arg) & 0x7f80;
    return expo < 0x7f80 && expo != 0;
}
__CMATH_DEVICE_FUNC bool signbit(_Bf16Impl arg) {
    return (arg._x & 0x8000) != 0;
}

#undef __FP32_FALLBACK1
#undef __FP32_FALLBACK2

#endif  //  __BF16_CMATH_IMPL

}  // namespace __bf16_cmath_impl

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_CMATH_IMPL_HPP_
