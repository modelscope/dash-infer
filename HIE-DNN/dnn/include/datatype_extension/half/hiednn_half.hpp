/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_half.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_HPP_

#include <cstddef>
#include <cstdint>
#include "cpphalf_impl.hpp"

namespace __hiednn_buildin {

constexpr size_t __HALF_SIZE_INBYTE = 2;

// ---------------------------------------------
// extended types forward declaration
// ---------------------------------------------
#ifdef HIEDNN_USE_BF16
struct bfloat16;
#endif

#if !defined(__HALF_IMPL)
#error __HALF_IMPL not defined
#endif

struct half {
    _HalfImpl _x;
    static_assert(sizeof(_HalfImpl) == __HALF_SIZE_INBYTE,
                  "invalid _HalfImpl size");

    /*
     * default rounding mode (float/double to half): nearest
     * Note we do avoid constructor init-list because of special host/device compilation rules
     */

    __F16_DEVICE_FUNC half() {}

    __F16_DEVICE_FUNC explicit half(_HalfImpl v) { _x = v; }

    // ---------------------------------------------
    // convert from other extended types
    // ---------------------------------------------
#ifdef HIEDNN_USE_BF16
    __F16_DEVICE_FUNC half(bfloat16 v);
#endif

    // ---------------------------------------------
    // convert from build-in types
    // ---------------------------------------------
    __F16_DEVICE_FUNC half(float v) {
        _x = _HalfImpl::float2half(v);
    }

    __F16_DEVICE_FUNC half(double v) {
        _x = _HalfImpl::double2half(v);
    }

    __F16_DEVICE_FUNC half(long long v) {
        _x = _HalfImpl::ll2half(v);
    }

    __F16_DEVICE_FUNC half(unsigned long long v) {
        _x = _HalfImpl::ull2half(v);
    }

    __F16_DEVICE_FUNC half(long v) {
        _x = _HalfImpl::ll2half(v);
    }

    __F16_DEVICE_FUNC half(unsigned long v) {
        _x = _HalfImpl::ull2half(v);
    }

    __F16_DEVICE_FUNC half(int v) {
        _x = _HalfImpl::int2half(v);
    }

    __F16_DEVICE_FUNC half(unsigned int v) {
        _x = _HalfImpl::uint2half(v);
    }

    __F16_DEVICE_FUNC half(short v) {
        _x = _HalfImpl::short2half(v);
    }

    __F16_DEVICE_FUNC half(unsigned short v) {
        _x = _HalfImpl::ushort2half(v);
    }

    __F16_DEVICE_FUNC half(char v) {
        _x = _HalfImpl::short2half(static_cast<short>(v));
    }

    __F16_DEVICE_FUNC half(signed char v) {
        _x = _HalfImpl::short2half(static_cast<short>(v));
    }

    __F16_DEVICE_FUNC half(unsigned char v) {
        _x = _HalfImpl::short2half(static_cast<short>(v));
    }

    __F16_DEVICE_FUNC half(bool v) {
        _x = _HalfImpl::short2half(static_cast<short>(v));
    }

    // ---------------------------------------------
    // convert to build-in types
    // ---------------------------------------------
    __F16_DEVICE_FUNC operator float() const {
        return _HalfImpl::half2float(_x);
    }

    __F16_DEVICE_FUNC operator double() const {
        return _HalfImpl::half2double(_x);
    }

    __F16_DEVICE_FUNC operator long long() const {
        return _HalfImpl::half2ll(_x);
    }

    __F16_DEVICE_FUNC operator unsigned long long() const {
        return _HalfImpl::half2ull(_x);
    }

    __F16_DEVICE_FUNC operator long() const {
        return _HalfImpl::half2ll(_x);
    }

    __F16_DEVICE_FUNC operator unsigned long() const {
        return _HalfImpl::half2ull(_x);
    }

    __F16_DEVICE_FUNC operator int() const {
        return _HalfImpl::half2int(_x);
    }

    __F16_DEVICE_FUNC operator unsigned int() const {
        return _HalfImpl::half2uint(_x);
    }

    __F16_DEVICE_FUNC operator short() const {
        return _HalfImpl::half2short(_x);
    }

    __F16_DEVICE_FUNC operator unsigned short() const {
        return _HalfImpl::half2ushort(_x);
    }

    __F16_DEVICE_FUNC operator char() const {
        return static_cast<char>(_HalfImpl::half2short(_x));
    }

    __F16_DEVICE_FUNC operator signed char() const {
        return static_cast<signed char>(_HalfImpl::half2short(_x));
    }

    __F16_DEVICE_FUNC operator unsigned char() const {
        return static_cast<unsigned char>(_HalfImpl::half2short(_x));
    }

    __F16_DEVICE_FUNC operator bool() const {
        return (reinterpret_cast<const std::uint16_t &>(_x) & 0x7fff) != 0;
    }
};  // struct half

// positive & negative
__F16_DEVICE_FUNC half operator-(half a) {
    std::uint16_t val = reinterpret_cast<std::uint16_t &>(a) ^ 0x8000;
    return reinterpret_cast<const half &>(val);
}
__F16_DEVICE_FUNC half operator+(half a) { return a; }


#define __TO_HALF_BINARY_OPERATOR(op, type) \
__F16_DEVICE_FUNC half operator op(type a, half b) { \
    return static_cast<half>(a) op b; \
} \
__F16_DEVICE_FUNC half operator op(half a, type b) { \
    return a op static_cast<half>(b); \
}

#define __HALF_BINARY_OPERATOR(op, impl_expr) \
__F16_DEVICE_FUNC float operator op(float a, half b) { \
    return a op static_cast<float>(b); \
} \
__F16_DEVICE_FUNC float operator op(half a, float b) { \
    return static_cast<float>(a) op b; \
} \
__F16_DEVICE_FUNC double operator op(double a, half b) { \
    return a op static_cast<double>(b); \
} \
__F16_DEVICE_FUNC double operator op(half a, double b) { \
    return static_cast<double>(a) op b; \
} \
__F16_DEVICE_FUNC half operator op(half a, half b) { \
    return half(_HalfImpl::impl_expr(a._x, b._x)); \
} \
__TO_HALF_BINARY_OPERATOR(op, long long) \
__TO_HALF_BINARY_OPERATOR(op, unsigned long long) \
__TO_HALF_BINARY_OPERATOR(op, long) \
__TO_HALF_BINARY_OPERATOR(op, unsigned long) \
__TO_HALF_BINARY_OPERATOR(op, int) \
__TO_HALF_BINARY_OPERATOR(op, unsigned int) \
__TO_HALF_BINARY_OPERATOR(op, short) \
__TO_HALF_BINARY_OPERATOR(op, unsigned short) \
__TO_HALF_BINARY_OPERATOR(op, char) \
__TO_HALF_BINARY_OPERATOR(op, signed char) \
__TO_HALF_BINARY_OPERATOR(op, unsigned char) \
__TO_HALF_BINARY_OPERATOR(op, bool)

// + - * /
__HALF_BINARY_OPERATOR(+, hadd)
__HALF_BINARY_OPERATOR(-, hsub)
__HALF_BINARY_OPERATOR(*, hmul)
__HALF_BINARY_OPERATOR(/, hdiv)
#undef __HALF_BINARY_OPERATOR
#undef __TO_HALF_BINARY_OPERATOR

// += -= *= /=
__F16_DEVICE_FUNC
half &operator+=(half &a, const half &b) { a = a + b; return a; }
__F16_DEVICE_FUNC
half &operator-=(half &a, const half &b) { a = a - b; return a; }
__F16_DEVICE_FUNC
half &operator*=(half &a, const half &b) { a = a * b; return a; }
__F16_DEVICE_FUNC
half &operator/=(half &a, const half &b) { a = a / b; return a; }

// ++ --
__F16_DEVICE_FUNC half &operator++(half &v) {
    std::uint16_t hone = 0x3c00;
    v += reinterpret_cast<const half &>(hone);
    return v;
}
__F16_DEVICE_FUNC half &operator--(half &v) {
    std::uint16_t hone = 0x3c00;
    v -= reinterpret_cast<const half &>(hone);
    return v;
}
__F16_DEVICE_FUNC half operator++(half &v, int) {
    half r = v;
    std::uint16_t hone = 0x3c00;
    v += reinterpret_cast<const half &>(hone);
    return r;
}
__F16_DEVICE_FUNC half operator--(half &v, int) {
    half r = v;
    std::uint16_t hone = 0x3c00;
    v -= reinterpret_cast<const half &>(hone);
    return r;
}

#define __TO_HALF_CMP_OPERATOR(op, type) \
__F16_DEVICE_FUNC bool operator op(type a, half b) { \
    return static_cast<half>(a) op b; \
} \
__F16_DEVICE_FUNC bool operator op(half a, type b) { \
    return a op static_cast<half>(b); \
}

#define __HALF_CMP_OPERATOR(op, impl_expr) \
__F16_DEVICE_FUNC bool operator op(float a, half b) { \
    return a op static_cast<float>(b); \
} \
__F16_DEVICE_FUNC bool operator op(half a, float b) { \
    return static_cast<float>(a) op b; \
} \
__F16_DEVICE_FUNC bool operator op(double a, half b) { \
    return a op static_cast<double>(b); \
} \
__F16_DEVICE_FUNC bool operator op(half a, double b) { \
    return static_cast<double>(a) op b; \
} \
__F16_DEVICE_FUNC bool operator op(half a, half b) { \
    return _HalfImpl::impl_expr(a._x, b._x); \
} \
__TO_HALF_CMP_OPERATOR(op, long long) \
__TO_HALF_CMP_OPERATOR(op, unsigned long long) \
__TO_HALF_CMP_OPERATOR(op, long) \
__TO_HALF_CMP_OPERATOR(op, unsigned long) \
__TO_HALF_CMP_OPERATOR(op, int) \
__TO_HALF_CMP_OPERATOR(op, unsigned int) \
__TO_HALF_CMP_OPERATOR(op, short) \
__TO_HALF_CMP_OPERATOR(op, unsigned short) \
__TO_HALF_CMP_OPERATOR(op, char) \
__TO_HALF_CMP_OPERATOR(op, signed char) \
__TO_HALF_CMP_OPERATOR(op, unsigned char) \
__TO_HALF_CMP_OPERATOR(op, bool)

// == != > < >= <=
__HALF_CMP_OPERATOR(==, heq)
__HALF_CMP_OPERATOR(!=, hne)
__HALF_CMP_OPERATOR(>, hgt)
__HALF_CMP_OPERATOR(<, hlt)
__HALF_CMP_OPERATOR(>=, hge)
__HALF_CMP_OPERATOR(<=, hle)
#undef __HALF_CMP_OPERATOR
#undef __TO_HALF_CMP_OPERATOR

#undef __HALF_IMPL
#undef __F16_DEVICE_FUNC

}  // namespace __hiednn_buildin

namespace hiednn {
typedef __hiednn_buildin::half half;
}

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_HIEDNN_HALF_HPP_
