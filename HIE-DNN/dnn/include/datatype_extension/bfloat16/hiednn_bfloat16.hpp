/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_bfloat16.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_HIEDNN_BFLOAT16_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_HIEDNN_BFLOAT16_HPP_

#include <cstdint>

#include "cppbfloat16_impl.hpp"

namespace __hiednn_buildin {

constexpr size_t __BF16_SIZE_INBYTE = 2;

// ---------------------------------------------
// extended types forward declaration
// ---------------------------------------------
#ifdef HIEDNN_USE_FP16
struct half;
#endif

#if !defined(__BF16_IMPL)
#error __BF16_IMPL not defined
#endif

struct bfloat16 {
    _Bf16Impl _x;
    static_assert(sizeof(_Bf16Impl) == __BF16_SIZE_INBYTE,
                  "invalid _Bf16Impl size");

/*
 * default rounding mode (float/double to bf16): nearest
 * Note we do avoid constructor init-list because of special host/device compilation rules
 */

    __BF16_DEVICE_FUNC bfloat16() {}

    __BF16_DEVICE_FUNC explicit bfloat16(_Bf16Impl v) { _x = v; }

    // ---------------------------------------------
    // convert from other extended types
    // ---------------------------------------------
#ifdef HIEDNN_USE_FP16
    __BF16_DEVICE_FUNC bfloat16(half v);
#endif

    // ---------------------------------------------
    // convert from build-in types
    // ---------------------------------------------
    __BF16_DEVICE_FUNC bfloat16(float v) {
        _x = _Bf16Impl::float2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(double v) {
        _x = _Bf16Impl::double2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(long long v) {
        _x = _Bf16Impl::ll2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned long long v) {
        _x = _Bf16Impl::ull2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(long v) {
        _x = _Bf16Impl::ll2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned long v) {
        _x = _Bf16Impl::ull2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(int v) {
        _x = _Bf16Impl::int2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned int v) {
        _x = _Bf16Impl::uint2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(short v) {
        _x = _Bf16Impl::short2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned short v) {
        _x = _Bf16Impl::ushort2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(char v) {
        _x = _Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(signed char v) {
        _x = _Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned char v) {
        _x = _Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(bool v) {
        _x = _Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    // ---------------------------------------------
    // convert to build-in types
    // ---------------------------------------------
    __BF16_DEVICE_FUNC operator float() const {
        return _Bf16Impl::bfloat162float(_x);
    }

    __BF16_DEVICE_FUNC operator double() const {
        return _Bf16Impl::bfloat162double(_x);
    }

    __BF16_DEVICE_FUNC operator long long() const {
        return _Bf16Impl::bfloat162ll(_x);
    }

    __BF16_DEVICE_FUNC operator unsigned long long() const {
        return _Bf16Impl::bfloat162ull(_x);
    }

    __BF16_DEVICE_FUNC operator long() const {
        return _Bf16Impl::bfloat162ll(_x);
    }

    __BF16_DEVICE_FUNC operator unsigned long() const {
        return _Bf16Impl::bfloat162ull(_x);
    }

    __BF16_DEVICE_FUNC operator int() const {
        return _Bf16Impl::bfloat162int(_x);
    }

    __BF16_DEVICE_FUNC operator unsigned int() const {
        return _Bf16Impl::bfloat162uint(_x);
    }

    __BF16_DEVICE_FUNC operator short() const {
        return _Bf16Impl::bfloat162short(_x);
    }

    __BF16_DEVICE_FUNC operator unsigned short() const {
        return _Bf16Impl::bfloat162ushort(_x);
    }

    __BF16_DEVICE_FUNC operator char() const {
        return static_cast<char>(_Bf16Impl::bfloat162short(_x));
    }

    __BF16_DEVICE_FUNC operator signed char() const {
        return static_cast<signed char>(_Bf16Impl::bfloat162short(_x));
    }

    __BF16_DEVICE_FUNC operator unsigned char() const {
        return static_cast<unsigned char>(_Bf16Impl::bfloat162short(_x));
    }

    __BF16_DEVICE_FUNC operator bool() const {
        return (reinterpret_cast<const std::uint16_t &>(_x) & 0x7fff) != 0;
    }
};  // struct bfloat16

// positive & negative
__BF16_DEVICE_FUNC bfloat16 operator-(bfloat16 a) {
    std::uint16_t ret = reinterpret_cast<std::uint16_t &>(a) ^ 0x8000;
    return reinterpret_cast<bfloat16 &>(ret);
}
__BF16_DEVICE_FUNC bfloat16 operator+(bfloat16 a) { return a; }


#define __TO_BF16_BINARY_OPERATOR(op, type) \
__BF16_DEVICE_FUNC bfloat16 operator op(type a, bfloat16 b) { \
    return static_cast<bfloat16>(a) op b; \
} \
__BF16_DEVICE_FUNC bfloat16 operator op(bfloat16 a, type b) { \
    return a op static_cast<bfloat16>(b); \
}

#define __BF16_BINARY_OPERATOR(op, impl_expr) \
__BF16_DEVICE_FUNC float operator op(float a, bfloat16 b) { \
    return a op static_cast<float>(b); \
} \
__BF16_DEVICE_FUNC float operator op(bfloat16 a, float b) { \
    return static_cast<float>(a) op b; \
} \
__BF16_DEVICE_FUNC double operator op(double a, bfloat16 b) { \
    return a op static_cast<double>(b); \
} \
__BF16_DEVICE_FUNC double operator op(bfloat16 a, double b) { \
    return static_cast<double>(a) op b; \
} \
__BF16_DEVICE_FUNC bfloat16 operator op(bfloat16 a, bfloat16 b) { \
    return bfloat16(_Bf16Impl::impl_expr(a._x, b._x)); \
} \
__TO_BF16_BINARY_OPERATOR(op, long long) \
__TO_BF16_BINARY_OPERATOR(op, unsigned long long) \
__TO_BF16_BINARY_OPERATOR(op, long) \
__TO_BF16_BINARY_OPERATOR(op, unsigned long) \
__TO_BF16_BINARY_OPERATOR(op, int) \
__TO_BF16_BINARY_OPERATOR(op, unsigned int) \
__TO_BF16_BINARY_OPERATOR(op, short) \
__TO_BF16_BINARY_OPERATOR(op, unsigned short) \
__TO_BF16_BINARY_OPERATOR(op, char) \
__TO_BF16_BINARY_OPERATOR(op, signed char) \
__TO_BF16_BINARY_OPERATOR(op, unsigned char) \
__TO_BF16_BINARY_OPERATOR(op, bool)

// + - * /
__BF16_BINARY_OPERATOR(+, bf16add)
__BF16_BINARY_OPERATOR(-, bf16sub)
__BF16_BINARY_OPERATOR(*, bf16mul)
__BF16_BINARY_OPERATOR(/, bf16div)
#undef __BF16_BINARY_OPERATOR
#undef __TO_BF16_BINARY_OPERATOR

// += -= *= /=
__BF16_DEVICE_FUNC
bfloat16 &operator+=(bfloat16 &a, const bfloat16 &b) { a = a + b; return a; }
__BF16_DEVICE_FUNC
bfloat16 &operator-=(bfloat16 &a, const bfloat16 &b) { a = a - b; return a; }
__BF16_DEVICE_FUNC
bfloat16 &operator*=(bfloat16 &a, const bfloat16 &b) { a = a * b; return a; }
__BF16_DEVICE_FUNC
bfloat16 &operator/=(bfloat16 &a, const bfloat16 &b) { a = a / b; return a; }

// ++ --
__BF16_DEVICE_FUNC bfloat16 &operator++(bfloat16 &v) {
    std::uint16_t one = 0x3f80;
    v += reinterpret_cast<const bfloat16 &>(one);
    return v;
}
__BF16_DEVICE_FUNC bfloat16 &operator--(bfloat16 &v) {
    std::uint16_t one = 0x3f80;
    v -= reinterpret_cast<const bfloat16 &>(one);
    return v;
}
__BF16_DEVICE_FUNC bfloat16 operator++(bfloat16 &v, int) {
    bfloat16 r = v;
    std::uint16_t one = 0x3f80;
    v += reinterpret_cast<const bfloat16 &>(one);
    return r;
}
__BF16_DEVICE_FUNC bfloat16 operator--(bfloat16 &v, int) {
    bfloat16 r = v;
    std::uint16_t one = 0x3f80;
    v -= reinterpret_cast<const bfloat16 &>(one);
    return r;
}

#define __TO_BF16_CMP_OPERATOR(op, type) \
__BF16_DEVICE_FUNC bool operator op(type a, bfloat16 b) { \
    return static_cast<bfloat16>(a) op b; \
} \
__BF16_DEVICE_FUNC bool operator op(bfloat16 a, type b) { \
    return a op static_cast<bfloat16>(b); \
}

#define __BF16_CMP_OPERATOR(op, impl_expr) \
__BF16_DEVICE_FUNC bool operator op(float a, bfloat16 b) { \
    return a op static_cast<float>(b); \
} \
__BF16_DEVICE_FUNC bool operator op(bfloat16 a, float b) { \
    return static_cast<float>(a) op b; \
} \
__BF16_DEVICE_FUNC bool operator op(double a, bfloat16 b) { \
    return a op static_cast<double>(b); \
} \
__BF16_DEVICE_FUNC bool operator op(bfloat16 a, double b) { \
    return static_cast<double>(a) op b; \
} \
__BF16_DEVICE_FUNC bool operator op(bfloat16 a, bfloat16 b) { \
    return _Bf16Impl::impl_expr(a._x, b._x); \
} \
__TO_BF16_CMP_OPERATOR(op, long long) \
__TO_BF16_CMP_OPERATOR(op, unsigned long long) \
__TO_BF16_CMP_OPERATOR(op, long) \
__TO_BF16_CMP_OPERATOR(op, unsigned long) \
__TO_BF16_CMP_OPERATOR(op, int) \
__TO_BF16_CMP_OPERATOR(op, unsigned int) \
__TO_BF16_CMP_OPERATOR(op, short) \
__TO_BF16_CMP_OPERATOR(op, unsigned short) \
__TO_BF16_CMP_OPERATOR(op, char) \
__TO_BF16_CMP_OPERATOR(op, signed char) \
__TO_BF16_CMP_OPERATOR(op, unsigned char) \
__TO_BF16_CMP_OPERATOR(op, bool)

// == != > < >= <=
__BF16_CMP_OPERATOR(==, bf16eq)
__BF16_CMP_OPERATOR(!=, bf16ne)
__BF16_CMP_OPERATOR(>, bf16gt)
__BF16_CMP_OPERATOR(<, bf16lt)
__BF16_CMP_OPERATOR(>=, bf16ge)
__BF16_CMP_OPERATOR(<=, bf16le)
#undef __BF16_CMP_OPERATOR
#undef __TO_BF16_CMP_OPERATOR

#undef __BF16_IMPL
#undef __BF16_DEVICE_FUNC

}  // namespace __hiednn_buildin

namespace hiednn {
typedef __hiednn_buildin::bfloat16 bfloat16;
}

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_HIEDNN_BFLOAT16_HPP_
