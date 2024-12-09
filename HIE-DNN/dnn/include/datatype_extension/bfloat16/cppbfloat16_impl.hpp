/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cppbfloat16_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_IMPL_HPP_

#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#include "cudabfloat16_impl.hpp"
#endif

namespace __hiednn_buildin {

#ifndef __BF16_IMPL
#define __BF16_IMPL

#define __BF16_DEVICE_FUNC inline

struct __attribute__((aligned(2))) _Bf16Impl {
    uint16_t _x;

    _Bf16Impl() { _x = 0; }

    // from float to bf16
    // 0 ---> 0
    // INF ---> INF
    // DENORMALISED ---> 0(this will be seemed as loss of significance)
    //                   OR DENORMALISED
    // NAN ---> NAN
    static _Bf16Impl float2bfloat16(float v) {
        _Bf16Impl ret;
        const uint32_t *vptr = reinterpret_cast<const uint32_t *>(&v);
        const uint32_t &vi = *vptr;
        // if v is nan, make sure bf16(v) is nan
        if ((vi & 0x7fffffffU) > 0x7f800000U) {
            ret._x = 0x7fffU;
        } else {
            ret._x = vi >> 16;
            // round to nearest
            if ((vi & 0xffffU) > 0x8000U) {
                ++ret._x;
            }
        }
        return ret;
    }

    static _Bf16Impl double2bfloat16(double v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl ll2bfloat16(long long v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl ull2bfloat16(unsigned long long v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl int2bfloat16(int v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl uint2bfloat16(unsigned int v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl short2bfloat16(short v) {
        return float2bfloat16(static_cast<float>(v));
    }

    static _Bf16Impl ushort2bfloat16(unsigned short v) {
        return float2bfloat16(static_cast<float>(v));
    }

    // from bf16 to float
    // 0 ---> 0
    // INF ---> INF
    // DENORMALISED ---> DENORMALISED
    // NAN ---> NAN
    static float bfloat162float(_Bf16Impl v) {
        uint32_t reti = static_cast<uint32_t>(v._x) << 16;
        const float *retfptr = reinterpret_cast<const float *>(&reti);
        const float &retf = *retfptr;
        return retf;
    }

    static double bfloat162double(_Bf16Impl v) {
        return static_cast<double>(bfloat162float(v));
    }

    static long long bfloat162ll(_Bf16Impl v) {
        return static_cast<long long>(bfloat162float(v));
    }

    static unsigned long long bfloat162ull(_Bf16Impl v) {
        return static_cast<unsigned long long>(bfloat162float(v));
    }

    static int bfloat162int(_Bf16Impl v) {
        return static_cast<int>(bfloat162float(v));
    }

    static unsigned int bfloat162uint(_Bf16Impl v) {
        return static_cast<unsigned int>(bfloat162float(v));
    }

    static short bfloat162short(_Bf16Impl v) {
        return static_cast<short>(bfloat162float(v));
    }

    static unsigned short bfloat162ushort(_Bf16Impl v) {
        return static_cast<unsigned short>(bfloat162float(v));
    }

    // + - * /
    static _Bf16Impl bf16add(_Bf16Impl a, _Bf16Impl b) {
        float val = bfloat162float(a) + bfloat162float(b);
        return float2bfloat16(val);
    }

    static _Bf16Impl bf16sub(_Bf16Impl a, _Bf16Impl b) {
        float val = bfloat162float(a) - bfloat162float(b);
        return float2bfloat16(val);
    }

    static _Bf16Impl bf16mul(_Bf16Impl a, _Bf16Impl b) {
        float val = bfloat162float(a) * bfloat162float(b);
        return float2bfloat16(val);
    }

    static _Bf16Impl bf16div(_Bf16Impl a, _Bf16Impl b) {
        float val = bfloat162float(a) / bfloat162float(b);
        return float2bfloat16(val);
    }

    // == != > < >= <=
    static bool bf16eq(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) == bfloat162float(b);
    }

    static bool bf16ne(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) != bfloat162float(b);
    }

    static bool bf16gt(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) > bfloat162float(b);
    }

    static bool bf16lt(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) < bfloat162float(b);
    }

    static bool bf16ge(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) >= bfloat162float(b);
    }

    static bool bf16le(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) <= bfloat162float(b);
    }
};

#endif  // __BF16_IMPL

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CPPBFLOAT16_IMPL_HPP_
