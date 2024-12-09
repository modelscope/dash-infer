/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudabfloat16_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_IMPL_HPP_

#include <cuda.h>
#include <cstdint>

namespace __hiednn_buildin {

#if defined(__CUDA_ARCH__)

#define __BF16_IMPL

#define __BF16_DEVICE_FUNC __device__ __forceinline__
#define __CUBF16_IMPL static __device__ __forceinline__
#define __BF16_NATIVE(_SM_CHECK, _CUDA_VERSION_CHECK) \
    (__CUDA_ARCH__ >= _SM_CHECK) && (CUDA_VERSION >= _CUDA_VERSION_CHECK)

struct __attribute__((aligned(2))) _Bf16Impl {
    uint16_t _x;

    __BF16_DEVICE_FUNC _Bf16Impl() {}

    __CUBF16_IMPL _Bf16Impl float2bfloat16(float v) {
        _Bf16Impl ret;
#if __BF16_NATIVE(800, 11000)
        asm ("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(ret._x) : "f"(v));
#else
        const uint32_t &vi = reinterpret_cast<const uint32_t &>(v);
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
#endif
        return ret;
    }

#if __BF16_NATIVE(900, 11080)

    __CUBF16_IMPL _Bf16Impl double2bfloat16(double v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.f64 %0, %1;\n" : "=h"(ret._x) : "d"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl ll2bfloat16(long long v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.s64 %0, %1;\n" : "=h"(ret._x) : "l"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl ull2bfloat16(unsigned long long v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.u64 %0, %1;\n" : "=h"(ret._x) : "l"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl int2bfloat16(int v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.s32 %0, %1;\n" : "=h"(ret._x) : "r"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl uint2bfloat16(unsigned int v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.u32 %0, %1;\n" : "=h"(ret._x) : "r"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl short2bfloat16(short v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.s16 %0, %1;\n" : "=h"(ret._x) : "h"(v));
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl ushort2bfloat16(unsigned short v) {
        _Bf16Impl ret;
        asm ("cvt.rn.bf16.u16 %0, %1;\n" : "=h"(ret._x) : "h"(v));
        return ret;
    }

    __CUBF16_IMPL float bfloat162float(_Bf16Impl v) {
        float ret;
        asm ("cvt.f32.bf16 %0, %1;\n" : "=f"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL double bfloat162double(_Bf16Impl v) {
        double ret;
        asm ("cvt.f64.bf16 %0, %1;\n" : "=d"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL long long bfloat162ll(_Bf16Impl v) {
        long long ret;
        asm ("cvt.rzi.s64.bf16 %0, %1;\n" : "=l"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL unsigned long long bfloat162ull(_Bf16Impl v) {
        unsigned long long ret;
        asm ("cvt.rzi.u64.bf16 %0, %1;\n" : "=l"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL int bfloat162int(_Bf16Impl v) {
        int ret;
        asm ("cvt.rzi.s32.bf16 %0, %1;\n" : "=r"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL unsigned int bfloat162uint(_Bf16Impl v) {
        unsigned int ret;
        asm ("cvt.rzi.u32.bf16 %0, %1;\n" : "=r"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL short bfloat162short(_Bf16Impl v) {
        short ret;
        asm ("cvt.rzi.s16.bf16 %0, %1;\n" : "=h"(ret) : "h"(v._x));
        return ret;
    }

    __CUBF16_IMPL unsigned short bfloat162ushort(_Bf16Impl v) {
        unsigned short ret;
        asm ("cvt.rzi.u16.bf16 %0, %1;\n" : "=h"(ret) : "h"(v._x));
        return ret;
    }

#else

    __CUBF16_IMPL _Bf16Impl double2bfloat16(double v) {
        return float2bfloat16(__double2float_rn(v));
    }

    __CUBF16_IMPL _Bf16Impl ll2bfloat16(long long v) {
        return float2bfloat16(__ll2float_rn(v));
    }

    __CUBF16_IMPL _Bf16Impl ull2bfloat16(unsigned long long v) {
        return float2bfloat16(__ull2float_rn(v));
    }

    __CUBF16_IMPL _Bf16Impl int2bfloat16(int v) {
        return float2bfloat16(__int2float_rn(v));
    }

    __CUBF16_IMPL _Bf16Impl uint2bfloat16(unsigned int v) {
        return float2bfloat16(__uint2float_rn(v));
    }

    __CUBF16_IMPL _Bf16Impl short2bfloat16(short v) {
        return float2bfloat16(static_cast<float>(v));
    }

    __CUBF16_IMPL _Bf16Impl ushort2bfloat16(unsigned short v) {
        return float2bfloat16(static_cast<float>(v));
    }

    __CUBF16_IMPL float bfloat162float(_Bf16Impl v) {
        float ret;
        asm ("mov.b32 %0, {0, %1};\n"
             : "=f"(ret)
             : "h"(v._x)
        );
        return ret;
    }

    __CUBF16_IMPL double bfloat162double(_Bf16Impl v) {
        return static_cast<double>(bfloat162float(v));
    }

    __CUBF16_IMPL long long bfloat162ll(_Bf16Impl v) {
        return static_cast<long long>(bfloat162float(v));
    }

    __CUBF16_IMPL unsigned long long bfloat162ull(_Bf16Impl v) {
        return static_cast<unsigned long long>(bfloat162float(v));
    }

    __CUBF16_IMPL int bfloat162int(_Bf16Impl v) {
        return static_cast<int>(bfloat162float(v));
    }

    __CUBF16_IMPL unsigned int bfloat162uint(_Bf16Impl v) {
        return static_cast<unsigned int>(bfloat162float(v));
    }

    __CUBF16_IMPL short bfloat162short(_Bf16Impl v) {
        return static_cast<short>(bfloat162float(v));
    }

    __CUBF16_IMPL unsigned short bfloat162ushort(_Bf16Impl v) {
        return static_cast<unsigned short>(bfloat162float(v));
    }

#endif

    // + - * /
#if __BF16_NATIVE(800, 11000)
    __CUBF16_IMPL _Bf16Impl bf16add(_Bf16Impl a, _Bf16Impl b) {
        _Bf16Impl ret;
        asm ("{.reg.b16     r;\n"
             " mov.b16 r,   0x3f80U;\n"
             " fma.rn.bf16  %0, %1, r, %2;}\n"
             : "=h"(ret._x)
             : "h"(a._x), "h"(b._x)
        );
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl bf16sub(_Bf16Impl a, _Bf16Impl b) {
        _Bf16Impl ret;
        asm ("{.reg.b16     r;\n"
             " mov.b16      r, 0xbf80U;\n"
             " fma.rn.bf16  %0, %2, r, %1;}\n"
             : "=h"(ret._x)
             : "h"(a._x), "h"(b._x)
        );
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl bf16mul(_Bf16Impl a, _Bf16Impl b) {
        _Bf16Impl ret;
        asm ("{.reg.b16     r;\n"
             " mov.b16      r, 0x0U;\n"
             " fma.rn.bf16  %0, %1, %2, r;}\n"
             : "=h"(ret._x)
             : "h"(a._x), "h"(b._x)
        );
        return ret;
    }

    __CUBF16_IMPL _Bf16Impl bf16div(_Bf16Impl a, _Bf16Impl b) {
        _Bf16Impl ret;
        asm ("{.reg.b32             a, b, c;\n"
             " mov.b32              a, {0, %1};\n"
             " mov.b32              b, {0, %2};\n"
             " div.approx.ftz.f32   c, a, b;\n"
             " cvt.rn.bf16.f32      %0, c;}\n"
             : "=h"(ret._x)
             : "h"(a._x), "h"(b._x)
        );
        return ret;
    }
#else
    __CUBF16_IMPL _Bf16Impl bf16add(_Bf16Impl a, _Bf16Impl b) {
        return float2bfloat16(bfloat162float(a) + bfloat162float(b));
    }

    __CUBF16_IMPL _Bf16Impl bf16sub(_Bf16Impl a, _Bf16Impl b) {
        return float2bfloat16(bfloat162float(a) - bfloat162float(b));
    }

    __CUBF16_IMPL _Bf16Impl bf16mul(_Bf16Impl a, _Bf16Impl b) {
        return float2bfloat16(bfloat162float(a) * bfloat162float(b));
    }

    __CUBF16_IMPL _Bf16Impl bf16div(_Bf16Impl a, _Bf16Impl b) {
        float ret;
        asm ("{.reg.b32             a, b;\n"
             " mov.b32              a, {0, %1};\n"
             " mov.b32              b, {0, %2};\n"
             " div.approx.ftz.f32   %0, a, b;}\n"
             : "=f"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return float2bfloat16(ret);
    }
#endif

    // == != > < >= <=
    __CUBF16_IMPL bool bf16eq(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) == bfloat162float(b);
    }

    __CUBF16_IMPL bool bf16ne(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) != bfloat162float(b);
    }

    __CUBF16_IMPL bool bf16gt(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) > bfloat162float(b);
    }

    __CUBF16_IMPL bool bf16lt(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) < bfloat162float(b);
    }

    __CUBF16_IMPL bool bf16ge(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) >= bfloat162float(b);
    }

    __CUBF16_IMPL bool bf16le(_Bf16Impl a, _Bf16Impl b) {
        return bfloat162float(a) <= bfloat162float(b);
    }
};

#undef __CUBF16_IMPL
#undef __BF16_NATIVE

#endif  // defined(__CUDA_ARCH__)

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_BFLOAT16_CUDABFLOAT16_IMPL_HPP_

