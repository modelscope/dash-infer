/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cudahalf_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_IMPL_HPP_

#include <cuda.h>
#include <cstdint>

namespace __hiednn_buildin {

#if defined(__CUDA_ARCH__)

#define __HALF_IMPL

#define __F16_DEVICE_FUNC __device__ __forceinline__
#define __CUF16_IMPL static __device__ __forceinline__
#define __FP16_NATIVE (__CUDA_ARCH__ >= 530) && (CUDA_VERSION >= 7000)

struct __attribute__((aligned(2))) _HalfImpl {
    uint16_t _x;

    __F16_DEVICE_FUNC _HalfImpl() {}

    __CUF16_IMPL _HalfImpl float2half(float v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.f32 %0, %1;\n" : "=h"(ret._x) : "f"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl double2half(double v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.f64 %0, %1;\n" : "=h"(ret._x) : "d"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl ll2half(long long v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.s64 %0, %1;\n" : "=h"(ret._x) : "l"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl ull2half(unsigned long long v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.u64 %0, %1;\n" : "=h"(ret._x) : "l"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl int2half(int v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.s32 %0, %1;\n" : "=h"(ret._x) : "r"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl uint2half(unsigned int v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.u32 %0, %1;\n" : "=h"(ret._x) : "r"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl short2half(short v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.s16 %0, %1;\n" : "=h"(ret._x) : "h"(v));
        return ret;
    }

    __CUF16_IMPL _HalfImpl ushort2half(unsigned short v) {
        _HalfImpl ret;
        asm ("cvt.rn.f16.u16 %0, %1;\n" : "=h"(ret._x) : "h"(v));
        return ret;
    }

    __CUF16_IMPL float half2float(_HalfImpl v) {
        float ret;
        asm ("cvt.f32.f16 %0, %1;\n" : "=f"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL double half2double(_HalfImpl v) {
        double ret;
        asm ("cvt.f64.f16 %0, %1;\n" : "=d"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL long long half2ll(_HalfImpl v) {
        long long ret;
        asm ("cvt.rzi.s64.f16 %0, %1;\n" : "=l"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL unsigned long long half2ull(_HalfImpl v) {
        unsigned long long ret;
        asm ("cvt.rzi.u64.f16 %0, %1;\n" : "=l"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL int half2int(_HalfImpl v) {
        int ret;
        asm ("cvt.rzi.s32.f16 %0, %1;\n" : "=r"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL unsigned int half2uint(_HalfImpl v) {
        unsigned int ret;
        asm ("cvt.rzi.u32.f16 %0, %1;\n" : "=r"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL short half2short(_HalfImpl v) {
        short ret;
        asm ("cvt.rzi.s16.f16 %0, %1;\n" : "=h"(ret) : "h"(v._x));
        return ret;
    }

    __CUF16_IMPL unsigned short half2ushort(_HalfImpl v) {
        unsigned short ret;
        asm ("cvt.rzi.u16.f16 %0, %1;\n" : "=h"(ret) : "h"(v._x));
        return ret;
    }

    // + - * /
#if __FP16_NATIVE
    __CUF16_IMPL _HalfImpl hadd(_HalfImpl a, _HalfImpl b) {
        _HalfImpl ret;
        asm ("add.rn.f16 %0, %1, %2;\n" : "=h"(ret._x) : "h"(a._x), "h"(b._x));
        return ret;
    }

    __CUF16_IMPL _HalfImpl hsub(_HalfImpl a, _HalfImpl b) {
        _HalfImpl ret;
        asm ("sub.rn.f16 %0, %1, %2;\n" : "=h"(ret._x) : "h"(a._x), "h"(b._x));
        return ret;
    }

    __CUF16_IMPL _HalfImpl hmul(_HalfImpl a, _HalfImpl b) {
        _HalfImpl ret;
        asm ("mul.rn.f16 %0, %1, %2;\n" : "=h"(ret._x) : "h"(a._x), "h"(b._x));
        return ret;
    }

    __CUF16_IMPL _HalfImpl hdiv(_HalfImpl a, _HalfImpl b) {
        _HalfImpl ret;
        asm ("{.reg.b32             af, bf, rcp, err, ret;\n"
             " .reg.b16             den, ret_abs;\n"
             " .reg.pred            p;\n"
             " mov.b16              den, 0x008fU;\n"
             " cvt.f32.f16          af, %1;\n"
             " cvt.f32.f16          bf, %2;\n"
             " rcp.approx.ftz.f32   rcp, bf;\n"
             " mul.f32              ret, af, rcp;\n"
             " cvt.rn.f16.f32       %0, ret;\n"
             // error fix for abs(ret) < den
             " abs.f16              ret_abs, %0;\n"
             " setp.lt.f16          p, ret_abs, den;\n"
             " setp.ne.and.b16      p, ret_abs, 0x0U, p;\n"
             " @p neg.f32           bf, bf;\n"
             " @p fma.rn.f32        err, ret, bf, af;\n"
             " @p fma.rn.f32        ret, err, rcp, ret;\n"
             " @p cvt.rn.f16.f32    %0, ret;}\n"
             : "=h"(ret._x)
             : "h"(a._x), "h"(b._x)
        );
        return ret;
    }
#else
    __CUF16_IMPL _HalfImpl hadd(_HalfImpl a, _HalfImpl b) {
        return float2half(half2float(a) + half2float(b));
    }

    __CUF16_IMPL _HalfImpl hsub(_HalfImpl a, _HalfImpl b) {
        return float2half(half2float(a) - half2float(b));
    }

    __CUF16_IMPL _HalfImpl hmul(_HalfImpl a, _HalfImpl b) {
        return float2half(half2float(a) * half2float(b));
    }

    __CUF16_IMPL _HalfImpl hdiv(_HalfImpl a, _HalfImpl b) {
        return float2half(half2float(a) / half2float(b));
    }
#endif

    // == != > < >= <=
#if __FP16_NATIVE
    __CUF16_IMPL bool heq(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.eq.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }

    __CUF16_IMPL bool hne(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.ne.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }

    __CUF16_IMPL bool hgt(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.gt.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }

    __CUF16_IMPL bool hlt(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.lt.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }

    __CUF16_IMPL bool hge(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.ge.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }

    __CUF16_IMPL bool hle(_HalfImpl a, _HalfImpl b) {
        uint16_t ret;
        asm ("{.reg.pred    p;\n"
             " setp.le.f16  p, %1, %2;\n"
             " selp.u16     %0, 1, 0, p;}\n"
             : "=h"(ret)
             : "h"(a._x), "h"(b._x)
        );
        return ret != 0;
    }
#else
    __CUF16_IMPL bool heq(_HalfImpl a, _HalfImpl b) {
        return half2float(a) == half2float(b);
    }

    __CUF16_IMPL bool hne(_HalfImpl a, _HalfImpl b) {
        return half2float(a) != half2float(b);
    }

    __CUF16_IMPL bool hgt(_HalfImpl a, _HalfImpl b) {
        return half2float(a) > half2float(b);
    }

    __CUF16_IMPL bool hlt(_HalfImpl a, _HalfImpl b) {
        return half2float(a) < half2float(b);
    }

    __CUF16_IMPL bool hge(_HalfImpl a, _HalfImpl b) {
        return half2float(a) >= half2float(b);
    }

    __CUF16_IMPL bool hle(_HalfImpl a, _HalfImpl b) {
        return half2float(a) <= half2float(b);
    }
#endif
};

#undef __CUF16_IMPL
#undef __FP16_NATIVE

#endif  // defined(__CUDA_ARCH__)

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CUDAHALF_IMPL_HPP_


