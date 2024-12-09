/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpphalf_impl.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_IMPL_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_IMPL_HPP_

#include <half.hpp>  // 3rd party half stdcpp impl
#if defined(__CUDACC__)
#include "cudahalf_impl.hpp"
#endif

namespace __hiednn_buildin {

#ifndef __HALF_IMPL
#define __HALF_IMPL

#define __F16_DEVICE_FUNC inline

struct __attribute__((aligned(2))) _HalfImpl {
    half_float::half _x;

    _HalfImpl() { _x = 0.f; }

    explicit _HalfImpl(half_float::half v) { _x = v; }

    static _HalfImpl float2half(float v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl double2half(double v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl ll2half(long long v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl ull2half(unsigned long long v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl int2half(int v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl uint2half(unsigned int v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl short2half(short v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static _HalfImpl ushort2half(unsigned short v) {
        return _HalfImpl(static_cast<half_float::half>(v));
    }

    static float half2float(_HalfImpl v) {
        return static_cast<float>(v._x);
    }

    static double half2double(_HalfImpl v) {
        return static_cast<double>(v._x);
    }

    static long long half2ll(_HalfImpl v) {
        return static_cast<long long>(v._x);
    }

    static unsigned long long half2ull(_HalfImpl v) {
        return static_cast<unsigned long long>(v._x);
    }

    static int half2int(_HalfImpl v) {
        return static_cast<int>(v._x);
    }

    static unsigned int half2uint(_HalfImpl v) {
        return static_cast<unsigned int>(v._x);
    }

    static short half2short(_HalfImpl v) {
        return static_cast<short>(v._x);
    }

    static unsigned short half2ushort(_HalfImpl v) {
        return static_cast<unsigned short>(v._x);
    }

    // + - * /
    static _HalfImpl hadd(_HalfImpl a, _HalfImpl b) {
        return _HalfImpl(a._x + b._x);
    }

    static _HalfImpl hsub(_HalfImpl a, _HalfImpl b) {
        return _HalfImpl(a._x - b._x);
    }

    static _HalfImpl hmul(_HalfImpl a, _HalfImpl b) {
        return _HalfImpl(a._x * b._x);
    }

    static _HalfImpl hdiv(_HalfImpl a, _HalfImpl b) {
        return _HalfImpl(a._x / b._x);
    }

    // == != > < >= <=
    static bool heq(_HalfImpl a, _HalfImpl b) {
        return a._x == b._x;
    }

    static bool hne(_HalfImpl a, _HalfImpl b) {
        return !(a._x == b._x);
    }

    static bool hgt(_HalfImpl a, _HalfImpl b) {
        return a._x > b._x;
    }

    static bool hlt(_HalfImpl a, _HalfImpl b) {
        return a._x < b._x;
    }

    static bool hge(_HalfImpl a, _HalfImpl b) {
        return a._x >= b._x;
    }

    static bool hle(_HalfImpl a, _HalfImpl b) {
        return a._x <= b._x;
    }
};

#endif  // __HALF_IMPL

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_HALF_CPPHALF_IMPL_HPP_
