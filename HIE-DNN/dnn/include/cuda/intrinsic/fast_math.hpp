/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fast_math.hpp
 */

#ifndef DNN_INCLUDE_CUDA_INTRINSIC_FAST_MATH_HPP_
#define DNN_INCLUDE_CUDA_INTRINSIC_FAST_MATH_HPP_

#include <cstdint>
#include <datatype_extension/datatype_extension.hpp>
#include <cuda/cuda_utils.hpp>

namespace hiednn {

namespace cuda {

namespace FastMath {

//---------------------------------------------
// floating point reciprocal (ftz)
//---------------------------------------------
__device__ __forceinline__
float FRcp(const float &x) {
    float ret;
    asm ("rcp.approx.ftz.f32 %0, %1;"
         : "=f"(ret) : "f"(x)
    );
    return ret;
}

__device__ __forceinline__
double FRcp(const double &x) {
    double ret;
    asm ("rcp.approx.ftz.f64 %0, %1;\n"
         : "=d"(ret) : "d"(x)
    );
    return ret;
}

#ifdef HIEDNN_USE_FP16
__device__ __forceinline__
half FRcp(const half &x) {
    return 1 / x;
}
#endif

#ifdef HIEDNN_USE_BF16
__device__ __forceinline__
bfloat16 FRcp(const bfloat16 &x) {
    float ret = FRcp(static_cast<float>(x));
    return static_cast<bfloat16>(ret);
}
#endif

//---------------------------------------------
// floating point division (ftz)
//---------------------------------------------
__device__ __forceinline__
float FDiv(const float &x, const float &y) {
    float ret;
    asm ("div.approx.ftz.f32 %0, %1, %2;"
         : "=f"(ret) : "f"(x), "f"(y)
    );
    return ret;
}

__device__ __forceinline__
double FDiv(const double &x, const double &y) {
    double ret;
    asm (".reg .f64 rcp;\n"
         "rcp.approx.ftz.f64 rcp, %2;\n"
         "mul.f64 %0, %1, rcp;\n"
         : "=d"(ret) : "d"(x), "d"(y)
    );
    return ret;
}

#ifdef HIEDNN_USE_FP16
__device__ __forceinline__
half FDiv(const half &x, const half &y) {
    return x / y;
}
#endif

#ifdef HIEDNN_USE_BF16
__device__ __forceinline__
bfloat16 FDiv(const bfloat16 &x, const bfloat16 &y) {
    float ret = FDiv(static_cast<float>(x), static_cast<float>(y));
    return static_cast<bfloat16>(ret);
}
#endif

}  // namespace FastMath

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_INTRINSIC_FAST_MATH_HPP_


