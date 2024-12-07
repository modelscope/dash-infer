/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    warp_shuffle.hpp
 */

#ifndef DNN_INCLUDE_CUDA_INTRINSIC_WARP_SHUFFLE_HPP_
#define DNN_INCLUDE_CUDA_INTRINSIC_WARP_SHUFFLE_HPP_

#include <cuda.h>
#include <cstdint>
#include <utils.hpp>
#include <cuda/cuda_utils.hpp>

namespace hiednn {

namespace cuda {

namespace intrinsic {

template <int SIZE>
struct ShflUIntT {
    using type = uint32_t;
};

template <>
struct ShflUIntT<1> {
    using type = uint8_t;
};

template <>
struct ShflUIntT<2> {
    using type = uint16_t;
};

__device__ __forceinline__
uint32_t ShflIdx(uint32_t mask,
                 uint32_t var,
                 uint32_t srcLane,
                 uint32_t width) {
    const uint32_t SHFL_C = ((WARP_SIZE - width) << 8) | (WARP_SIZE - 1);

    uint32_t ret;
    asm volatile (
#if CUDA_VERSION < 9000
        "shfl.idx.b32 %0, %1, %2, %3;\n"
#else
        "shfl.sync.idx.b32 %0, %1, %2, %3, %4;\n"
#endif
        : "=r"(ret) : "r"(var), "r"(srcLane), "r"(SHFL_C), "r"(mask)
    );
    return ret;
}

__device__ __forceinline__
uint32_t ShflUp(uint32_t mask,
                uint32_t var,
                uint32_t delta,
                uint32_t width) {
    const uint32_t SHFL_C = (WARP_SIZE - width) << 8;

    uint32_t ret;
    asm volatile (
#if CUDA_VERSION < 9000
        "shfl.up.b32 %0, %1, %2, %3;\n"
#else
        "shfl.sync.up.b32 %0, %1, %2, %3, %4;\n"
#endif
        : "=r"(ret) : "r"(var), "r"(delta), "r"(SHFL_C), "r"(mask)
    );
    return ret;
}

__device__ __forceinline__
uint32_t ShflDown(uint32_t mask,
                  uint32_t var,
                  uint32_t delta,
                  uint32_t width) {
    const uint32_t SHFL_C = ((WARP_SIZE - width) << 8) | (WARP_SIZE - 1);

    uint32_t ret;
    asm volatile (
#if CUDA_VERSION < 9000
        "shfl.down.b32 %0, %1, %2, %3;\n"
#else
        "shfl.sync.down.b32 %0, %1, %2, %3, %4;\n"
#endif
        : "=r"(ret) : "r"(var), "r"(delta), "r"(SHFL_C), "r"(mask)
    );
    return ret;
}

__device__ __forceinline__
uint32_t ShflBfly(uint32_t mask,
                  uint32_t var,
                  uint32_t laneMask,
                  uint32_t width) {
    const uint32_t SHFL_C = ((WARP_SIZE - width) << 8) | (WARP_SIZE - 1);

    uint32_t ret;
    asm volatile (
#if CUDA_VERSION < 9000
        "shfl.bfly.b32 %0, %1, %2, %3;\n"
#else
        "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;\n"
#endif
        : "=r"(ret) : "r"(var), "r"(laneMask), "r"(SHFL_C), "r"(mask)
    );
    return ret;
}

}  // namespace intrinsic

template <typename T>
__device__ __forceinline__
T ShflIdx(uint32_t mask, const T &var, uint32_t srcLane, uint32_t width) {
    static_assert(IsPowOf2(sizeof(T)), "sizeof(T) must be power of 2.");

    T ret;
    using ShflT = typename intrinsic::ShflUIntT<sizeof(T)>::type;
    const ShflT *x = reinterpret_cast<const ShflT *>(&var);
    ShflT *y = reinterpret_cast<ShflT *>(&ret);

    #pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(ShflT); ++i) {
        y[i] = intrinsic::ShflIdx(
            mask, static_cast<uint32_t>(x[i]), srcLane, width);
    }

    return ret;
}

template <typename T>
__device__ __forceinline__
T ShflUp(uint32_t mask, const T &var, uint32_t delta, uint32_t width) {
    static_assert(IsPowOf2(sizeof(T)), "sizeof(T) must be power of 2.");

    T ret;
    using ShflT = typename intrinsic::ShflUIntT<sizeof(T)>::type;
    const ShflT *x = reinterpret_cast<const ShflT *>(&var);
    ShflT *y = reinterpret_cast<ShflT *>(&ret);

    #pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(ShflT); ++i) {
        y[i] = intrinsic::ShflUp(
            mask, static_cast<uint32_t>(x[i]), delta, width);
    }

    return ret;
}

template <typename T>
__device__ __forceinline__
T ShflDown(uint32_t mask, const T &var, uint32_t delta, uint32_t width) {
    static_assert(IsPowOf2(sizeof(T)), "sizeof(T) must be power of 2.");

    T ret;
    using ShflT = typename intrinsic::ShflUIntT<sizeof(T)>::type;
    const ShflT *x = reinterpret_cast<const ShflT *>(&var);
    ShflT *y = reinterpret_cast<ShflT *>(&ret);

    #pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(ShflT); ++i) {
        y[i] = intrinsic::ShflDown(
            mask, static_cast<uint32_t>(x[i]), delta, width);
    }

    return ret;
}

template <typename T>
__device__ __forceinline__
T ShflBfly(uint32_t mask, const T &var, uint32_t laneMask, uint32_t width) {
    static_assert(IsPowOf2(sizeof(T)), "sizeof(T) must be power of 2.");

    T ret;
    using ShflT = typename intrinsic::ShflUIntT<sizeof(T)>::type;
    const ShflT *x = reinterpret_cast<const ShflT *>(&var);
    ShflT *y = reinterpret_cast<ShflT *>(&ret);

    #pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(ShflT); ++i) {
        y[i] = intrinsic::ShflBfly(
            mask, static_cast<uint32_t>(x[i]), laneMask, width);
    }

    return ret;
}

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_INTRINSIC_WARP_SHUFFLE_HPP_


