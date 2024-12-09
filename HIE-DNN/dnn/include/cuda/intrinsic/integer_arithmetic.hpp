/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    integer_arithmetic.hpp
 */

#ifndef DNN_INCLUDE_CUDA_INTRINSIC_INTEGER_ARITHMETIC_HPP_
#define DNN_INCLUDE_CUDA_INTRINSIC_INTEGER_ARITHMETIC_HPP_

#include <cstdint>

namespace hiednn {

namespace cuda {

//----------------------------------------
// Popc: count the number of one bits
//----------------------------------------
namespace intrinsic {

__device__ __forceinline__
int Popc32(const uint32_t &x) {
    int ret;
    asm("popc.b32 %0, %1;" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ __forceinline__
int Popc64(const uint64_t &x) {
    int ret;
    asm("popc.b64 %0, %1;" : "=r"(ret) : "l"(x));
    return ret;
}

template <int S, typename T>
struct PopcImpl;

template <typename T>
struct PopcImpl<1, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint8_t &>(x);
        return Popc32(xi);
    }
};

template <typename T>
struct PopcImpl<2, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint16_t &>(x);
        return Popc32(xi);
    }
};

template <typename T>
struct PopcImpl<4, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Popc32(reinterpret_cast<const uint32_t &>(x));
    }
};

template <typename T>
struct PopcImpl<8, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Popc64(reinterpret_cast<const uint64_t &>(x));
    }
};

}  // namespace intrinsic

template <typename T>
__device__ __forceinline__
int Popc(const T &x) {
    static_assert(sizeof(T) == 1 ||
                  sizeof(T) == 2 ||
                  sizeof(T) == 4 ||
                  sizeof(T) == 8,
                  "Popc: invalid typename T");
    return intrinsic::PopcImpl<sizeof(T), T>()(x);
}

//----------------------------------------
// Clz: count the number of leading zeros from the most-significant bit
//----------------------------------------
namespace intrinsic {

__device__ __forceinline__
int Clz32(const uint32_t &x) {
    int ret;
    asm("clz.b32 %0, %1;" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ __forceinline__
int Clz64(const uint64_t &x) {
    int ret;
    asm("clz.b64 %0, %1;" : "=r"(ret) : "l"(x));
    return ret;
}

template <int S, typename T>
struct ClzImpl;

template <typename T>
struct ClzImpl<1, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint8_t &>(x);
        return Clz32(xi);
    }
};

template <typename T>
struct ClzImpl<2, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint16_t &>(x);
        return Clz32(xi);
    }
};

template <typename T>
struct ClzImpl<4, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Clz32(reinterpret_cast<const uint32_t &>(x));
    }
};

template <typename T>
struct ClzImpl<8, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Clz64(reinterpret_cast<const uint64_t &>(x));
    }
};

}  // namespace intrinsic

template <typename T>
__device__ __forceinline__
int Clz(const T &x) {
    static_assert(sizeof(T) == 1 ||
                  sizeof(T) == 2 ||
                  sizeof(T) == 4 ||
                  sizeof(T) == 8,
                  "Clz: invalid typename T");
    return intrinsic::ClzImpl<sizeof(T), T>()(x);
}

//----------------------------------------
// Ffs: find the position of the least significant bit set to 1
//----------------------------------------
namespace intrinsic {

__device__ __forceinline__
int Ffs32(const uint32_t &x) {
    int ret;
    asm("{.reg.b32 r;\n"
        " brev.b32 r, %1;\n"
        " bfind.shiftamt.u32 r, r;\n"
        " add.u32 %0, r, 1;}\n"
        : "=r"(ret) : "r"(x)
    );
    return ret;
}

__device__ __forceinline__
int Ffs64(const uint64_t &x) {
    int ret;
    asm("{.reg.b64 r0;\n"
        " .reg.b32 r1"
        " brev.b64 r0, %1;\n"
        " bfind.shiftamt.u64 r1, r0;\n"
        " add.u32 %0, r1, 1;}\n"
        : "=r"(ret) : "l"(x)
    );
    return ret;
}

template <int S, typename T>
struct FfsImpl;

template <typename T>
struct FfsImpl<1, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint8_t &>(x);
        return Ffs32(xi);
    }
};

template <typename T>
struct FfsImpl<2, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        uint32_t xi = reinterpret_cast<const uint16_t &>(x);
        return Ffs32(xi);
    }
};

template <typename T>
struct FfsImpl<4, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Ffs32(reinterpret_cast<const uint32_t &>(x));
    }
};

template <typename T>
struct FfsImpl<8, T> {
    __device__ __forceinline__ int operator()(const T &x) {
        return Ffs64(reinterpret_cast<const uint64_t &>(x));
    }
};

}  // namespace intrinsic

template <typename T>
__device__ __forceinline__
int Ffs(const T &x) {
    static_assert(sizeof(T) == 1 ||
                  sizeof(T) == 2 ||
                  sizeof(T) == 4 ||
                  sizeof(T) == 8,
                  "Clz: invalid typename T");
    return intrinsic::FfsImpl<sizeof(T), T>()(x);
}

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_INTRINSIC_INTEGER_ARITHMETIC_HPP_

