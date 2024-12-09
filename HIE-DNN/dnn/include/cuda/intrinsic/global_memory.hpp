/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    global_memory.hpp
 */

#ifndef DNN_INCLUDE_CUDA_INTRINSIC_GLOBAL_MEMORY_HPP_
#define DNN_INCLUDE_CUDA_INTRINSIC_GLOBAL_MEMORY_HPP_

#include <cstdint>

namespace hiednn {

namespace cuda {

// global memory load/store cache operator
enum GmemCacheOp {
    CACHE_DEFAULT = 0,
    NC,
    CG,
    CS,  // evict-first
};

// global memory load/store cache prefetch
enum GmemPrefetch {
    FETCH_DEFAULT = 0,
    LTC64,
    LTC128,
    LTC256,
};

constexpr GmemPrefetch LTCMAX = LTC256;

namespace intrinsic {

#define _EMPTY_PTX

// ---------------------------------------------------------------------
// LDGIntrinsic: LDG instruction wrapper
// ---------------------------------------------------------------------
template <GmemCacheOp CACHEOP,
          GmemPrefetch PREFETCH,
          int BYTE,
          typename T>
struct LDGIntrinsic;

#define _LDG1_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
template <typename T> \
struct LDGIntrinsic<CACHEOP, PREFETCH, 1, T> { \
    __device__ __forceinline__ \
    LDGIntrinsic(T *x, const void *ptr) { \
        uint32_t reg; \
        asm volatile ( \
            "ld.global"#CACHEOP_PTX#PREFETCH_PTX".b8 %0, [%1];" \
            : "=r"(reg) : "l"(ptr) \
        ); \
        reinterpret_cast<uint8_t &>(*x) = static_cast<uint8_t>(reg); \
    } \
};

#define _LDG2_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
template <typename T> \
struct LDGIntrinsic<CACHEOP, PREFETCH, 2, T> { \
    __device__ __forceinline__ \
    LDGIntrinsic(T *x, const void *ptr) { \
        uint32_t reg; \
        asm volatile ( \
            "ld.global"#CACHEOP_PTX#PREFETCH_PTX".b16 %0, [%1];" \
            : "=r"(reg) : "l"(ptr) \
        ); \
        reinterpret_cast<uint16_t &>(*x) = static_cast<uint16_t>(reg); \
    } \
};

#define _LDG4_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
template <typename T> \
struct LDGIntrinsic<CACHEOP, PREFETCH, 4, T> { \
    __device__ __forceinline__ \
    LDGIntrinsic(T *x, const void *ptr) { \
        asm volatile ( \
            "ld.global"#CACHEOP_PTX#PREFETCH_PTX".b32 %0, [%1];" \
            : "=r"(reinterpret_cast<uint32_t &>(*x)) \
            : "l"(ptr) \
        ); \
    } \
};

#define _LDG8_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
template <typename T> \
struct LDGIntrinsic<CACHEOP, PREFETCH, 8, T> { \
    __device__ __forceinline__ \
    LDGIntrinsic(T *x, const void *ptr) { \
        asm volatile ( \
            "ld.global"#CACHEOP_PTX#PREFETCH_PTX".v2.b32 {%0, %1}, [%2];" \
            : "=r"(reinterpret_cast<uint2 &>(*x).x) \
              "=r"(reinterpret_cast<uint2 &>(*x).y) \
            : "l"(ptr) \
        ); \
    } \
};

#define _LDG16_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
template <typename T> \
struct LDGIntrinsic<CACHEOP, PREFETCH, 16, T> { \
    __device__ __forceinline__ \
    LDGIntrinsic(T *x, const void *ptr) { \
        asm volatile ( \
            "ld.global"#CACHEOP_PTX#PREFETCH_PTX".v4.b32 " \
            "{%0, %1, %2, %3}, [%4];" \
            : "=r"(reinterpret_cast<uint4 &>(*x).x) \
              "=r"(reinterpret_cast<uint4 &>(*x).y) \
              "=r"(reinterpret_cast<uint4 &>(*x).z) \
              "=r"(reinterpret_cast<uint4 &>(*x).w) \
            : "l"(ptr) \
        ); \
    } \
};

#define _LDG_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
    _LDG1_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
    _LDG2_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
    _LDG4_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
    _LDG8_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX) \
    _LDG16_INTRINSIC(CACHEOP, CACHEOP_PTX, PREFETCH, PREFETCH_PTX)

// only support sm_32+ devices
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 320)
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, FETCH_DEFAULT, _EMPTY_PTX)
_LDG_INTRINSIC(NC, .nc, FETCH_DEFAULT, _EMPTY_PTX)
_LDG_INTRINSIC(CG, .cg, FETCH_DEFAULT, _EMPTY_PTX)
_LDG_INTRINSIC(CS, .cs, FETCH_DEFAULT, _EMPTY_PTX)

#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040)

_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC64, .L2::64B)
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC128, .L2::128B)
_LDG_INTRINSIC(NC, .nc, LTC64, .L2::64B)
_LDG_INTRINSIC(NC, .nc, LTC128, .L2::128B)
_LDG_INTRINSIC(CG, .cg, LTC64, .L2::64B)
_LDG_INTRINSIC(CG, .cg, LTC128, .L2::128B)
_LDG_INTRINSIC(CS, .cs, LTC64, .L2::64B)
_LDG_INTRINSIC(CS, .cs, LTC128, .L2::128B)
#if __CUDA_ARCH__ >= 800
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC256, .L2::256B)
_LDG_INTRINSIC(NC, .nc, LTC256, .L2::256B)
_LDG_INTRINSIC(CG, .cg, LTC256, .L2::256B)
_LDG_INTRINSIC(CS, .cs, LTC256, .L2::256B)
#else  // __CUDA_ARCH__ < 800
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC256, .L2::128B)
_LDG_INTRINSIC(NC, .nc, LTC256, .L2::128B)
_LDG_INTRINSIC(CG, .cg, LTC256, .L2::128B)
_LDG_INTRINSIC(CS, .cs, LTC256, .L2::128B)
#endif  // __CUDA_ARCH__ >= 800

#else  // __CUDA_ARCH__ < 750 || CUDA_VERSION < 11040
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC64, _EMPTY_PTX)
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC128, _EMPTY_PTX)
_LDG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX, LTC256, _EMPTY_PTX)
_LDG_INTRINSIC(NC, .nc, LTC64, _EMPTY_PTX)
_LDG_INTRINSIC(NC, .nc, LTC128, _EMPTY_PTX)
_LDG_INTRINSIC(NC, .nc, LTC256, _EMPTY_PTX)
_LDG_INTRINSIC(CG, .cg, LTC64, _EMPTY_PTX)
_LDG_INTRINSIC(CG, .cg, LTC128, _EMPTY_PTX)
_LDG_INTRINSIC(CG, .cg, LTC256, _EMPTY_PTX)
_LDG_INTRINSIC(CS, .cs, LTC64, _EMPTY_PTX)
_LDG_INTRINSIC(CS, .cs, LTC128, _EMPTY_PTX)
_LDG_INTRINSIC(CS, .cs, LTC256, _EMPTY_PTX)
#endif  // __CUDA_ARCH__ >= 750

#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 320)

#undef _LDG1_INTRINSIC
#undef _LDG2_INTRINSIC
#undef _LDG4_INTRINSIC
#undef _LDG8_INTRINSIC
#undef _LDG16_INTRINSIC
#undef _LDG_INTRINSIC

// ---------------------------------------------------------------------
// STGIntrinsic: STG instruction wrapper
// ---------------------------------------------------------------------
template <GmemCacheOp CACHEOP,
          int BYTE,
          typename T>
struct STGIntrinsic;

#define _STG1_INTRINSIC(CACHEOP, CACHEOP_PTX) \
template <typename T> \
struct STGIntrinsic<CACHEOP, 1, T> { \
    __device__ __forceinline__ \
    STGIntrinsic(const T &x, void *ptr) { \
        uint32_t reg = reinterpret_cast<const uint8_t &>(x); \
        asm volatile ( \
            "st.global"#CACHEOP_PTX".b8 [%0], %1;" \
            : : "l"(ptr), "r"(reg) \
        ); \
    } \
};

#define _STG2_INTRINSIC(CACHEOP, CACHEOP_PTX) \
template <typename T> \
struct STGIntrinsic<CACHEOP, 2, T> { \
    __device__ __forceinline__ \
    STGIntrinsic(const T &x, void *ptr) { \
        uint32_t reg = reinterpret_cast<const uint16_t &>(x); \
        asm volatile ( \
            "st.global"#CACHEOP_PTX".b16 [%0], %1;" \
            : : "l"(ptr), "r"(reg) \
        ); \
    } \
};

#define _STG4_INTRINSIC(CACHEOP, CACHEOP_PTX) \
template <typename T> \
struct STGIntrinsic<CACHEOP, 4, T> { \
    __device__ __forceinline__ \
    STGIntrinsic(const T &x, void *ptr) { \
        asm volatile ( \
            "st.global"#CACHEOP_PTX".b32 [%0], %1;" \
            : \
            : "l"(ptr), \
              "r"(reinterpret_cast<const uint32_t &>(x)) \
        ); \
    } \
};

#define _STG8_INTRINSIC(CACHEOP, CACHEOP_PTX) \
template <typename T> \
struct STGIntrinsic<CACHEOP, 8, T> { \
    __device__ __forceinline__ \
    STGIntrinsic(const T &x, void *ptr) { \
        asm volatile ( \
            "st.global"#CACHEOP_PTX".v2.b32 [%0], {%1, %2};" \
            : \
            : "l"(ptr), \
              "r"(reinterpret_cast<const uint2 &>(x).x), \
              "r"(reinterpret_cast<const uint2 &>(x).y) \
        ); \
    } \
};

#define _STG16_INTRINSIC(CACHEOP, CACHEOP_PTX) \
template <typename T> \
struct STGIntrinsic<CACHEOP, 16, T> { \
    __device__ __forceinline__ \
    STGIntrinsic(const T &x, void *ptr) { \
        asm volatile ( \
            "st.global"#CACHEOP_PTX".v4.b32 [%0], {%1, %2, %3, %4};" \
            : \
            : "l"(ptr), \
              "r"(reinterpret_cast<const uint4 &>(x).x), \
              "r"(reinterpret_cast<const uint4 &>(x).y), \
              "r"(reinterpret_cast<const uint4 &>(x).z), \
              "r"(reinterpret_cast<const uint4 &>(x).w) \
        ); \
    } \
};

#define _STG_INTRINSIC(CACHEOP, CACHEOP_PTX) \
    _STG1_INTRINSIC(CACHEOP, CACHEOP_PTX) \
    _STG2_INTRINSIC(CACHEOP, CACHEOP_PTX) \
    _STG4_INTRINSIC(CACHEOP, CACHEOP_PTX) \
    _STG8_INTRINSIC(CACHEOP, CACHEOP_PTX) \
    _STG16_INTRINSIC(CACHEOP, CACHEOP_PTX) \

_STG_INTRINSIC(CACHE_DEFAULT, _EMPTY_PTX)
_STG_INTRINSIC(CG, .cg)
_STG_INTRINSIC(CS, .cs)

#undef _STG1_INTRINSIC
#undef _STG2_INTRINSIC
#undef _STG4_INTRINSIC
#undef _STG8_INTRINSIC
#undef _STG16_INTRINSIC
#undef _STG_INTRINSIC

#undef _EMPTY_PTX

}  // namespace intrinsic

// ldg_cg for all types (1-, 2-, 4-, 8-, 16-byte types)
template <GmemCacheOp CACHEOP = CACHE_DEFAULT,
          GmemPrefetch PREFETCH = FETCH_DEFAULT,
          typename T>
__device__ __forceinline__ void Ldg(T *x, const void *ptr) {
    static_assert(sizeof(T) == 1 ||
                  sizeof(T) == 2 ||
                  sizeof(T) == 4 ||
                  sizeof(T) == 8 ||
                  sizeof(T) == 16,
                  "ldg: invalid typename T");

    intrinsic::LDGIntrinsic<CACHEOP, PREFETCH, sizeof(T), T>(x, ptr);
}

template <GmemCacheOp CACHEOP = CACHE_DEFAULT,
          typename T>
__device__ __forceinline__ void Stg(const T &x, void *ptr) {
    static_assert(sizeof(T) == 1 ||
                  sizeof(T) == 2 ||
                  sizeof(T) == 4 ||
                  sizeof(T) == 8 ||
                  sizeof(T) == 16,
                  "stg: invalid typename T");

    intrinsic::STGIntrinsic<CACHEOP, sizeof(T), T>(x, ptr);
}

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_INTRINSIC_GLOBAL_MEMORY_HPP_


