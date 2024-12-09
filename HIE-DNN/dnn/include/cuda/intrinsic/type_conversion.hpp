/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    type_conversion.hpp
 */

#ifndef DNN_INCLUDE_CUDA_INTRINSIC_TYPE_CONVERSION_HPP_
#define DNN_INCLUDE_CUDA_INTRINSIC_TYPE_CONVERSION_HPP_

#include <cstdint>
#include <datatype_extension/datatype_extension.hpp>

namespace hiednn {

namespace cuda {

// floating point round and convert to integer

// round to negative infinity
template <typename D, typename S>
__device__ __forceinline__
D F2I_RD(const S &s);

// round to positive infinity
template <typename D, typename S>
__device__ __forceinline__
D F2I_RU(const S &s);

// round to nearest
template <typename D, typename S>
__device__ __forceinline__
D F2I_RN(const S &s);

// round to zero
template <typename D, typename S>
__device__ __forceinline__
D F2I_RZ(const S &s);

// --------------------------------------
// float to integer
// --------------------------------------
#define __FLOAT_INT(RND, RND_PTX) \
template <> \
__device__ __forceinline__ \
int8_t F2I_##RND<int8_t, float>(const float &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s8.f32 %0, %1;" \
        : "=r"(d) : "f"(s) \
    ); \
    return static_cast<int8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
uint8_t F2I_##RND<uint8_t, float>(const float &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u8.f32 %0, %1;" \
        : "=r"(d) : "f"(s) \
    ); \
    return static_cast<uint8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
int16_t F2I_##RND<int16_t, float>(const float &s) { \
    int16_t d; \
    asm ( \
        "cvt."#RND_PTX".s16.f32 %0, %1;" \
        : "=h"(d) : "f"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint16_t F2I_##RND<uint16_t, float>(const float &s) { \
    uint16_t d; \
    asm ( \
        "cvt."#RND_PTX".u16.f32 %0, %1;" \
        : "=h"(d) : "f"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int32_t F2I_##RND<int32_t, float>(const float &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s32.f32 %0, %1;" \
        : "=r"(d) : "f"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint32_t F2I_##RND<uint32_t, float>(const float &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u32.f32 %0, %1;" \
        : "=r"(d) : "f"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int64_t F2I_##RND<int64_t, float>(const float &s) { \
    int64_t d; \
    asm ( \
        "cvt."#RND_PTX".s64.f32 %0, %1;" \
        : "=l"(d) : "f"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint64_t F2I_##RND<uint64_t, float>(const float &s) { \
    uint64_t d; \
    asm ( \
        "cvt."#RND_PTX".u64.f32 %0, %1;" \
        : "=l"(d) : "f"(s) \
    ); \
    return d; \
}

__FLOAT_INT(RD, rmi)
__FLOAT_INT(RU, rpi)
__FLOAT_INT(RN, rni)
__FLOAT_INT(RZ, rzi)
#undef __FLOAT_INT

// --------------------------------------
// double to integer
// --------------------------------------
#define __DOUBLE_INT(RND, RND_PTX) \
template <> \
__device__ __forceinline__ \
int8_t F2I_##RND<int8_t, double>(const double &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s8.f64 %0, %1;" \
        : "=r"(d) : "d"(s) \
    ); \
    return static_cast<int8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
uint8_t F2I_##RND<uint8_t, double>(const double &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u8.f64 %0, %1;" \
        : "=r"(d) : "d"(s) \
    ); \
    return static_cast<uint8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
int16_t F2I_##RND<int16_t, double>(const double &s) { \
    int16_t d; \
    asm ( \
        "cvt."#RND_PTX".s16.f64 %0, %1;" \
        : "=h"(d) : "d"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint16_t F2I_##RND<uint16_t, double>(const double &s) { \
    uint16_t d; \
    asm ( \
        "cvt."#RND_PTX".u16.f64 %0, %1;" \
        : "=h"(d) : "d"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int32_t F2I_##RND<int32_t, double>(const double &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s32.f64 %0, %1;" \
        : "=r"(d) : "d"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint32_t F2I_##RND<uint32_t, double>(const double &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u32.f64 %0, %1;" \
        : "=r"(d) : "d"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int64_t F2I_##RND<int64_t, double>(const double &s) { \
    int64_t d; \
    asm ( \
        "cvt."#RND_PTX".s64.f64 %0, %1;" \
        : "=l"(d) : "d"(s) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint64_t F2I_##RND<uint64_t, double>(const double &s) { \
    uint64_t d; \
    asm ( \
        "cvt."#RND_PTX".u64.f64 %0, %1;" \
        : "=l"(d) : "d"(s) \
    ); \
    return d; \
}

__DOUBLE_INT(RD, rmi)
__DOUBLE_INT(RU, rpi)
__DOUBLE_INT(RN, rni)
__DOUBLE_INT(RZ, rzi)
#undef __DOUBLE_INT

// --------------------------------------
// half to integer
// --------------------------------------
#define __HALF_INT(RND, RND_PTX) \
template <> \
__device__ __forceinline__ \
int8_t F2I_##RND<int8_t, half>(const half &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s8.f16 %0, %1;" \
        : "=r"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return static_cast<int8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
uint8_t F2I_##RND<uint8_t, half>(const half &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u8.f16 %0, %1;" \
        : "=r"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return static_cast<uint8_t>(d); \
} \
\
template <> \
__device__ __forceinline__ \
int16_t F2I_##RND<int16_t, half>(const half &s) { \
    int16_t d; \
    asm ( \
        "cvt."#RND_PTX".s16.f16 %0, %1;" \
        : "=h"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint16_t F2I_##RND<uint16_t, half>(const half &s) { \
    uint16_t d; \
    asm ( \
        "cvt."#RND_PTX".u16.f16 %0, %1;" \
        : "=h"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int32_t F2I_##RND<int32_t, half>(const half &s) { \
    int32_t d; \
    asm ( \
        "cvt."#RND_PTX".s32.f16 %0, %1;" \
        : "=r"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint32_t F2I_##RND<uint32_t, half>(const half &s) { \
    uint32_t d; \
    asm ( \
        "cvt."#RND_PTX".u32.f16 %0, %1;" \
        : "=r"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
int64_t F2I_##RND<int64_t, half>(const half &s) { \
    int64_t d; \
    asm ( \
        "cvt."#RND_PTX".s64.f16 %0, %1;" \
        : "=l"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
} \
\
template <> \
__device__ __forceinline__ \
uint64_t F2I_##RND<uint64_t, half>(const half &s) { \
    uint64_t d; \
    asm ( \
        "cvt."#RND_PTX".u64.f16 %0, %1;" \
        : "=l"(d) \
        : "h"(reinterpret_cast<const uint16_t &>(s)) \
    ); \
    return d; \
}

#ifdef HIEDNN_USE_FP16
__HALF_INT(RD, rmi)
__HALF_INT(RU, rpi)
__HALF_INT(RN, rni)
__HALF_INT(RZ, rzi)
#endif

// --------------------------------------
// bfloat16 to integer
// --------------------------------------
#define __BFLOAT16_INT(RND, RND_PTX) \
template <> \
__device__ __forceinline__ \
int8_t F2I_##RND<int8_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<int8_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
uint8_t F2I_##RND<uint8_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<uint8_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
int16_t F2I_##RND<int16_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<int16_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
uint16_t F2I_##RND<uint16_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<uint16_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
int32_t F2I_##RND<int32_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<int32_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
uint32_t F2I_##RND<uint32_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<uint32_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
int64_t F2I_##RND<int64_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<int64_t>(static_cast<float>(s)); \
} \
\
template <> \
__device__ __forceinline__ \
uint64_t F2I_##RND<uint64_t, bfloat16>(const bfloat16 &s) { \
    return F2I_##RND<uint64_t>(static_cast<float>(s)); \
}

#ifdef HIEDNN_USE_BF16
__BFLOAT16_INT(RD, rmi)
__BFLOAT16_INT(RU, rpi)
__BFLOAT16_INT(RN, rni)
__BFLOAT16_INT(RZ, rzi)
#endif

#undef __HALF_INT
#undef __BFLOAT16_INT

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_INTRINSIC_TYPE_CONVERSION_HPP_

