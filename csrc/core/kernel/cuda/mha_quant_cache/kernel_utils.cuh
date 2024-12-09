/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_utils.cuh
 */

#ifndef __MHA_QUANT_CACHE_KERNEL_UTILS_CUH__
#define __MHA_QUANT_CACHE_KERNEL_UTILS_CUH__
#include <cuda/cuda_common.h>
#include <stdint.h>

#include <cuda/hie/cuda_intdivider.hpp>
#include <limits>

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
using u32div_t = hie::internal::IntDivModer<uint32_t>;
#ifdef ENABLE_FP16
using half_t = half;
#else   // ENABLE_FP16
using half_t = uint16_t;
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
using bf16_t = __hie_buildin::bfloat16;
#else   // ENABLE_BF16
using bf16_t = uint16_t;
#endif  // ENABLE_BF16

namespace utils {
template <typename T>
T cal_ceil(T ind, T div) {
  return (ind % div) ? (1 + ind / div) : (ind / div);
}
template <typename T>
T cal_align(T ind, T div) {
  return div * cal_ceil<T>(ind, div);
}

constexpr int32_t warp_size = 32;
template <int32_t ALIGN, typename TYPE>
struct alignas(ALIGN * sizeof(TYPE)) packed_data {
  TYPE pack[ALIGN];
};
template <int32_t ALIGN, typename TYPE>
struct non_packed_data {
  TYPE pack[ALIGN];
};
template <typename T>
struct MaxOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x > y ? x : y;
  }
};
template <>
struct MaxOp<float> {
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x > y ? x : y;
  }
};
template <typename T>
struct MinOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x < y ? x : y;
  }
};
template <>
struct MinOp<float> {
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x < y ? x : y;
  }
};
template <typename T>
struct SumOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x + y;
  }
};

template <template <class> class Func, typename T, int32_t nTid = warp_size>
__device__ __forceinline__ T ReduceThread(T val) {
  static_assert(nTid == 2 || nTid == 4 || nTid == 8 || nTid == 16 ||
                nTid == 32);
#pragma unroll
  for (int i = nTid; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = Func<T>::op(tmp, val);
  }
  return val;
}

template <typename CPT, typename QT, int32_t QTPack>
__device__ __forceinline__ packed_data<QTPack, CPT> UnpackW8AndDequant(
    const QT& w8, const CPT& scale, const CPT& zero) {
  packed_data<QTPack, CPT> ret;
  if (QTPack == 1) {
    ret.pack[0] = (static_cast<CPT>(w8 - zero)) * scale;
  } else if (std::is_same<QT, int8_t>::value && QTPack == 2) {
    ret.pack[0] = (static_cast<CPT>((w8 << 4) >> 4) - zero) * scale;
    ret.pack[1] = (static_cast<CPT>(w8 >> 4) - zero) * scale;
  } else if (std::is_same<QT, uint8_t>::value && QTPack == 2) {
    ret.pack[0] = (static_cast<CPT>(w8 & 0xf) - zero) * scale;
    ret.pack[1] = (static_cast<CPT>(w8 >> 4) - zero) * scale;
  } else {
    asm volatile("brkpt;\n" ::);
  }
  return ret;
}

template <typename STT, int32_t QTPack = 1>
struct quant_tag {
  constexpr static bool require_quant = false;
  constexpr static float quant_zero = static_cast<float>(0.f);  // not use
  constexpr static float quant_div_scale = 1.f;                 // not use
  constexpr static float quant_max = INFINITY;
  constexpr static float quant_min = -INFINITY;
};

constexpr float quant_i8max =
    static_cast<float>(std::numeric_limits<int8_t>::max());
constexpr float quant_i8min =
    static_cast<float>(std::numeric_limits<int8_t>::min());
template <>
struct quant_tag<int8_t, 1> {
  constexpr static bool require_quant = true;
  constexpr static float quant_zero = quant_i8min;
  constexpr static float quant_div_scale = quant_i8max - quant_i8min;
  constexpr static float quant_max = quant_i8max;
  constexpr static float quant_min = quant_i8min;
};

// UINT4x2
constexpr float quant_u4max = static_cast<float>(15);
constexpr float quant_u4min = static_cast<float>(0);
template <>
struct quant_tag<uint8_t, 2> {
  constexpr static bool require_quant = true;
  constexpr static float quant_zero = quant_u4min;
  constexpr static float quant_div_scale = quant_u4max - quant_u4min;
  constexpr static float quant_max = quant_u4max;
  constexpr static float quant_min = quant_u4min;
};

template <typename TYPE>
struct pack_infer {
  constexpr static int32_t PACK = 1;
};
template <>
struct pack_infer<float> {
  constexpr static int32_t PACK = 4;
};
template <>
struct pack_infer<half_t> {
  constexpr static int32_t PACK = 8;
};
template <>
struct pack_infer<bf16_t> {
  constexpr static int32_t PACK = 8;
};
template <>
struct pack_infer<int8_t> {
  constexpr static int32_t PACK = 16;
};
template <>
struct pack_infer<uint8_t> {
  constexpr static int32_t PACK = 16;
};

enum LdgHint {
  NONE,
  CACHE_L2_ONLY,
  CACHE_STREAMING,
  CACHE_NON_COHERENT,
  EVICT_LAST,
};

template <typename TYPE, LdgHint HINT = NONE>
struct LdgFPCache {
  __device__ __forceinline__ TYPE operator()(const void* gptr) {
    return reinterpret_cast<const TYPE*>(gptr)[0];
  }
};

template <>
struct LdgFPCache<float, EVICT_LAST> {
  __device__ __forceinline__ float operator()(const void* gptr) {
    float regs;
    asm volatile("ld.global.L1::evict_last.b32 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgFPCache<half_t, EVICT_LAST> {
  __device__ __forceinline__ half_t operator()(const void* gptr) {
    half_t regs;
    asm volatile("ld.global.L1::evict_last.b16 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgFPCache<bf16_t, EVICT_LAST> {
  __device__ __forceinline__ bf16_t operator()(const void* gptr) {
    bf16_t regs;
    asm volatile("ld.global.L1::evict_last.b16 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

using i8x4_t = packed_data<4, int8_t>;
using i8x8_t = packed_data<8, int8_t>;
template <int32_t PACK, LdgHint HINT = NONE>
struct LdgI8Cache {
  __device__ __forceinline__ packed_data<PACK, int8_t> operator()(
      const void* gptr) {
    return reinterpret_cast<const packed_data<PACK, int8_t>*>(gptr)[0];
  }
};

template <>
struct LdgI8Cache<4, CACHE_L2_ONLY> {
  __device__ __forceinline__ i8x4_t operator()(const void* gptr) {
    i8x4_t regs;
    asm volatile("ld.global.cg.b32 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgI8Cache<8, CACHE_L2_ONLY> {
  __device__ __forceinline__ i8x8_t operator()(const void* gptr) {
    i8x8_t regs;
    asm volatile("ld.global.cg.v2.b32 {%0, %1}, [%2];"
                 : "=r"(reinterpret_cast<uint2&>(regs).x) "=r"(
                     reinterpret_cast<uint2&>(regs).y)
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgI8Cache<4, CACHE_STREAMING> {
  __device__ __forceinline__ i8x4_t operator()(const void* gptr) {
    i8x4_t regs;
    asm volatile("ld.global.cs.b32 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgI8Cache<8, CACHE_STREAMING> {
  __device__ __forceinline__ i8x8_t operator()(const void* gptr) {
    i8x8_t regs;
    asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2];"
                 : "=r"(reinterpret_cast<uint2&>(regs).x) "=r"(
                     reinterpret_cast<uint2&>(regs).y)
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgI8Cache<4, CACHE_NON_COHERENT> {
  __device__ __forceinline__ i8x4_t operator()(const void* gptr) {
    i8x4_t regs;
    asm volatile("ld.global.cs.nc.b32 %0, [%1];"
                 : "=r"(reinterpret_cast<uint32_t&>(regs))
                 : "l"(gptr));
    return regs;
  }
};

template <>
struct LdgI8Cache<8, CACHE_NON_COHERENT> {
  __device__ __forceinline__ i8x8_t operator()(const void* gptr) {
    i8x8_t regs;
    asm volatile("ld.global.cs.nc.v2.b32 {%0, %1}, [%2];"
                 : "=r"(reinterpret_cast<uint2&>(regs).x) "=r"(
                     reinterpret_cast<uint2&>(regs).y)
                 : "l"(gptr));
    return regs;
  }
};

}  // namespace utils
}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
#endif  // __MHA_QUANT_CACHE_KERNEL_UTILS_CUH__
