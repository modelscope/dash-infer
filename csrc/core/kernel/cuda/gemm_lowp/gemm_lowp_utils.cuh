/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_lowp_utils.cuh
 */

#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "../hie/cuda_activation.hpp"
#include "cuda/cuda_common.h"
#include "cuda/gemm_utils.h"

template <typename T>
struct HalfType;
template <>
struct HalfType<half> {
  using T1 = __half;
  using T2 = __half2;
};
template <>
struct HalfType<hie::bfloat16> {
  using T1 = __nv_bfloat16;
  using T2 = __nv_bfloat162;
};

template <typename T>
__device__ __forceinline__ T dequantize_func(const T& data, const T& scale,
                                             const T& zero) {
  return (data - zero) * scale;
}

template <>
__device__ __forceinline__ __nv_bfloat162 dequantize_func<__nv_bfloat162>(
    const __nv_bfloat162& data, const __nv_bfloat162& scale,
    const __nv_bfloat162& zero) {
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  return (data - zero) * scale;
#else
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 3
  __builtin_unreachable();
#else
  return __nv_bfloat162{};
#endif  // __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 3
#endif  // __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
}

template <typename T>
__device__ __host__ __forceinline__ T Ceil(const T a, const T b) {
  return (a + b - 1) / b * b;
}

template <typename T>
__device__ __host__ __forceinline__ T DivCeil(const T a, const T b) {
  return (a + b - 1) / b;
}
template <typename T>
__device__ __forceinline__ void ldg128_cg(T& r0, T& r1, T& r2, T& r3,
                                          const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg128_cg: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
      " @p ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg128_ca(T& r0, T& r1, T& r2, T& r3,
                                          const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg128_ca: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.ca.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
      " @p ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "l"(ptr), "r"((int)guard));
}

// template <typename T>
// __device__ __forceinline__ void stg128(const T& r0, const T& r1, const T& r2,
//                                        const T& r3, const void* ptr) {
//   static_assert(sizeof(T) == 4, "stg128: invalid T");
//   asm volatile("{st.global.v4.b32 [%0], {%1, %2, %3, %4};}\n"
//                :
//                : "l"(ptr), "r"(reinterpret_cast<const uint32_t&>(r0)),
//                  "r"(reinterpret_cast<const uint32_t&>(r1)),
//                  "r"(reinterpret_cast<const uint32_t&>(r2)),
//                  "r"(reinterpret_cast<const uint32_t&>(r3)));
// }

template <typename T>
__device__ __forceinline__ void stg16(const T& r0, const void* ptr,
                                      bool guard) {
  static_assert(sizeof(T) == 2, "stg16: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %1, 0;\n"
      " @p st.global.b16 [%0], %2;}\n"
      :
      : "l"(ptr), "r"((int)guard), "h"(reinterpret_cast<const uint16_t&>(r0)));
}

template <typename T>
__device__ __forceinline__ void stg128(const T& r0, const T& r1, const T& r2,
                                       const T& r3, const void* ptr,
                                       bool guard) {
  static_assert(sizeof(T) == 4, "stg128: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %1, 0;\n"
      " @p st.global.v4.b32 [%0], {%2, %3, %4, %5};}\n"
      :
      : "l"(ptr), "r"((int)guard), "r"(reinterpret_cast<const uint32_t&>(r0)),
        "r"(reinterpret_cast<const uint32_t&>(r1)),
        "r"(reinterpret_cast<const uint32_t&>(r2)),
        "r"(reinterpret_cast<const uint32_t&>(r3)));
}

template <typename T>
__device__ __forceinline__ void ldg128_ca_0(T& r0, T& r1, T& r2, T& r3,
                                            const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg128_ca_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @!p mov.b32 %2, 0;\n"
      " @!p mov.b32 %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.ca.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
      " @p ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg128_cg_0(T& r0, T& r1, T& r2, T& r3,
                                            const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg128_cg_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @!p mov.b32 %2, 0;\n"
      " @!p mov.b32 %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
      " @p ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg64_ca_0(T& r0, T& r1, const void* ptr,
                                           bool guard) {
  static_assert(sizeof(T) == 4, "ldg64_ca_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.ca.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
      " @p ld.global.ca.v2.b32 {%0, %1}, [%2];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg64_ca(T& r0, T& r1, const void* ptr,
                                         bool guard) {
  static_assert(sizeof(T) == 4, "ldg64_ca: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.ca.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
      " @p ld.global.ca.v2.b32 {%0, %1}, [%2];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg64_nc_0(T& r0, T& r1, const void* ptr,
                                           bool guard) {
  static_assert(sizeof(T) == 4, "ldg64_nc_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.nc.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
      " @p ld.global.nc.v2.b32 {%0, %1}, [%2];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg64_nc(T& r0, T& r1, const void* ptr,
                                         bool guard) {
  static_assert(sizeof(T) == 4, "ldg64_nc: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.nc.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
      " @p ld.global.nc.v2.b32 {%0, %1}, [%2];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg32_cs(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg32_cs: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b32 %0, [%1];}\n"
#else
      " @p ld.global.cs.b32 %0, [%1];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg32_cs_0(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg32_cs_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b32 %0, [%1];}\n"
#else
      " @p ld.global.cs.b32 %0, [%1];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg32_cg_0(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg32_cg_0: invalid T");
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 {%0}, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cg.L2::128B.b32 {%0}, [%1];}\n"
#else
      " @p ld.global.cg.b32 {%0}, [%1];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg32_cg(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg32_cg: invalid T");
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cg.L2::128B.b32 {%0}, [%1];}\n"
#else
      " @p ld.global.cg.b32 {%0}, [%1];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg32_ca(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 4, "ldg32_ca: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.ca.L2::128B.b32 %0, [%1];}\n"
#else
      " @p ld.global.ca.b32 %0, [%1];}\n"
#endif
      : "=r"(reinterpret_cast<uint32_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldg16_cg_0(T& r0, const void* ptr, bool guard) {
  static_assert(sizeof(T) == 2, "ldg16_cg_0: invalid T");

  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b16 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cg.L2::128B.b16 {%0}, [%1];}\n"
#else
      " @p ld.global.ca.b16 {%0}, [%1];}\n"
#endif
      : "=h"(reinterpret_cast<uint16_t&>(r0))
      : "l"(ptr), "r"((int)guard));
}

template <typename T>
__device__ __forceinline__ void ldsm_4(T& r0, T& r1, T& r2, T& r3,
                                       const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "ldsm_4: invalid T");
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "r"(addr));
#endif
}

template <typename T>
__device__ __forceinline__ void ldsm_4_trans(T& r0, T& r1, T& r2, T& r3,
                                             const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "ldsm_4_trans: invalid T");
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "r"(addr));
#endif
}

template <typename T>
__device__ __forceinline__ void hmma1688_f16(T& d0, T& d1, const T& a0,
                                             const T& a1, const T& b0) {
  static_assert(sizeof(T) == 4, "hmma1688_f16: invalid T");
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, "
      "{%4}, {%0, %1};\n"
      : "+r"(d0), "+r"(d1)
      : "r"(a0), "r"(a1), "r"(b0));
#endif
}

template <typename T>
__device__ __forceinline__ void hmma1688_f32(T& d0, T& d1, T& d2, T& d3,
                                             const T& a0, const T& a1,
                                             const T& b0) {
  static_assert(sizeof(T) == 4, "hmma1688_f32: invalid T");
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11000) && !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, "
      "%5}, {%6}, {%0, %1, %2, %3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// inplace hadd
template <typename T>
__device__ __forceinline__ void hadd8(T& d0, T& d1, T& d2, T& d3, const T& a0,
                                      const T& a1, const T& a2, const T& a3) {
  asm("add.rn.f16x2 %0, %0, %4;\n"
      "add.rn.f16x2 %1, %1, %5;\n"
      "add.rn.f16x2 %2, %2, %6;\n"
      "add.rn.f16x2 %3, %3, %7;\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3));
}

// ****************

template <typename T>
struct DivModRet {
  T div;
  T mod;
  __device__ __forceinline__ DivModRet(T d, T m) : div(d), mod(m) {}
};

template <typename T>
class IntDivMod {
 public:
  uint32_t d_;

  template <typename DT>
  static bool OutOfBound(const DT* d, int count) {
    return true;
  }

  IntDivMod() {}

  explicit IntDivMod(T d) : d_(d) {}

  __device__ __forceinline__ T Div(T n) const { return n / d_; }

  __device__ __forceinline__ T Mod(T n) const { return n % d_; }

  __device__ __forceinline__ DivModRet<T> DivMod(T n) const {
    return DivModRet<T>(n / d_, n % d_);
  }
};

template <>
class IntDivMod<uint32_t> {
 public:
  uint32_t d_;

 private:
  uint32_t magic_;
  uint32_t shift_;

 public:
  template <typename DT>
  static bool OutOfBound(const DT* d, int count) {
    for (int i = 0; i < count; ++i) {
      if (static_cast<uint64_t>(d[i]) > INT32_MAX) {
        return true;
      }
    }
    return false;
  }

  IntDivMod() {}

  explicit IntDivMod(uint32_t d) {
    // assert(d >= 1 && d <= INT32_MAX);
    d_ = d;

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) {
        break;
      }
    }
    uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
                         // assert(magic_ != 0 && magic_ == tmp_magic);
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
#ifdef __CUDA_ARCH__
    return (__umulhi(n, magic_) + n) >> shift_;
#else
    uint64_t tmp = (static_cast<uint64_t>(n) * magic_) >> 32;
    return (static_cast<uint32_t>(tmp) + n) >> shift_;
#endif
  }

  __device__ __forceinline__ uint32_t Mod(uint32_t n) const {
    return n - Div(n) * d_;
  }

  __device__ __forceinline__ DivModRet<uint32_t> DivMod(uint32_t n) const {
    uint32_t d = Div(n);
    return DivModRet<uint32_t>(d, n - d_ * d);
  }
};

using U32DivMod = IntDivMod<uint32_t>;

namespace PackConfig {
// max size in byte of the vector (packed elements)
constexpr size_t MAX_PACKED_BYTE = 16;
// max number of packed elements
constexpr size_t MAX_PACKED_SIZE = 8;
}  // namespace PackConfig

// Get pack size based on alignment
template <typename T>
inline size_t GetPackSize(const T* ptr) {
  if (sizeof(T) > PackConfig::MAX_PACKED_BYTE) {
    return 1;
  }

  size_t packSize =
      PackConfig::MAX_PACKED_BYTE / sizeof(T) < PackConfig::MAX_PACKED_SIZE
          ? PackConfig::MAX_PACKED_BYTE / sizeof(T)
          : PackConfig::MAX_PACKED_SIZE;

  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

  while (addr % (packSize * sizeof(T)) != 0) {
    packSize /= 2;
  }

  return packSize;
}

struct PackedEltwiseConfig {
  int64_t nPack;
  int64_t nThread;
  int64_t nBlock;

  /*
   * @n: number of elements
   * @packSize: packed size in elements
   * @block: size of thread block
   */
  PackedEltwiseConfig(int64_t n, int64_t packSize, int64_t block) {
    nPack = n / packSize;
    nThread = nPack + (n % packSize);
    nBlock = (nThread + block - 1) / block;
  }
};

// [NumSplitK, M, N]
template <typename FT, template <class> class ActiveFunc>
__global__ void reduce_sum(const FT* data_in, const FT* bias, FT* data_out,
                           const int M, const int N, const int K,
                           const int NumSplitK) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t m_idx = blockIdx.y;

  const FT* data_in_ptr = data_in + m_idx * N;
  FT* data_out_ptr = data_out + m_idx * N;
  if (tid < N) {
    FT bias_val = FT(0);
    if (bias != nullptr) {
      bias_val = bias[tid];
    }
    FT sum = 0;
    for (int i = 0; i < NumSplitK; ++i) {
      FT val = data_in_ptr[i * M * N + tid];
      sum += val;
    }
    sum += bias_val;
    data_out_ptr[tid] = ActiveFunc<FT>::Op(sum);
  }
}

template <typename FType, int BLOCK, int UNROLL,
          template <class> class ActiveFunc>
__global__ void add_bias_kernel(const FType* bias, FType* data_out,
                                const uint32_t M, const uint32_t N,
                                const float alpha, const U32DivMod nDivMod,
                                PackedEltwiseConfig packConfig) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

  if (idx < packConfig.nPack) {
    FType fval_reg[UNROLL];

    *reinterpret_cast<uint4*>(fval_reg) =
        *(reinterpret_cast<const uint4*>(data_out) + idx);

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      fval_reg[i] = static_cast<FType>((float)fval_reg[i] * alpha);
      if (bias != nullptr) {
        int n_idx = nDivMod.Mod(idx * UNROLL + i);
        fval_reg[i] += bias[n_idx];
      }
      fval_reg[i] = ActiveFunc<FType>::Op(fval_reg[i]);
    }

    stg128(*reinterpret_cast<uint32_t*>(fval_reg),
           *reinterpret_cast<uint32_t*>(fval_reg + 2),
           *reinterpret_cast<uint32_t*>(fval_reg + 4),
           *reinterpret_cast<uint32_t*>(fval_reg + 6),
           reinterpret_cast<uint4*>(data_out) + idx);
  } else if (idx < packConfig.nThread) {
    idx = idx - packConfig.nPack + packConfig.nPack * UNROLL;
    FType fval = static_cast<FType>((float)data_out[idx] * alpha);
    if (bias != nullptr) {
      int n_idx = nDivMod.Mod(idx);
      fval += bias[n_idx];
    }
    data_out[idx] = ActiveFunc<FType>::Op(fval);
  }
}

template <typename FType, template <class> class ActiveFunc>
void add_bias(const FType* bias, FType* data_out, const uint32_t M,
              const uint32_t N, const float alpha, cudaStream_t stream) {
  int packSize = GetPackSize(data_out);
  const int BLOCK_SIZE = 128;
  PackedEltwiseConfig packConfig(M * N, packSize, BLOCK_SIZE);
  U32DivMod nDivMod(N);
  switch (packSize) {
    case 8:
      add_bias_kernel<FType, BLOCK_SIZE, 8, ActiveFunc>
          <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
              bias, data_out, M, N, alpha, nDivMod, packConfig);
      break;
    default:
      LOG(ERROR) << "Now only support in/out ptr is 16-byte aligned";
      break;
  }
}

template <typename FType, int BLOCK, int N_MATRIX,
          template <class> class ActiveFunc>
__global__ void gemm_f16_splitk_reduce_kernel(const FType* C_split,
                                              const FType* B_scale,
                                              const FType* bias, FType* C,
                                              int n, int n_matrix,
                                              int matrix_size, float alpha) {
  int idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx >= matrix_size) {
    return;
  }

  float sum = 0.0f;

  int n_mat = N_MATRIX > 0 ? N_MATRIX : n_matrix;
  for (int i = 0; i < n_mat; ++i) {
    sum += static_cast<float>(C_split[idx + i * matrix_size]);
  }

  // for gemm_lowp perc gemm/gemv kernel, the scale multiplication can be placed
  // here
  if (B_scale != nullptr) {
    sum *= static_cast<float>(B_scale[idx % n]);
  }
  // C = alpha * A * B
  sum = sum * alpha;
  if (bias != nullptr) {
    sum += static_cast<float>(bias[idx % n]);
  }
  FType sum_t = static_cast<FType>(sum);
  C[idx] = ActiveFunc<FType>::Op(sum_t);
}

template <typename FType, template <class> class ActiveFunc>
void gemm_f16_splitk_reduce(const FType* C_split, const FType* B_scale,
                            const FType* bias, FType* C, const int m,
                            const int n, const int n_matrix, const float alpha,
                            cudaStream_t stream) {
  const int BLOCK = 128;
  int matrix_size = m * n;
  int grid = (matrix_size + BLOCK - 1) / BLOCK;

  void (*kernel)(const FType*, const FType*, const FType*, FType*, int, int,
                 int, float) = nullptr;

  switch (n_matrix) {
    case 4:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 4, ActiveFunc>;
      break;
    case 5:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 5, ActiveFunc>;
      break;
    case 6:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 6, ActiveFunc>;
      break;
    case 7:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 7, ActiveFunc>;
      break;
    case 8:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 8, ActiveFunc>;
      break;
    case 9:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 9, ActiveFunc>;
      break;
    case 10:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 10, ActiveFunc>;
      break;
    case 11:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 11, ActiveFunc>;
      break;
    case 12:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, 12, ActiveFunc>;
      break;
    default:
      kernel = gemm_f16_splitk_reduce_kernel<FType, BLOCK, -1, ActiveFunc>;
      break;
  }

  kernel<<<grid, BLOCK, 0, stream>>>(C_split, B_scale, bias, C, n, n_matrix,
                                     matrix_size, alpha);
}

// Transpose 4x4 8bit
__device__ __forceinline__ uint32_t prmt(uint32_t s_03, uint32_t s_47,
                                         uint32_t index) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(s_03), "r"(s_47), "r"(index));
  return r;
}

__device__ __forceinline__ void transpose4x4x8bit(const uint32_t (&in)[4],
                                                  uint32_t (&out)[4]) {
  uint32_t x = prmt(in[0], in[1], 0x5140);
  uint32_t y = prmt(in[2], in[3], 0x5140);
  uint32_t z = prmt(in[0], in[1], 0x7362);
  uint32_t w = prmt(in[2], in[3], 0x7362);

  out[0] = prmt(x, y, 0x5410);
  out[1] = prmt(x, y, 0x7632);
  out[2] = prmt(z, w, 0x5410);
  out[3] = prmt(z, w, 0x7632);
}

/**
 * @brief hmma 16816 TensorCore
 */
template <typename T0, typename T1>
__device__ __forceinline__ void hmma_16816(const hie::Array<T0, 8>& a,
                                           const hie::Array<T0, 4>& b,
                                           const hie::Array<T1, 4>& c,
                                           hie::Array<T1, 4>& d);

template <>
__device__ __forceinline__ void hmma_16816<half, float>(
    const hie::Array<half, 8>& a, const hie::Array<half, 4>& b,
    const hie::Array<float, 4>& c, hie::Array<float, 4>& d) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const* C = reinterpret_cast<float const*>(&c);
  float* D = reinterpret_cast<float*>(&d);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
}

template <>
__device__ __forceinline__ void hmma_16816<hie::bfloat16, float>(
    const hie::Array<hie::bfloat16, 8>& a,
    const hie::Array<hie::bfloat16, 4>& b, const hie::Array<float, 4>& c,
    hie::Array<float, 4>& d) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const* C = reinterpret_cast<float const*>(&c);
  float* D = reinterpret_cast<float*>(&d);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
}

template <typename FType>
__device__ __forceinline__ void hmma16816_f32(float (&d)[4],
                                              const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]);

template <>
__device__ __forceinline__ void hmma16816_f32<__half>(float (&d)[4],
                                                      const uint32_t (&a)[4],
                                                      const uint32_t (&b)[2]) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template <>
__device__ __forceinline__ void hmma16816_f32<hie::bfloat16>(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11) && \
    !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/**
 * @brief Asynchronous copy since Ampere Arch
 */
template <int SIZE_IN_BYTES>
__device__ __forceinline__ void cp_async(const uint32_t smem_addr,
                                         const void* gmem_ptr,
                                         const int src_in_bytes, bool guard) {
  static_assert(
      (SIZE_IN_BYTES == 4 || SIZE_IN_BYTES == 8 || SIZE_IN_BYTES == 16),
      "Size is not supported");
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %4, 0;\n"
#if __CUDACC_VER_MINOR__ >= 4 && !defined(__HGGCCC__)
      " @p cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;}\n"
#else
      " @p cp.async.cg.shared.global [%0], [%1], %2, %3;}\n"
#endif
      ::"r"(smem_addr),
      "l"(gmem_ptr), "n"(SIZE_IN_BYTES), "r"(src_in_bytes), "r"((int)guard));
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
__device__ __forceinline__ void cp_asyc_wait_group() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

/**
 * @brief mma 884 TensorCore for Volta
 */
__device__ __forceinline__ void mma_h884(const uint32_t& a0, const uint32_t& a1,
                                         const uint32_t& b0, const uint32_t& b1,
                                         uint32_t& d0, uint32_t& d1,
                                         uint32_t& d2, uint32_t& d3) {
#if !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(d0), "r"(d1), "r"(d2), "r"(d3));
#endif
}


static constexpr int WARP_SIZE = 32;

/**
 * @brief Warp Reduce and Block Reduce
 */

template <typename T>
struct MaxOp {
 public:
  static constexpr T init = std::numeric_limits<T>::min();
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
 public:
  static constexpr float init = -std::numeric_limits<float>::infinity();
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x > y ? x : y;
  }
};

template <typename T>
struct SumOp {
 public:
  static constexpr T init = T(0);
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x + y;
  }
};

template <template <class> class Func, typename T>
__device__ __forceinline__ T ReduceWarp(T val) {
// float ret;
// constexpr uint32_t MASK = 0xffffffff;
// asm("redux.sync.max.u32 %0, %1, 0xffffffff;"
//     : "=r"(reinterpret_cast<uint32_t&>(ret))
//     : "r"(reinterpret_cast<const uint32_t&>(val)));
// return ret;
#pragma unroll
  for (int i = WARP_SIZE; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = Func<T>::op(tmp, val);
  }

  return val;
}

template <template <class> class Func, typename T, int BLOCK>
__device__ __forceinline__ T ReduceBlock(const T& val) {
  static_assert(BLOCK >= WARP_SIZE, "Invalid Block size");
  __shared__ T smem[BLOCK / WARP_SIZE];
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  T val_reg = Func<T>::init;
  val_reg = Func<T>::op(val_reg, val);
  val_reg = ReduceWarp<Func, T>(val_reg);
  if (BLOCK / WARP_SIZE > 1) {
    if (lane_id == 0) {
      smem[warp_id] = val_reg;
    }
    __syncthreads();
    if (warp_id == 0) {
      val_reg = lane_id < BLOCK / WARP_SIZE ? smem[lane_id] : Func<T>::init;
      val_reg = ReduceWarp<Func, T>(val_reg);
    }
  }
  __syncthreads();
  return val_reg;
}

/**
 * @brief fast conversion uint8->f16 (with bias 128)
 */
template <typename T>
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128(const uint32_t& idata,
                                                          T* fdata);

template <>
// fast convertion: 4xuint8 to 4xhalf, substracting bias = 128
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128<__half2>(
    const uint32_t& idata, __half2* fdata) {
  uint32_t i10, i32;
  asm volatile(
      "prmt.b32 %0, %2, 0x64, 0x4140;"
      "prmt.b32 %1, %2, 0x64, 0x4342;"
      : "=r"(i10), "=r"(i32)
      : "r"(idata));

  static constexpr uint32_t MAGIC_NUM = 0x64806480;
  fdata[0] = __hsub2(reinterpret_cast<const __half2&>(i10),
                     reinterpret_cast<const __half2&>(MAGIC_NUM));
  fdata[1] = __hsub2(reinterpret_cast<const __half2&>(i32),
                     reinterpret_cast<const __half2&>(MAGIC_NUM));
}

template <>
// fast convertion: 4xuint8 to 4xbfloat16, substracting bias = 128
__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128<__nv_bfloat162>(
    const uint32_t& idata, __nv_bfloat162* fdata) {
  float fp32_imd[4];
  uint32_t* fp32_imd_casted = reinterpret_cast<uint32_t*>(fp32_imd);
  asm volatile(
      "prmt.b32 %0, %4, 0x4B000000, 0x7650;"
      "prmt.b32 %1, %4, 0x4B000000, 0x7651;"
      "prmt.b32 %2, %4, 0x4B000000, 0x7652;"
      "prmt.b32 %3, %4, 0x4B000000, 0x7653;"
      : "=r"(fp32_imd_casted[0]), "=r"(fp32_imd_casted[1]),
        "=r"(fp32_imd_casted[2]), "=r"(fp32_imd_casted[3])
      : "r"(idata));

  fp32_imd[0] -= 8388736.f;
  fp32_imd[1] -= 8388736.f;
  fp32_imd[2] -= 8388736.f;
  fp32_imd[3] -= 8388736.f;

  uint32_t* bf16_res = reinterpret_cast<uint32_t*>(fdata);
  asm volatile(
      "prmt.b32 %0, %2, %3, 0x7632;"
      "prmt.b32 %1, %4, %5, 0x7632;"
      : "=r"(bf16_res[0]), "=r"(bf16_res[1])
      : "r"(fp32_imd_casted[0]), "r"(fp32_imd_casted[1]),
        "r"(fp32_imd_casted[2]), "r"(fp32_imd_casted[3]));
}

static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162bfloat162(x);
#else
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 3
  __builtin_unreachable();
#else
  return nv_bfloat162{};
#endif  // __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 3
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
}

static __device__ half2 inline num2num2(const half x) {
  return __half2half2(x);
}
