/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_utils.h
 */

#pragma once
#include "cuda_common.h"  // NOLINT

// PTX assembly for load from global memory
__device__ __forceinline__ void ldg64(uint32_t& r0, uint32_t& r1,
                                      const void* ptr) {
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(" {ld.global.cg.L2::128B.v2.u32 {%0, %1}, [%2];}\n"
               : "=r"(r0), "=r"(r1)
               : "l"(ptr));
#else   // (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040)
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "l"(ptr));
#endif  // (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040)
}

__device__ __forceinline__ void ldg128(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                       uint32_t& r3, const void* ptr) {
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile("{ ld.global.cg.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
#else   // (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040)
  asm volatile(" {ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
#endif  // (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040)
}

// PTX assembly for store to global memory
__device__ __forceinline__ void stg64(const uint32_t& r0, const uint32_t& r1,
                                      const void* ptr) {
  asm volatile("{st.global.v2.b32 [%0], {%1, %2};}\n"
               :
               : "l"(ptr), "r"(r0), "r"(r1));
}

__device__ __forceinline__ void stg128(const uint32_t& r0, const uint32_t& r1,
                                       const uint32_t& r2, const uint32_t& r3,
                                       const void* ptr) {
  asm volatile("{st.global.v4.b32 [%0], {%1, %2, %3, %4};}\n"
               :
               : "l"(ptr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

// convert 64-bit pointer to 32-bit smem addr
__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

// PTX assembly for store to shared memory
template <typename T>
__device__ __forceinline__ void sts16(const T reg0, const uint32_t addr) {
  static_assert(sizeof(T) == 2, "sts16: invalid T");

  asm volatile("st.shared.b16 [%0], %1;\n"
               :
               : "r"(addr), "h"(reinterpret_cast<const uint16_t&>(reg0)));
}
template <typename T>
__device__ __forceinline__ void sts32(const T reg0, const uint32_t addr) {
  static_assert(sizeof(T) == 4, "sts32: invalid T");

  asm volatile("st.shared.b32 [%0], %1;\n"
               :
               : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(reg0)));
}
template <typename T>
__device__ __forceinline__ void sts64(const T reg0, const T reg1,
                                      const uint32_t addr) {
  static_assert(sizeof(T) == 4, "sts64: invalid T");

  asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
               :
               : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(reg0)),
                 "r"(reinterpret_cast<const uint32_t&>(reg1)));
}
template <typename T>
__device__ __forceinline__ void sts128(const T reg0, const T reg1, const T reg2,
                                       const T reg3, const uint32_t addr) {
  static_assert(sizeof(T) == 4, "sts128: invalid T");

  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(reg0)),
                 "r"(reinterpret_cast<const uint32_t&>(reg1)),
                 "r"(reinterpret_cast<const uint32_t&>(reg2)),
                 "r"(reinterpret_cast<const uint32_t&>(reg3)));
}

// store data to shared memory using inline PTX Assembly
// here consider different packing size
template <typename T, uint32_t UNROLL>
__device__ __forceinline__ void store_sdata(void* data, uint32_t addr) {
  constexpr uint32_t WordSize = sizeof(T) * UNROLL;
  if (WordSize == 2) {
    sts16(reinterpret_cast<uint16_t*>(data)[0], addr);
  } else if (WordSize == 4) {
    sts32(reinterpret_cast<uint32_t*>(data)[0], addr);
  } else if (WordSize == 8) {
    sts64(reinterpret_cast<uint32_t*>(data)[0],
          reinterpret_cast<uint32_t*>(data)[1], addr);
  } else if (WordSize == 16) {
    sts128(reinterpret_cast<uint32_t*>(data)[0],
           reinterpret_cast<uint32_t*>(data)[1],
           reinterpret_cast<uint32_t*>(data)[2],
           reinterpret_cast<uint32_t*>(data)[3], addr);
  } else {
    static_assert(WordSize <= 16, "");
  }
}

// PTX assembly for load from shared memory
template <typename T>
__device__ __forceinline__ void lds32(T& reg0, const uint32_t addr) {
  static_assert(sizeof(T) == 4, "lds32: invalid T");
  asm volatile("ld.shared.b32 {%0}, [%1];\n"
               : "=r"(reinterpret_cast<uint32_t&>(reg0))
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void lds64(T& reg0, T& reg1, const uint32_t addr) {
  static_assert(sizeof(T) == 4, "lds64: invalid T");

  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n"
               : "=r"(reinterpret_cast<uint32_t&>(reg0)),
                 "=r"(reinterpret_cast<uint32_t&>(reg1))
               : "r"(addr));
}
template <typename T>
__device__ __forceinline__ void lds128(T& reg0, T& reg1, T& reg2, T& reg3,
                                       const uint32_t addr) {
  static_assert(sizeof(T) == 4, "lds128: invalid T");

  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(reinterpret_cast<uint32_t&>(reg0)),
                 "=r"(reinterpret_cast<uint32_t&>(reg1)),
                 "=r"(reinterpret_cast<uint32_t&>(reg2)),
                 "=r"(reinterpret_cast<uint32_t&>(reg3))
               : "r"(addr));
}

// optional cache operators on load instructions intoduced in PTX ISA 2.0
// Cache operators on load instructions are treated as performance hints only.
enum LoadCacheOperator : int32_t {
  CA = 0,  // cache in all levels (L1 and L2)
  CG = 1,  // cache in L2 and below, not L1
  CS = 2   // evict-first policy in L1 and L2
           // Future extensions go here.
           // ...
};

// PTX assembly for load from global memory with memory guard and cache hint
template <typename T>
__device__ __forceinline__ void ldg128_cache_hint(
    T& r0, T& r1, T& r2, T& r3, const void* ptr, bool guard,
    LoadCacheOperator cache_hint) {
  static_assert(sizeof(T) == 4, "ldg128_cache_hint: invalid T");

  if (cache_hint == LoadCacheOperator::CA) {
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
  } else if (cache_hint == LoadCacheOperator::CG) {
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
  } else if (cache_hint == LoadCacheOperator::CS) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %5, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cs.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#else
        " @p ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];}\n"
#endif
        : "=r"(reinterpret_cast<uint32_t&>(r0)),
          "=r"(reinterpret_cast<uint32_t&>(r1)),
          "=r"(reinterpret_cast<uint32_t&>(r2)),
          "=r"(reinterpret_cast<uint32_t&>(r3))
        : "l"(ptr), "r"((int)guard));
  }
}
template <typename T>
__device__ __forceinline__ void ldg64_cache_hint(T& r0, T& r1, const void* ptr,
                                                 bool guard,
                                                 LoadCacheOperator cache_hint) {
  static_assert(sizeof(T) == 4, "ldg64_cache_hint: invalid T");
  if (cache_hint == LoadCacheOperator::CA) {
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
  } else if (cache_hint == LoadCacheOperator::CG) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cg.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
        " @p ld.global.cg.v2.b32 {%0, %1}, [%2];}\n"
#endif
        : "=r"(reinterpret_cast<uint32_t&>(r0)),
          "=r"(reinterpret_cast<uint32_t&>(r1))
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CS) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cs.L2::128B.v2.b32 {%0, %1}, [%2];}\n"
#else
        " @p ld.global.cs.v2.b32 {%0, %1}, [%2];}\n"
#endif
        : "=r"(reinterpret_cast<uint32_t&>(r0)),
          "=r"(reinterpret_cast<uint32_t&>(r1))
        : "l"(ptr), "r"((int)guard));
  }
}
template <typename T>
__device__ __forceinline__ void ldg32_cache_hint(T& r0, const void* ptr,
                                                 bool guard,
                                                 LoadCacheOperator cache_hint) {
  static_assert(sizeof(T) == 4, "ldg32_cache_hint: invalid T");
  if (cache_hint == LoadCacheOperator::CA) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.ca.L2::128B.b32 {%0}, [%1];}\n"
#else
        " @p ld.global.ca.b32 {%0}, [%1];}\n"
#endif
        : "=r"(reinterpret_cast<uint32_t&>(r0))
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CG) {
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
  } else if (cache_hint == LoadCacheOperator::CS) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cs.L2::128B.b32 {%0}, [%1];}\n"
#else
        " @p ld.global.cs.b32 {%0}, [%1];}\n"
#endif
        : "=r"(reinterpret_cast<uint32_t&>(r0))
        : "l"(ptr), "r"((int)guard));
  }
}
template <typename T>
__device__ __forceinline__ void ldg16_cache_hint(T& r0, const void* ptr,
                                                 bool guard,
                                                 LoadCacheOperator cache_hint) {
  static_assert(sizeof(T) == 2, "ldg16_cache_hint: invalid T");
  if (cache_hint == LoadCacheOperator::CA) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.ca.L2::128B.b16 {%0}, [%1];}\n"
#else
        " @p ld.global.ca.b16 {%0}, [%1];}\n"
#endif
        : "=h"(reinterpret_cast<uint16_t&>(r0))
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CG) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cg.L2::128B.b16 {%0}, [%1];}\n"
#else
        " @p ld.global.cg.b16 {%0}, [%1];}\n"
#endif
        : "=h"(reinterpret_cast<uint16_t&>(r0))
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CS) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cs.L2::128B.b16 {%0}, [%1];}\n"
#else
        " @p ld.global.cs.b16 {%0}, [%1];}\n"
#endif
        : "=h"(reinterpret_cast<uint16_t&>(r0))
        : "l"(ptr), "r"((int)guard));
  }
}
template <typename T>
__device__ __forceinline__ void ldg8_cache_hint(T& r0, const void* ptr,
                                                bool guard,
                                                LoadCacheOperator cache_hint) {
  static_assert(sizeof(T) == 1, "ldg8_cache_hint: invalid T");
  int d;
  if (cache_hint == LoadCacheOperator::CA) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.ca.L2::128B.b8 {%0}, [%1];}\n"
#else
        " @p ld.global.ca.b8 {%0}, [%1];}\n"
#endif
        : "=r"(d)
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CG) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cg.L2::128B.b8 {%0}, [%1];}\n"
#else
        " @p ld.global.cg.b8 {%0}, [%1];}\n"
#endif
        : "=r"(d)
        : "l"(ptr), "r"((int)guard));
  } else if (cache_hint == LoadCacheOperator::CS) {
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
        " @p ld.global.cs.L2::128B.b8 {%0}, [%1];}\n"
#else
        " @p ld.global.cs.b8 {%0}, [%1];}\n"
#endif
        : "=r"(d)
        : "l"(ptr), "r"((int)guard));
  }
  r0 = d;
}

// load data from global memory using inline PTX Assembly
// here consider cache operator and memory guard
// UNROLL means pack size
template <typename T, uint32_t UNROLL>
__device__ __forceinline__ void load_gdata_cache_hint(
    void* data, const void* ptr, bool guard, LoadCacheOperator cache_hint) {
  constexpr uint32_t WordSize = sizeof(T) * UNROLL;
  if (WordSize == 1) {
    ldg8_cache_hint(*reinterpret_cast<uint8_t*>(data), ptr, guard, cache_hint);
  } else if (WordSize == 2) {
    ldg16_cache_hint(*reinterpret_cast<uint16_t*>(data), ptr, guard,
                     cache_hint);
  } else if (WordSize == 4) {
    ldg32_cache_hint(*reinterpret_cast<uint32_t*>(data), ptr, guard,
                     cache_hint);
  } else if (WordSize == 8) {
    ldg64_cache_hint(reinterpret_cast<uint32_t*>(data)[0],
                     reinterpret_cast<uint32_t*>(data)[1], ptr, guard,
                     cache_hint);
  } else if (WordSize == 16) {
    ldg128_cache_hint(reinterpret_cast<uint32_t*>(data)[0],
                      reinterpret_cast<uint32_t*>(data)[1],
                      reinterpret_cast<uint32_t*>(data)[2],
                      reinterpret_cast<uint32_t*>(data)[3], ptr, guard,
                      cache_hint);
  } else {
    static_assert(WordSize <= 16, "");
  }
}

// load data from global memory considering different packing size
template <typename T, uint32_t UNROLL>
__device__ __forceinline__ void load_gdata(void* data, const void* ptr) {
  constexpr uint32_t WordSize = sizeof(T) * UNROLL;
  if (WordSize == 1) {
    reinterpret_cast<int8_t*>(data)[0] =
        *(reinterpret_cast<const int8_t*>(ptr));
  } else if (WordSize == 2) {
    reinterpret_cast<int16_t*>(data)[0] =
        *(reinterpret_cast<const int16_t*>(ptr));
  } else if (WordSize == 4) {
    reinterpret_cast<int32_t*>(data)[0] =
        *(reinterpret_cast<const int32_t*>(ptr));
  } else if (WordSize == 8) {
    ldg64(reinterpret_cast<uint32_t*>(data)[0],
          reinterpret_cast<uint32_t*>(data)[1], ptr);
  } else if (WordSize == 16) {
    ldg128(reinterpret_cast<uint32_t*>(data)[0],
           reinterpret_cast<uint32_t*>(data)[1],
           reinterpret_cast<uint32_t*>(data)[2],
           reinterpret_cast<uint32_t*>(data)[3], ptr);
  } else {
    static_assert(WordSize <= 16, "");
  }
}

// store data to global memory considering different packing size
template <typename T, uint32_t UNROLL>
__device__ __forceinline__ void store_gdata(void* data, void* ptr) {
  constexpr uint32_t WordSize = sizeof(T) * UNROLL;
  if (WordSize == 1) {
    reinterpret_cast<int8_t*>(ptr)[0] = reinterpret_cast<int8_t*>(data)[0];
  } else if (WordSize == 2) {
    reinterpret_cast<int16_t*>(ptr)[0] = reinterpret_cast<int16_t*>(data)[0];
  } else if (WordSize == 4) {
    reinterpret_cast<int32_t*>(ptr)[0] = reinterpret_cast<int32_t*>(data)[0];
  } else if (WordSize == 8) {
    stg64(reinterpret_cast<const uint32_t*>(data)[0],
          reinterpret_cast<const uint32_t*>(data)[1], ptr);
  } else if (WordSize == 16) {
    stg128(reinterpret_cast<const uint32_t*>(data)[0],
           reinterpret_cast<const uint32_t*>(data)[1],
           reinterpret_cast<const uint32_t*>(data)[2],
           reinterpret_cast<const uint32_t*>(data)[3], ptr);
  } else {
    static_assert(WordSize <= 16, "");
  }
}

struct GEMM_Fp16_Params {
  const half* A_ptr;
  const half* B_ptr;
  half* C_ptr;
  half* C_split_ptr;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t SplitK;
};