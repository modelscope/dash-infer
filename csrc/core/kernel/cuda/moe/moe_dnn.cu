/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_dnn.cu
 */
#include <cuda_bf16.h>  // CUDA-11.0+
#include <cuda_fp16.h>  // CUDA_9.0+

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../cuda_kernel.h"
#include "allspark.pb.h"
#include "moe_dnn.h"
namespace allspark {
namespace cuda {
template <typename T16>
struct T16x2;

template <>
struct T16x2<__half> {
  using T = __half2;
};

template <>
struct T16x2<__nv_bfloat16> {
  using T = __nv_bfloat162;
};

struct DivMod {
  uint32_t div;
  uint32_t mod;
  __device__ __forceinline__ DivMod(uint32_t d, uint32_t m) : div(d), mod(m) {}
};

struct U16DivMod {
  uint32_t magic_;
  uint16_t shift_;
  uint16_t d_;

  __host__ __device__ U16DivMod() {}

  __host__ __device__ __forceinline__ explicit U16DivMod(uint32_t d) : d_(d) {
#ifdef __CUDA_ARCH__
    uint32_t msb;
    asm("bfind.u32 %0, %1;" : "=r"(msb) : "r"(d));
    if ((1U << msb) < d) {
      ++msb;
    }
    shift_ = msb;
#else
    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1U << shift_) >= d) break;
    }
#endif
    uint64_t tmp_magic = ((1LU << 32) * ((1U << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
  }

  __device__ __forceinline__ uint32_t div(uint32_t n) {
    return (__umulhi(n, magic_) + n) >> shift_;
  }

  __device__ __forceinline__ uint32_t mod(uint32_t n) {
    return n - div(n) * d_;
  }

  __device__ __forceinline__ DivMod divmod(uint32_t n) {
    uint32_t d = div(n);
    return DivMod(d, n - d_ * d);
  }
};

struct U32DivMod {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  __host__ __device__ U32DivMod() {}

  __host__ __device__ __forceinline__ explicit U32DivMod(uint32_t d) : d_(d) {
#ifdef __CUDA_ARCH__
    asm("bfind.u32 %0, %1;" : "=r"(shift_) : "r"(d));
    if ((1U << shift_) < d) {
      ++shift_;
    }
#else
    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1U << shift_) >= d) break;
    }
#endif
    uint64_t tmp_magic = ((1LU << 32) * ((1U << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
  }

  __device__ __forceinline__ uint32_t div(uint32_t n) {
    return (__umulhi(n, magic_) + n) >> shift_;
  }

  __device__ __forceinline__ uint32_t mod(uint32_t n) {
    return n - div(n) * d_;
  }

  __device__ __forceinline__ DivMod divmod(uint32_t n) {
    uint32_t d = div(n);
    return DivMod(d, n - d_ * d);
  }
};

struct alignas(16) TileRemap {
  U16DivMod wave_m_divmod;       // 8 Byte
  U16DivMod last_wave_m_divmod;  // 8 Byte
  U32DivMod wave_size_divmod;    // 12 Byte
  uint32_t full_wave_tiles;      // 4 Byte

  __host__ __device__ TileRemap() {}

  __device__ __forceinline__ TileRemap(uint32_t m_tiles, uint32_t m_tile_size,
                                       uint32_t n_tiles, uint32_t n_tile_size,
                                       float tiles_per_wave) {
    // wave_m * (wave_m / (n_tile_size / m_tile_size)) = tiles_per_wave
    // pseudo code:
    // wave_m = sqrt(tiles_per_wave * n_tile_size / m_tile_size);
    uint32_t wave_m;
    asm("{.reg .b32 r0, r1;\n"
        " rcp.approx.ftz.f32 r0, %3;\n"
        " mul.rn.f32 r1, %1, %2;\n"
        " mul.rn.f32 r0, r0, r1;\n"
        " sqrt.approx.ftz.f32 r0, r0;\n"
        " cvt.rni.u32.f32 %0, r0;}"
        : "=r"(wave_m)
        : "f"(tiles_per_wave), "f"(static_cast<float>(n_tile_size)),
          "f"(static_cast<float>(m_tile_size)));
    if (wave_m > m_tiles) {
      wave_m = m_tiles;
    }
    // if last_wave_m is 0, set value 1 to avoid U16DivMod exception
    uint32_t last_wave_m = m_tiles % wave_m == 0 ? 1 : m_tiles % wave_m;
    uint32_t wave_size = wave_m * n_tiles;
    full_wave_tiles = m_tiles / wave_m * wave_size;

    wave_m_divmod = U16DivMod(wave_m);
    last_wave_m_divmod = U16DivMod(last_wave_m);
    wave_size_divmod = U32DivMod(wave_size);
  }

  __device__ __forceinline__ void remap(const uint32_t& ctaid,
                                        uint32_t* n_tile_id,
                                        uint32_t* m_tile_id) {
    auto wave_size_dm = wave_size_divmod.divmod(ctaid);
    auto wave_m_dm = ctaid < full_wave_tiles
                         ? wave_m_divmod.divmod(wave_size_dm.mod)
                         : last_wave_m_divmod.divmod(wave_size_dm.mod);
    const auto& wave_m = wave_m_divmod.d_;
    *n_tile_id = wave_m_dm.div;
    *m_tile_id = wave_size_dm.div * wave_m + wave_m_dm.mod;
  }
};

struct alignas(16) BatchInfo {
  uint32_t batchId;
  uint32_t m;
  uint32_t ctaOffset;
  uint32_t COffset;
};

__device__ __forceinline__ uint32_t smem_u32addr(const void* smemptr) {
  uint32_t u32addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr; }\n"
      : "=r"(u32addr)
      : "l"(smemptr));
  return u32addr;
}

__device__ __forceinline__ void ldgsts64(const uint32_t& smem_addr,
                                         const void* gmem_ptr, bool guard) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 8;}\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void ldgsts64_zfill(const uint32_t& smem_addr,
                                               const void* gmem_ptr, bool guard,
                                               bool ignore_src) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p0, p1;\n"
      " setp.ne.b32 p0, %2, 0;\n"
      " setp.ne.b32 p1, %3, 0;\n"
      " @p0 cp.async.ca.shared.global.L2::128B [%0], [%1], 8, p1;}\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard), "r"((int)ignore_src));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void ldgsts128(const uint32_t& smem_addr,
                                          const void* gmem_ptr, bool guard) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p cp.async.cg.shared.global.L2::128B [%0], [%1], 16;}\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void ldgsts128_zfill(const uint32_t& smem_addr,
                                                const void* gmem_ptr,
                                                bool guard, bool ignore_src) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "{.reg.pred p0, p1;\n"
      " setp.ne.b32 p0, %2, 0;\n"
      " setp.ne.b32 p1, %3, 0;\n"
      " @p0 cp.async.cg.shared.global.L2::128B [%0], [%1], 16, p1;}\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard), "r"((int)ignore_src));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void ldgsts_group_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#else
  asm volatile("trap;");
#endif
}

template <int N>
__device__ __forceinline__ void ldgsts_group_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#else
  asm volatile("trap;");
#endif
}

template <typename T>
__device__ __forceinline__ void lds128(T& r0, T& r1, T& r2, T& r3,
                                       const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "lds128: invalid T");
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(reinterpret_cast<uint32_t&>(r0)),
                 "=r"(reinterpret_cast<uint32_t&>(r1)),
                 "=r"(reinterpret_cast<uint32_t&>(r2)),
                 "=r"(reinterpret_cast<uint32_t&>(r3))
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void stg64(const T& r0, const T& r1, const void* ptr,
                                      bool guard) {
  static_assert(sizeof(T) == 4, "stg64: invalid T");
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %1, 0;\n"
      " @p st.global.v2.b32 [%0], {%2, %3};}\n"
      :
      : "l"(ptr), "r"((int)guard), "r"(reinterpret_cast<const uint32_t&>(r0)),
        "r"(reinterpret_cast<const uint32_t&>(r1)));
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

__device__ __forceinline__ void cvt_f32_to_t16(__half2& dst, const float& src0,
                                               const float& src1) {
#if __CUDA_ARCH__ >= 800
  asm("cvt.rn.f16x2.f32 %0, %2, %1;"
      : "=r"(reinterpret_cast<uint32_t&>(dst))
      : "f"(src0), "f"(src1));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cvt_f32_to_t16(__nv_bfloat162& dst,
                                               const float& src0,
                                               const float& src1) {
#if __CUDA_ARCH__ >= 800
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
      : "=r"(reinterpret_cast<uint32_t&>(dst))
      : "f"(src0), "f"(src1));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void warp_prefix_scan(uint32_t mask, uint32_t& var,
                                                 int offset, int width) {
  int shfl_c = (32 - width) << 8;
  asm("{.reg .s32 r0;\n"
      " .reg .pred p0;\n"
      " shfl.sync.up.b32 r0|p0, %0, %1, %2, %3;\n"
      " @p0 add.u32 %0, %0, r0;}"
      : "+r"(var)
      : "r"(offset), "r"(shfl_c), "r"(mask));
}

template <typename T16>
__device__ __forceinline__ void hgmma_64x16_f32_transA(
    uint32_t (&C_frag)[2][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);
template <typename T16>
__device__ __forceinline__ void hgmma_64x32_f32_transA(
    uint32_t (&C_frag)[4][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);
template <typename T16>
__device__ __forceinline__ void hgmma_64x48_f32_transA(
    uint32_t (&C_frag)[6][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);
template <typename T16>
__device__ __forceinline__ void hgmma_64x96_f32_transA(
    uint32_t (&C_frag)[12][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);
template <typename T16>
__device__ __forceinline__ void hgmma_64x128_f32_transB(
    uint32_t (&C_frag)[16][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);
template <typename T16>
__device__ __forceinline__ void hgmma_64x256_f32_transB(
    uint32_t (&C_frag)[32][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc);

template <>
__device__ __forceinline__ void hgmma_64x16_f32_transA<__half>(
    uint32_t (&C_frag)[2][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x32_f32_transA<__half>(
    uint32_t (&C_frag)[4][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      "%16, %17, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x48_f32_transA<__half>(
    uint32_t (&C_frag)[6][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n48k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23},"
      "%24, %25, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x96_f32_transA<__half>(
    uint32_t (&C_frag)[12][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n96k16.f32.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47}, "
      "%48, %49, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x128_f32_transB<__half>(
    uint32_t (&C_frag)[16][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
      "%64, %65, 1, 1, 1, 0, 1;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1]),
        "+r"(C_frag[12][0][0]), "+r"(C_frag[12][0][1]), "+r"(C_frag[12][1][0]),
        "+r"(C_frag[12][1][1]), "+r"(C_frag[13][0][0]), "+r"(C_frag[13][0][1]),
        "+r"(C_frag[13][1][0]), "+r"(C_frag[13][1][1]), "+r"(C_frag[14][0][0]),
        "+r"(C_frag[14][0][1]), "+r"(C_frag[14][1][0]), "+r"(C_frag[14][1][1]),
        "+r"(C_frag[15][0][0]), "+r"(C_frag[15][0][1]), "+r"(C_frag[15][1][0]),
        "+r"(C_frag[15][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x256_f32_transB<__half>(
    uint32_t (&C_frag)[32][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
      " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
      " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
      " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
      " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
      " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
      " %104, %105, %106, %107, %108, %109, %110, %111, "
      " %112, %113, %114, %115, %116, %117, %118, %119, "
      " %120, %121, %122, %123, %124, %125, %126, %127},"
      "%128, %129, 1, 1, 1, 0, 1;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1]),
        "+r"(C_frag[12][0][0]), "+r"(C_frag[12][0][1]), "+r"(C_frag[12][1][0]),
        "+r"(C_frag[12][1][1]), "+r"(C_frag[13][0][0]), "+r"(C_frag[13][0][1]),
        "+r"(C_frag[13][1][0]), "+r"(C_frag[13][1][1]), "+r"(C_frag[14][0][0]),
        "+r"(C_frag[14][0][1]), "+r"(C_frag[14][1][0]), "+r"(C_frag[14][1][1]),
        "+r"(C_frag[15][0][0]), "+r"(C_frag[15][0][1]), "+r"(C_frag[15][1][0]),
        "+r"(C_frag[15][1][1]), "+r"(C_frag[16][0][0]), "+r"(C_frag[16][0][1]),
        "+r"(C_frag[16][1][0]), "+r"(C_frag[16][1][1]), "+r"(C_frag[17][0][0]),
        "+r"(C_frag[17][0][1]), "+r"(C_frag[17][1][0]), "+r"(C_frag[17][1][1]),
        "+r"(C_frag[18][0][0]), "+r"(C_frag[18][0][1]), "+r"(C_frag[18][1][0]),
        "+r"(C_frag[18][1][1]), "+r"(C_frag[19][0][0]), "+r"(C_frag[19][0][1]),
        "+r"(C_frag[19][1][0]), "+r"(C_frag[19][1][1]), "+r"(C_frag[20][0][0]),
        "+r"(C_frag[20][0][1]), "+r"(C_frag[20][1][0]), "+r"(C_frag[20][1][1]),
        "+r"(C_frag[21][0][0]), "+r"(C_frag[21][0][1]), "+r"(C_frag[21][1][0]),
        "+r"(C_frag[21][1][1]), "+r"(C_frag[22][0][0]), "+r"(C_frag[22][0][1]),
        "+r"(C_frag[22][1][0]), "+r"(C_frag[22][1][1]), "+r"(C_frag[23][0][0]),
        "+r"(C_frag[23][0][1]), "+r"(C_frag[23][1][0]), "+r"(C_frag[23][1][1]),
        "+r"(C_frag[24][0][0]), "+r"(C_frag[24][0][1]), "+r"(C_frag[24][1][0]),
        "+r"(C_frag[24][1][1]), "+r"(C_frag[25][0][0]), "+r"(C_frag[25][0][1]),
        "+r"(C_frag[25][1][0]), "+r"(C_frag[25][1][1]), "+r"(C_frag[26][0][0]),
        "+r"(C_frag[26][0][1]), "+r"(C_frag[26][1][0]), "+r"(C_frag[26][1][1]),
        "+r"(C_frag[27][0][0]), "+r"(C_frag[27][0][1]), "+r"(C_frag[27][1][0]),
        "+r"(C_frag[27][1][1]), "+r"(C_frag[28][0][0]), "+r"(C_frag[28][0][1]),
        "+r"(C_frag[28][1][0]), "+r"(C_frag[28][1][1]), "+r"(C_frag[29][0][0]),
        "+r"(C_frag[29][0][1]), "+r"(C_frag[29][1][0]), "+r"(C_frag[29][1][1]),
        "+r"(C_frag[30][0][0]), "+r"(C_frag[30][0][1]), "+r"(C_frag[30][1][0]),
        "+r"(C_frag[30][1][1]), "+r"(C_frag[31][0][0]), "+r"(C_frag[31][0][1]),
        "+r"(C_frag[31][1][0]), "+r"(C_frag[31][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x16_f32_transA<__nv_bfloat16>(
    uint32_t (&C_frag)[2][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x32_f32_transA<__nv_bfloat16>(
    uint32_t (&C_frag)[4][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      "%16, %17, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x48_f32_transA<__nv_bfloat16>(
    uint32_t (&C_frag)[6][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23},"
      "%24, %25, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x96_f32_transA<__nv_bfloat16>(
    uint32_t (&C_frag)[12][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47}, "
      "%48, %49, 1, 1, 1, 1, 0;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x128_f32_transB<__nv_bfloat16>(
    uint32_t (&C_frag)[16][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
      "%64, %65, 1, 1, 1, 0, 1;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1]),
        "+r"(C_frag[12][0][0]), "+r"(C_frag[12][0][1]), "+r"(C_frag[12][1][0]),
        "+r"(C_frag[12][1][1]), "+r"(C_frag[13][0][0]), "+r"(C_frag[13][0][1]),
        "+r"(C_frag[13][1][0]), "+r"(C_frag[13][1][1]), "+r"(C_frag[14][0][0]),
        "+r"(C_frag[14][0][1]), "+r"(C_frag[14][1][0]), "+r"(C_frag[14][1][1]),
        "+r"(C_frag[15][0][0]), "+r"(C_frag[15][0][1]), "+r"(C_frag[15][1][0]),
        "+r"(C_frag[15][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

template <>
__device__ __forceinline__ void hgmma_64x256_f32_transB<__nv_bfloat16>(
    uint32_t (&C_frag)[32][2][2], const uint64_t& A_desc,
    const uint64_t& B_desc) {
#if __CUDA_ARCH__ >= 900
  asm("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
      " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
      " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
      " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
      " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
      " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
      " %104, %105, %106, %107, %108, %109, %110, %111, "
      " %112, %113, %114, %115, %116, %117, %118, %119, "
      " %120, %121, %122, %123, %124, %125, %126, %127},"
      "%128, %129, 1, 1, 1, 0, 1;"
      : "+r"(C_frag[0][0][0]), "+r"(C_frag[0][0][1]), "+r"(C_frag[0][1][0]),
        "+r"(C_frag[0][1][1]), "+r"(C_frag[1][0][0]), "+r"(C_frag[1][0][1]),
        "+r"(C_frag[1][1][0]), "+r"(C_frag[1][1][1]), "+r"(C_frag[2][0][0]),
        "+r"(C_frag[2][0][1]), "+r"(C_frag[2][1][0]), "+r"(C_frag[2][1][1]),
        "+r"(C_frag[3][0][0]), "+r"(C_frag[3][0][1]), "+r"(C_frag[3][1][0]),
        "+r"(C_frag[3][1][1]), "+r"(C_frag[4][0][0]), "+r"(C_frag[4][0][1]),
        "+r"(C_frag[4][1][0]), "+r"(C_frag[4][1][1]), "+r"(C_frag[5][0][0]),
        "+r"(C_frag[5][0][1]), "+r"(C_frag[5][1][0]), "+r"(C_frag[5][1][1]),
        "+r"(C_frag[6][0][0]), "+r"(C_frag[6][0][1]), "+r"(C_frag[6][1][0]),
        "+r"(C_frag[6][1][1]), "+r"(C_frag[7][0][0]), "+r"(C_frag[7][0][1]),
        "+r"(C_frag[7][1][0]), "+r"(C_frag[7][1][1]), "+r"(C_frag[8][0][0]),
        "+r"(C_frag[8][0][1]), "+r"(C_frag[8][1][0]), "+r"(C_frag[8][1][1]),
        "+r"(C_frag[9][0][0]), "+r"(C_frag[9][0][1]), "+r"(C_frag[9][1][0]),
        "+r"(C_frag[9][1][1]), "+r"(C_frag[10][0][0]), "+r"(C_frag[10][0][1]),
        "+r"(C_frag[10][1][0]), "+r"(C_frag[10][1][1]), "+r"(C_frag[11][0][0]),
        "+r"(C_frag[11][0][1]), "+r"(C_frag[11][1][0]), "+r"(C_frag[11][1][1]),
        "+r"(C_frag[12][0][0]), "+r"(C_frag[12][0][1]), "+r"(C_frag[12][1][0]),
        "+r"(C_frag[12][1][1]), "+r"(C_frag[13][0][0]), "+r"(C_frag[13][0][1]),
        "+r"(C_frag[13][1][0]), "+r"(C_frag[13][1][1]), "+r"(C_frag[14][0][0]),
        "+r"(C_frag[14][0][1]), "+r"(C_frag[14][1][0]), "+r"(C_frag[14][1][1]),
        "+r"(C_frag[15][0][0]), "+r"(C_frag[15][0][1]), "+r"(C_frag[15][1][0]),
        "+r"(C_frag[15][1][1]), "+r"(C_frag[16][0][0]), "+r"(C_frag[16][0][1]),
        "+r"(C_frag[16][1][0]), "+r"(C_frag[16][1][1]), "+r"(C_frag[17][0][0]),
        "+r"(C_frag[17][0][1]), "+r"(C_frag[17][1][0]), "+r"(C_frag[17][1][1]),
        "+r"(C_frag[18][0][0]), "+r"(C_frag[18][0][1]), "+r"(C_frag[18][1][0]),
        "+r"(C_frag[18][1][1]), "+r"(C_frag[19][0][0]), "+r"(C_frag[19][0][1]),
        "+r"(C_frag[19][1][0]), "+r"(C_frag[19][1][1]), "+r"(C_frag[20][0][0]),
        "+r"(C_frag[20][0][1]), "+r"(C_frag[20][1][0]), "+r"(C_frag[20][1][1]),
        "+r"(C_frag[21][0][0]), "+r"(C_frag[21][0][1]), "+r"(C_frag[21][1][0]),
        "+r"(C_frag[21][1][1]), "+r"(C_frag[22][0][0]), "+r"(C_frag[22][0][1]),
        "+r"(C_frag[22][1][0]), "+r"(C_frag[22][1][1]), "+r"(C_frag[23][0][0]),
        "+r"(C_frag[23][0][1]), "+r"(C_frag[23][1][0]), "+r"(C_frag[23][1][1]),
        "+r"(C_frag[24][0][0]), "+r"(C_frag[24][0][1]), "+r"(C_frag[24][1][0]),
        "+r"(C_frag[24][1][1]), "+r"(C_frag[25][0][0]), "+r"(C_frag[25][0][1]),
        "+r"(C_frag[25][1][0]), "+r"(C_frag[25][1][1]), "+r"(C_frag[26][0][0]),
        "+r"(C_frag[26][0][1]), "+r"(C_frag[26][1][0]), "+r"(C_frag[26][1][1]),
        "+r"(C_frag[27][0][0]), "+r"(C_frag[27][0][1]), "+r"(C_frag[27][1][0]),
        "+r"(C_frag[27][1][1]), "+r"(C_frag[28][0][0]), "+r"(C_frag[28][0][1]),
        "+r"(C_frag[28][1][0]), "+r"(C_frag[28][1][1]), "+r"(C_frag[29][0][0]),
        "+r"(C_frag[29][0][1]), "+r"(C_frag[29][1][0]), "+r"(C_frag[29][1][1]),
        "+r"(C_frag[30][0][0]), "+r"(C_frag[30][0][1]), "+r"(C_frag[30][1][0]),
        "+r"(C_frag[30][1][1]), "+r"(C_frag[31][0][0]), "+r"(C_frag[31][0][1]),
        "+r"(C_frag[31][1][0]), "+r"(C_frag[31][1][1])
      : "l"(A_desc), "l"(B_desc));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void warpgroup_arrive() {
#if __CUDA_ARCH__ >= 900
  asm volatile("wgmma.fence.sync.aligned;");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void warpgroup_commit() {
#if __CUDA_ARCH__ >= 900
  asm volatile("wgmma.commit_group.sync.aligned;");
#else
  asm volatile("trap;");
#endif
}

template <int N>
__device__ __forceinline__ void warpgroup_depbar_le() {
#if __CUDA_ARCH__ >= 900
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(N));
#else
  asm volatile("trap;");
#endif
}

/**
 * m_tile: 128
 * n_tile: 256
 * k_tile: 64x4
 * smem size: 192KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m128n256_k64x4_hgmma64x256_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (16 + 32) * 4 = 192 KB
  // A_smem: 128 * 64 * sizeof(T16) = 16 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 16 * 1024;

  uint32_t matA_row_id[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = tile_y * 128 + threadIdx.x / 8 + i * 32;
    if (m_idx < m) {
      asm("ld.global.nc.b32 %0, [%1];"
          : "=r"(matA_row_id[i])
          : "l"(matA_row_idx + m_idx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matA_row_id[i] = 0;
    }
  }

  const char* A_ldg_ptr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    A_ldg_ptr[i] = reinterpret_cast<const char*>(A + matA_row_id[i] * k +
                                                 (threadIdx.x % 8) * 8);
  }
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 8 % 8)) * 16;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start = (A_smem_addr + warpgroup_id * 8192) / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4];
  {
    uint32_t matrix_start = B_smem_addr / 16;
    uint32_t ld = 8 * 128 / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start | (ld << 16);
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t lo = desc_lo + i * 16 * 256 * sizeof(T16) / 16;
      asm("mov.b64 %0, {%1, %2};" : "=l"(matB_desc[i]) : "r"(lo), "r"(desc_hi));
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore = threadIdx.x % 8 * 8 >= first_k_tile;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldgsts128_zfill(A_sts_addr + i * 32 * 64 * sizeof(T16), A_ldg_ptr[i],
                      true, A_ignore);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      A_ldg_ptr[i] += first_k_tile * sizeof(T16);
    }
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 2; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ldgsts128(
            A_sts_addr + prefetch_iter * 49152 + i * 32 * 64 * sizeof(T16),
            A_ldg_ptr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 49152 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        A_ldg_ptr[i] += 64 * sizeof(T16);
      }
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 49152 * 2;

  // C register fragment
  uint32_t C_frag[32][2][2];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      C_frag[i][j][0] = 0;
      C_frag[i][j][1] = 0;
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<1>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 2; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      hgmma_64x256_f32_transB<T16>(C_frag, matA_desc[i], matB_desc[i]);
    }
    warpgroup_commit();

// tile prefetch
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldgsts128(A_sts_addr + sts_offset + i * 32 * 64 * sizeof(T16),
                A_ldg_ptr[i], true);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 49152 * 1 ? -(49152 / 16 * 3) : 49152 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matB_desc[i])
          : "r"(lds_offset));
    }

    // switch double buffer
    sts_offset = sts_offset < 49152 * 3 ? sts_offset + 49152 : 0;

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      A_ldg_ptr[i] += 64 * sizeof(T16);
    }
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<1>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      hgmma_64x256_f32_transB<T16>(C_frag, matA_desc[i], matB_desc[i]);
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 49152 * 1 ? -(49152 / 16 * 3) : 49152 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matB_desc[i])
          : "r"(lds_offset));
    }

    // switch double buffer
    sts_offset = sts_offset < 49152 * 3 ? sts_offset + 49152 : 0;

    ldgsts_group_wait<1>();
    __syncthreads();
  }

  warpgroup_depbar_le<0>();
#pragma unroll
  for (int i = 0; i < 32; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      // guide the compiler to reuse C_frag register
      cvt_f32_to_t16(reinterpret_cast<typename T16x2<T16>::T&>(C_frag[i][j][0]),
                     reinterpret_cast<const float&>(C_frag[i][j][0]),
                     reinterpret_cast<const float&>(C_frag[i][j][1]));
    }
  }

  uint32_t* C_sts_ptr =
      reinterpret_cast<uint32_t*>(smem) +
      (threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4) * (128 + 4) +
      threadIdx.x % 4;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 32; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      C_sts_ptr[i * 4 + j * 8 * (128 + 4)] = C_frag[i][j][0];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(uint32_t) * (threadIdx.x / 32 * (128 + 4) + threadIdx.x % 32 * 4);
  uint32_t C_stg_reg[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    lds128(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_reg[i][2], C_stg_reg[i][3],
           C_lds_addr + sizeof(uint32_t) * (i * 8 * (128 + 4)));
  }

  uint32_t m_idx = tile_y * 128 + (threadIdx.x / 32);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 32) * 8;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    stg128(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_reg[i][2], C_stg_reg[i][3],
           C_stg_ptr + i * 8 * n, m_idx + i * 8 < m && n_idx < n);
  }
}

/**
 * m_tile: 96
 * n_tile: 256
 * k_tile: 64x5
 * smem size: 220KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m96n256_k64x5_hgmma64x96_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (12 + 32) * 5 = 220 KB
  // A_smem: 96 * 64 * sizeof(T16) = 12 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 12 * 1024;

  uint32_t matA_row_id[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    int m_idx = tile_y * 96 + threadIdx.x / 8 + i * 32;
    if (m_idx < m) {
      asm("ld.global.nc.b32 %0, [%1];"
          : "=r"(matA_row_id[i])
          : "l"(matA_row_idx + m_idx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matA_row_id[i] = 0;
    }
  }

  const char* A_ldg_ptr[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    A_ldg_ptr[i] = reinterpret_cast<const char*>(A + matA_row_id[i] * k +
                                                 threadIdx.x % 8 * 8);
  }
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 8 % 8)) * 16;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t matrix_start = A_smem_addr / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4][2];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start =
        (B_smem_addr + warpgroup_id * (8 * 64 * sizeof(T16))) / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t lo = desc_lo + (i * 16 * 256 + j * 8 * 128) * sizeof(T16) / 16;
        asm("mov.b64 %0, {%1, %2};"
            : "=l"(matB_desc[i][j])
            : "r"(lo), "r"(desc_hi));
      }
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore = threadIdx.x % 8 * 8 >= first_k_tile;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      ldgsts128_zfill(A_sts_addr + i * 32 * 64 * sizeof(T16), A_ldg_ptr[i],
                      true, A_ignore);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      A_ldg_ptr[i] += first_k_tile * sizeof(T16);
    }
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 3; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        ldgsts128(
            A_sts_addr + prefetch_iter * 45056 + i * 32 * 64 * sizeof(T16),
            A_ldg_ptr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 45056 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        A_ldg_ptr[i] += 64 * sizeof(T16);
      }
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 45056 * 3;

  // C register fragment
  uint32_t C_frag[2][12][2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 12; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        C_frag[i][j][p][0] = 0;
        C_frag[i][j][p][1] = 0;
      }
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<2>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 3; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x96_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

// tile prefetch
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      ldgsts128(A_sts_addr + sts_offset + i * 32 * 64 * sizeof(T16),
                A_ldg_ptr[i], true);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 45056 * 2 ? -(45056 / 16 * 4) : 45056 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 45056 * 4 ? sts_offset + 45056 : 0;

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      A_ldg_ptr[i] += 64 * sizeof(T16);
    }
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<2>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x96_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 45056 * 2 ? -(45056 / 16 * 4) : 45056 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 45056 * 4 ? sts_offset + 45056 : 0;

    ldgsts_group_wait<2>();
    __syncthreads();
  }

  uint32_t* C_sts_ptr = reinterpret_cast<uint32_t*>(smem) +
                        threadIdx.x % 4 * (256 + 4) * 2 +
                        threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
  warpgroup_depbar_le<0>();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 12; ++j) {
      C_sts_ptr[i * 128 + j * 8 * (256 + 4)] = C_frag[i][j][0][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + (256 + 4)] = C_frag[i][j][0][1];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8] = C_frag[i][j][1][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8 + (256 + 4)] =
          C_frag[i][j][1][1];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(float) * (threadIdx.x / 64 * (256 + 4) + threadIdx.x % 64 * 4);
  float C_lds_reg[24][4];
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    lds128(C_lds_reg[i][0], C_lds_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3],
           C_lds_addr + sizeof(float) * (i * 4 * (256 + 4)));
  }

  typename T16x2<T16>::T C_stg_reg[24][2];
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    cvt_f32_to_t16(C_stg_reg[i][0], C_lds_reg[i][0], C_lds_reg[i][1]);
    cvt_f32_to_t16(C_stg_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3]);
  }

  uint32_t m_idx = tile_y * 96 + (threadIdx.x / 64);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 64) * 4;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    stg64(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_ptr + i * 4 * n,
          m_idx + i * 4 < m && n_idx < n);
  }
}

/**
 * m_tile: 64
 * n_tile: 256
 * k_tile: 64x5
 * smem size: 200KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m64n256_k64x5_hgmma64x128_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (8 + 32) * 5 = 200 KB
  // A_smem: 64 * 64 * sizeof(T16) = 8 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 8 * 1024;

  uint32_t matA_row_id[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int m_idx = tile_y * 64 + threadIdx.x / 8 + i * 32;
    if (m_idx < m) {
      asm("ld.global.nc.b32 %0, [%1];"
          : "=r"(matA_row_id[i])
          : "l"(matA_row_idx + m_idx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matA_row_id[i] = 0;
    }
  }

  const char* A_ldg_ptr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    A_ldg_ptr[i] = reinterpret_cast<const char*>(A + matA_row_id[i] * k +
                                                 (threadIdx.x % 8) * 8);
  }
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  uint32_t A_ldg_guard = 0;
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int m_idx = tile_y * 64 + threadIdx.x / 8 + i * 32;
    if (m_idx < m) {
      A_ldg_guard |= (1U << i);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 8 % 8)) * 16;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t matrix_start = A_smem_addr / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start =
        (B_smem_addr + warpgroup_id * (8 * 128 * sizeof(T16))) / 16;
    uint32_t ld = 8 * 128 / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start | (ld << 16);
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t lo = desc_lo + i * 16 * 256 * sizeof(T16) / 16;
      asm("mov.b64 %0, {%1, %2};" : "=l"(matB_desc[i]) : "r"(lo), "r"(desc_hi));
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore = threadIdx.x % 8 * 8 >= first_k_tile;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldgsts128_zfill(A_sts_addr + i * 32 * 64 * sizeof(T16), A_ldg_ptr[i],
                      true, A_ignore);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      A_ldg_ptr[i] += first_k_tile * sizeof(T16);
    }
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 3; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        ldgsts128(
            A_sts_addr + prefetch_iter * 40960 + i * 32 * 64 * sizeof(T16),
            A_ldg_ptr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 40960 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        A_ldg_ptr[i] += 64 * sizeof(T16);
      }
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 40960 * 3;

  // C register fragment
  uint32_t C_frag[16][2][2];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      C_frag[i][j][0] = 0;
      C_frag[i][j][1] = 0;
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<2>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 3; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      hgmma_64x128_f32_transB<T16>(C_frag, matA_desc[i], matB_desc[i]);
    }
    warpgroup_commit();

// tile prefetch
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldgsts128(A_sts_addr + sts_offset + i * 32 * 64 * sizeof(T16),
                A_ldg_ptr[i], true);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 40960 * 2 ? -(40960 / 16 * 4) : 40960 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matB_desc[i])
          : "r"(lds_offset));
    }

    // switch double buffer
    sts_offset = sts_offset < 40960 * 4 ? sts_offset + 40960 : 0;

// ldg pointer for next tile
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      A_ldg_ptr[i] += 64 * sizeof(T16);
    }
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<2>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      hgmma_64x128_f32_transB<T16>(C_frag, matA_desc[i], matB_desc[i]);
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 40960 * 2 ? -(40960 / 16 * 4) : 40960 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matB_desc[i])
          : "r"(lds_offset));
    }

    // switch double buffer
    sts_offset = sts_offset < 40960 * 4 ? sts_offset + 40960 : 0;

    ldgsts_group_wait<2>();
    __syncthreads();
  }

  warpgroup_depbar_le<0>();
#pragma unroll
  for (int i = 0; i < 16; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      // guide the compiler to reuse C_frag register
      cvt_f32_to_t16(reinterpret_cast<typename T16x2<T16>::T&>(C_frag[i][j][0]),
                     reinterpret_cast<const float&>(C_frag[i][j][0]),
                     reinterpret_cast<const float&>(C_frag[i][j][1]));
    }
  }

  uint32_t* C_sts_ptr =
      reinterpret_cast<uint32_t*>(smem) +
      (threadIdx.x % 128 / 32 * 16 + threadIdx.x % 32 / 4) * (128 + 4) +
      threadIdx.x / 128 * 64 + threadIdx.x % 4;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 16; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      C_sts_ptr[i * 4 + j * 8 * (128 + 4)] = C_frag[i][j][0];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(uint32_t) * (threadIdx.x / 32 * (128 + 4) + threadIdx.x % 32 * 4);
  uint32_t C_stg_reg[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lds128(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_reg[i][2], C_stg_reg[i][3],
           C_lds_addr + sizeof(uint32_t) * (i * 8 * (128 + 4)));
  }

  uint32_t m_idx = tile_y * 64 + (threadIdx.x / 32);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 32) * 8;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    stg128(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_reg[i][2], C_stg_reg[i][3],
           C_stg_ptr + i * 8 * n, m_idx + i * 8 < m && n_idx < n);
  }
}

/**
 * m_tile: 48
 * n_tile: 256
 * k_tile: 64x5
 * smem size: 190KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m48n256_k64x5_hgmma64x48_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (6 + 32) * 5 = 190 KB
  // A_smem: 48 * 64 * sizeof(T16) = 6 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 6 * 1024;

  uint32_t matA_row_id0, matA_row_id1;
  {
    int m_idx0 = tile_y * 48 + threadIdx.x / 8;
    int m_idx1 = tile_y * 48 + 32 + threadIdx.x / 16;
    if (m_idx0 < m) {
      asm("ld.global.nc.b32 %0, [%1];"
          : "=r"(matA_row_id0)
          : "l"(matA_row_idx + m_idx0));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matA_row_id0 = 0;
    }
    if (m_idx1 < m) {
      asm("ld.global.nc.b32 %0, [%1];"
          : "=r"(matA_row_id1)
          : "l"(matA_row_idx + m_idx1));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matA_row_id1 = 0;
    }
  }

  const char* A_ldg_ptr0 = reinterpret_cast<const char*>(A + matA_row_id0 * k +
                                                         (threadIdx.x % 8) * 8);
  const char* A_ldg_ptr1 = reinterpret_cast<const char*>(
      A + matA_row_id1 * k + (threadIdx.x % 16) * 4);
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  bool A_ldg_guard0 = tile_y * 48 + threadIdx.x / 8 < m;
  bool A_ldg_guard1 = tile_y * 48 + threadIdx.x / 16 + 32 < m;
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr0 =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 8 % 8)) * 16;
  uint32_t A_sts_addr1 =
      A_smem_addr + 256 * 16 + (threadIdx.x ^ (threadIdx.x / 16 % 8 * 2)) * 8;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t matrix_start = A_smem_addr / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4][2];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start =
        (B_smem_addr + warpgroup_id * (8 * 64 * sizeof(T16))) / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t lo = desc_lo + (i * 16 * 256 + j * 8 * 128) * sizeof(T16) / 16;
        asm("mov.b64 %0, {%1, %2};"
            : "=l"(matB_desc[i][j])
            : "r"(lo), "r"(desc_hi));
      }
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore0 = threadIdx.x % 8 * 8 >= first_k_tile;
    bool A_ignore1 = threadIdx.x % 16 * 4 >= first_k_tile;
    ldgsts128_zfill(A_sts_addr0, A_ldg_ptr0, A_ldg_guard0, A_ignore0);
    ldgsts64_zfill(A_sts_addr1, A_ldg_ptr1, A_ldg_guard1, A_ignore1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

    // ldg pointer for next tile
    A_ldg_ptr0 += first_k_tile * sizeof(T16);
    A_ldg_ptr1 += first_k_tile * sizeof(T16);
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 3; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
      ldgsts128(A_sts_addr0 + prefetch_iter * 38912, A_ldg_ptr0, A_ldg_guard0);
      ldgsts64(A_sts_addr1 + prefetch_iter * 38912, A_ldg_ptr1, A_ldg_guard1);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 38912 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
      A_ldg_ptr0 += 64 * sizeof(T16);
      A_ldg_ptr1 += 64 * sizeof(T16);
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 38912 * 3;

  // C register fragment
  uint32_t C_frag[2][6][2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 6; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        C_frag[i][j][p][0] = 0;
        C_frag[i][j][p][1] = 0;
      }
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<2>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 3; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x48_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // tile prefetch
    ldgsts128(A_sts_addr0 + sts_offset, A_ldg_ptr0, A_ldg_guard0);
    ldgsts64(A_sts_addr1 + sts_offset, A_ldg_ptr1, A_ldg_guard1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 38912 * 2 ? -(38912 / 16 * 4) : 38912 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 38912 * 4 ? sts_offset + 38912 : 0;

    // ldg pointer for next tile
    A_ldg_ptr0 += 64 * sizeof(T16);
    A_ldg_ptr1 += 64 * sizeof(T16);
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<2>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x48_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 38912 * 2 ? -(38912 / 16 * 4) : 38912 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 38912 * 4 ? sts_offset + 38912 : 0;

    ldgsts_group_wait<2>();
    __syncthreads();
  }

  uint32_t* C_sts_ptr = reinterpret_cast<uint32_t*>(smem) +
                        threadIdx.x % 4 * (256 + 4) * 2 +
                        threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
  warpgroup_depbar_le<0>();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      C_sts_ptr[i * 128 + j * 8 * (256 + 4)] = C_frag[i][j][0][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + (256 + 4)] = C_frag[i][j][0][1];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8] = C_frag[i][j][1][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8 + (256 + 4)] =
          C_frag[i][j][1][1];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(float) * (threadIdx.x / 64 * (256 + 4) + threadIdx.x % 64 * 4);
  float C_lds_reg[12][4];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    lds128(C_lds_reg[i][0], C_lds_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3],
           C_lds_addr + sizeof(float) * (i * 4 * (256 + 4)));
  }

  typename T16x2<T16>::T C_stg_reg[12][2];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    cvt_f32_to_t16(C_stg_reg[i][0], C_lds_reg[i][0], C_lds_reg[i][1]);
    cvt_f32_to_t16(C_stg_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3]);
  }

  uint32_t m_idx = tile_y * 48 + (threadIdx.x / 64);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 64) * 4;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    stg64(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_ptr + i * 4 * n,
          m_idx + i * 4 < m && n_idx < n);
  }
}

/**
 * m_tile: 32
 * n_tile: 256
 * k_tile: 64x6
 * smem size: 216KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m32n256_k64x6_hgmma64x32_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (4 + 32) * 6 = 216 KB
  // A_smem: 32 * 64 * sizeof(T16) = 4 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 4 * 1024;

  uint32_t matA_row_id;
  if (tile_y * 32 + threadIdx.x / 8 < m) {
    asm("ld.global.nc.b32 %0, [%1];"
        : "=r"(matA_row_id)
        : "l"(matA_row_idx + tile_y * 32 + threadIdx.x / 8));
  } else {
    // map the out-of-bound threads to row0 of matrixA,
    // to avoid predicated ld instructions
    matA_row_id = 0;
  }

  const char* A_ldg_ptr = reinterpret_cast<const char*>(A + matA_row_id * k +
                                                        (threadIdx.x % 8) * 8);
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 8 % 8)) * 16;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t matrix_start = A_smem_addr / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4][2];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start =
        (B_smem_addr + warpgroup_id * (8 * 64 * sizeof(T16))) / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t lo = desc_lo + (i * 16 * 256 + j * 8 * 128) * sizeof(T16) / 16;
        asm("mov.b64 %0, {%1, %2};"
            : "=l"(matB_desc[i][j])
            : "r"(lo), "r"(desc_hi));
      }
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore = threadIdx.x % 8 * 8 >= first_k_tile;
    ldgsts128_zfill(A_sts_addr, A_ldg_ptr, true, A_ignore);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

    // ldg pointer for next tile
    A_ldg_ptr += first_k_tile * sizeof(T16);
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 4; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
      ldgsts128(A_sts_addr + prefetch_iter * 36864, A_ldg_ptr, true);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 36864 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
      A_ldg_ptr += 64 * sizeof(T16);
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 36864 * 4;

  // C register fragment
  uint32_t C_frag[2][4][2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        C_frag[i][j][p][0] = 0;
        C_frag[i][j][p][1] = 0;
      }
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<3>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 4; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x32_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // tile prefetch
    ldgsts128(A_sts_addr + sts_offset, A_ldg_ptr, true);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 36864 * 3 ? -(36864 / 16 * 5) : 36864 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 36864 * 5 ? sts_offset + 36864 : 0;

    // ldg pointer for next tile
    A_ldg_ptr += 64 * sizeof(T16);
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<3>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x32_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 36864 * 3 ? -(36864 / 16 * 5) : 36864 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 36864 * 5 ? sts_offset + 36864 : 0;

    ldgsts_group_wait<3>();
    __syncthreads();
  }

  uint32_t* C_sts_ptr = reinterpret_cast<uint32_t*>(smem) +
                        threadIdx.x % 4 * (256 + 4) * 2 +
                        threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
  warpgroup_depbar_le<0>();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      C_sts_ptr[i * 128 + j * 8 * (256 + 4)] = C_frag[i][j][0][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + (256 + 4)] = C_frag[i][j][0][1];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8] = C_frag[i][j][1][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8 + (256 + 4)] =
          C_frag[i][j][1][1];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(float) * (threadIdx.x / 64 * (256 + 4) + threadIdx.x % 64 * 4);
  float C_lds_reg[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lds128(C_lds_reg[i][0], C_lds_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3],
           C_lds_addr + sizeof(float) * (i * 4 * (256 + 4)));
  }

  typename T16x2<T16>::T C_stg_reg[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    cvt_f32_to_t16(C_stg_reg[i][0], C_lds_reg[i][0], C_lds_reg[i][1]);
    cvt_f32_to_t16(C_stg_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3]);
  }

  uint32_t m_idx = tile_y * 32 + (threadIdx.x / 64);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 64) * 4;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    stg64(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_ptr + i * 4 * n,
          m_idx + i * 4 < m && n_idx < n);
  }
}

/**
 * m_tile: 16
 * n_tile: 256
 * k_tile: 64x6
 * smem size: 204KB
 */
template <typename T16>
__device__ __forceinline__ void hgemm_f32_m16n256_k64x6_hgmma64x16_ldg8_loop(
    const T16* A, const T16* B, const uint32_t* matA_row_idx, T16* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tile_x, const uint32_t& tile_y, const uint32_t& B_ldg_step,
    const uint32_t& B_tile_step) {
  // (2 + 32) * 6 = 204 KB
  // A_smem: 16 * 64 * sizeof(T16) = 2 KB
  // B_smem: 64 * 256 * sizeof(T16) = 32 KB
  uint32_t A_smem_addr = smem_u32addr(smem);
  uint32_t B_smem_addr = A_smem_addr + 2 * 1024;

  uint32_t matA_row_id;
  if (tile_y * 16 + threadIdx.x / 16 < m) {
    asm("ld.global.nc.b32 %0, [%1];"
        : "=r"(matA_row_id)
        : "l"(matA_row_idx + tile_y * 16 + threadIdx.x / 16));
  } else {
    // map the out-of-bound threads to row0 of matrixA,
    // to avoid predicated ld instructions
    matA_row_id = 0;
  }

  const char* A_ldg_ptr = reinterpret_cast<const char*>(A + matA_row_id * k +
                                                        (threadIdx.x % 16) * 4);
  const char* B_ldg_ptr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tile_x * 256 + (threadIdx.x % 8) * 8);

  // ldg_guard to avoid LDG out of bound
  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = tile_x * 256 + threadIdx.x % 8 * 8 + i * 8 * 8;
    if (n_idx < n) {
      B_ldg_guard |= (1U << i);
    }
  }

  uint32_t A_sts_addr =
      A_smem_addr + (threadIdx.x ^ (threadIdx.x / 16 % 8 * 2)) * 8;
  uint32_t B_sts_addr = B_smem_addr +
                        (threadIdx.x / 64) * 8 * 256 * sizeof(T16) +
                        ((threadIdx.x % 64) ^ (threadIdx.x / 8 % 8)) * 16;

  // matrix descriptor of GMMA
  uint64_t matA_desc[4];
  {
    uint32_t matrix_start = A_smem_addr / 16;
    uint32_t stride = 8 * 64 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("mov.b64 %0, {%1, %2};"
          : "=l"(matA_desc[i])
          : "r"(desc_lo + i * 2), "r"(desc_hi));
    }
  }
  uint64_t matB_desc[4][2];
  {
    uint32_t warpgroup_id = threadIdx.x / 128;
    uint32_t matrix_start =
        (B_smem_addr + warpgroup_id * (8 * 64 * sizeof(T16))) / 16;
    uint32_t stride = 8 * 256 * sizeof(T16) / 16;
    uint32_t swzl_mode = 1;
    uint32_t desc_lo = matrix_start;
    uint32_t desc_hi = stride | (swzl_mode << 30);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t lo = desc_lo + (i * 16 * 256 + j * 8 * 128) * sizeof(T16) / 16;
        asm("mov.b64 %0, {%1, %2};"
            : "=l"(matB_desc[i][j])
            : "r"(lo), "r"(desc_hi));
      }
    }
  }

  uint32_t k_tiles = (k + 63) / 64;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - (k_tiles * 64 - 64);
    bool A_ignore = threadIdx.x % 16 * 4 >= first_k_tile;
    ldgsts64_zfill(A_sts_addr, A_ldg_ptr, true, A_ignore);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bool B_ignore = threadIdx.x / 8 + i * 32 >= first_k_tile;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128_zfill(B_sts_addr + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                        B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                        (B_ldg_guard & (1U << j)) != 0, B_ignore);
      }
    }
    ldgsts_group_commit();

    // ldg pointer for next tile
    A_ldg_ptr += first_k_tile * sizeof(T16);
    B_ldg_ptr += first_k_tile * n * sizeof(T16);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetch_iter = 1; prefetch_iter < 4; ++prefetch_iter) {
    if (prefetch_iter < k_tiles) {
      ldgsts64(A_sts_addr + prefetch_iter * 34816, A_ldg_ptr, true);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldgsts128(B_sts_addr + prefetch_iter * 34816 +
                        (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
                    B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
                    (B_ldg_guard & (1U << j)) != 0);
        }
      }
      A_ldg_ptr += 64 * sizeof(T16);
      B_ldg_ptr += B_tile_step;
    }
    ldgsts_group_commit();
  }

  // smem double buffer offset
  uint32_t sts_offset = 34816 * 4;

  // C register fragment
  uint32_t C_frag[2][2][2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        C_frag[i][j][p][0] = 0;
        C_frag[i][j][p][1] = 0;
      }
    }
  }

  // wait for the 1'st tile
  ldgsts_group_wait<3>();
  __syncthreads();

// k_tiles loop
#pragma unroll 1
  for (; k_tiles > 4; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x16_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // tile prefetch
    ldgsts64(A_sts_addr + sts_offset, A_ldg_ptr, true);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldgsts128(
            B_sts_addr + sts_offset + (i * 32 * 256 + j * 8 * 64) * sizeof(T16),
            B_ldg_ptr + i * B_ldg_step + j * 64 * sizeof(T16),
            (B_ldg_guard & (1U << j)) != 0);
      }
    }
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 34816 * 3 ? -(34816 / 16 * 5) : 34816 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 34816 * 5 ? sts_offset + 34816 : 0;

    // ldg pointer for next tile
    A_ldg_ptr += 64 * sizeof(T16);
    B_ldg_ptr += B_tile_step;

    warpgroup_depbar_le<1>();
    ldgsts_group_wait<3>();
    __syncthreads();
  }

// k_tiles loop without prefetch
#pragma unroll 1
  for (; k_tiles > 0; --k_tiles) {
    // HGMMA loop
    warpgroup_arrive();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        hgmma_64x16_f32_transA<T16>(C_frag[j], matB_desc[i][j], matA_desc[i]);
      }
    }
    warpgroup_commit();

    // dummy ldgsts group commit to make ldgsts_group_wait<2> work
    ldgsts_group_commit();

    // update matrix descriptor
    uint32_t lds_offset =
        sts_offset == 34816 * 3 ? -(34816 / 16 * 5) : 34816 / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("{.reg .b32 lo, hi;\n"
          " mov.b64 {lo, hi}, %0;\n"
          " add.s32 lo, lo, %1;\n"
          " mov.b64 %0, {lo, hi};}"
          : "+l"(matA_desc[i])
          : "r"(lds_offset));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        asm("{.reg .b32 lo, hi;\n"
            " mov.b64 {lo, hi}, %0;\n"
            " add.s32 lo, lo, %1;\n"
            " mov.b64 %0, {lo, hi};}"
            : "+l"(matB_desc[i][j])
            : "r"(lds_offset));
      }
    }

    // switch double buffer
    sts_offset = sts_offset < 34816 * 5 ? sts_offset + 34816 : 0;

    ldgsts_group_wait<3>();
    __syncthreads();
  }

  uint32_t* C_sts_ptr = reinterpret_cast<uint32_t*>(smem) +
                        threadIdx.x % 4 * (256 + 4) * 2 +
                        threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
  warpgroup_depbar_le<0>();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      C_sts_ptr[i * 128 + j * 8 * (256 + 4)] = C_frag[i][j][0][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + (256 + 4)] = C_frag[i][j][0][1];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8] = C_frag[i][j][1][0];
      C_sts_ptr[i * 128 + j * 8 * (256 + 4) + 8 + (256 + 4)] =
          C_frag[i][j][1][1];
    }
  }
  __syncthreads();

  uint32_t C_lds_addr =
      smem_u32addr(smem) +
      sizeof(float) * (threadIdx.x / 64 * (256 + 4) + threadIdx.x % 64 * 4);
  float C_lds_reg[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lds128(C_lds_reg[i][0], C_lds_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3],
           C_lds_addr + sizeof(float) * (i * 4 * (256 + 4)));
  }

  typename T16x2<T16>::T C_stg_reg[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    cvt_f32_to_t16(C_stg_reg[i][0], C_lds_reg[i][0], C_lds_reg[i][1]);
    cvt_f32_to_t16(C_stg_reg[i][1], C_lds_reg[i][2], C_lds_reg[i][3]);
  }

  uint32_t m_idx = tile_y * 16 + (threadIdx.x / 64);
  uint32_t n_idx = tile_x * 256 + (threadIdx.x % 64) * 4;
  T16* C_stg_ptr = C + m_idx * n + n_idx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    stg64(C_stg_reg[i][0], C_stg_reg[i][1], C_stg_ptr + i * 4 * n,
          m_idx + i * 4 < m && n_idx < n);
  }
}

template <typename T16, int MAX_NMATB>
__global__ void hgemm_f32_n256k64_hgmma_ldg8_kernel(
    const T16* A, const T16* B, T16* C,
    const uint32_t* ctaIdBarriers,  // {MAX_NMATB}
    const BatchInfo* batchInfos,    // {nMatB}
    const TileRemap* tileRemaps,    // {nMatB}
    const uint32_t* matARowIdx,     // {nMatB, matARows}
    U32DivMod gridXDivMod, uint32_t grid, uint32_t matARows, uint32_t n,
    uint32_t k,
    uint32_t BLdgStep,     // 32 * n * sizeof(T16)
    uint32_t BTileStep) {  // 64 * n * sizeof(T16)
  /**
   * CTA Tile Configuration:
   *
   * n_tile: 256
   * ------------------------------------------------
   * m_tile 128: k_tile 64x4, warpgroup tile 64x256, hgmma tile 64x256, 2x1
   * warpgroups m_tile 96: k_tile 64x5, warpgroup tile 96x128, hgmma tile 64x96,
   * 1x2 warpgroups m_tile 64: k_tile 64x5, warpgroup tile 64x128, hgmma tile
   * 64x128, 1x2 warpgroups m_tile 48: k_tile 64x5, warpgroup tile 48x128, hgmma
   * tile 64x48, 1x2 warpgroups m_tile 32: k_tile 64x6, warpgroup tile 32x128,
   * hgmma tile 64x32, 1x2 warpgroups m_tile 16: k_tile 64x6, warpgroup tile
   * 16x128, hgmma tile 64x16, 1x2 warpgroups
   */

  // 220 KB shared memory
  extern __shared__ char smem[];

  uint32_t laneId = threadIdx.x % 32;
  uint32_t ctaIdBarrier[MAX_NMATB / 32];
#pragma unroll
  for (int i = 0; i < MAX_NMATB / 32; ++i) {
    asm("ld.global.cg.b32 %0, [%1];"
        : "=r"(ctaIdBarrier[i])
        : "l"(ctaIdBarriers + laneId + i * 32));
  }
  uint32_t totalCta =
      __shfl_sync(0xffffffff, ctaIdBarrier[MAX_NMATB / 32 - 1], 31, 32);

#pragma unroll 1
  for (uint32_t vCtaId = blockIdx.x; vCtaId < totalCta; vCtaId += grid) {
    uint32_t ctaIdZ = 0;
#pragma unroll
    for (int i = 0; i < MAX_NMATB / 32; ++i) {
      asm("{.reg .pred p0;\n"
          " .reg .b32 r0;\n"
          " setp.ge.u32 p0, %1, %2;\n"
          " vote.sync.ballot.b32 r0, p0, 0xffffffff;\n"
          " popc.b32 r0, r0;\n"
          " add.u32 %0, %0, r0;}\n"
          : "+r"(ctaIdZ)
          : "r"(vCtaId), "r"(ctaIdBarrier[i]));
    }

    // GEMM tile info
    BatchInfo batchInfo;
    asm("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(reinterpret_cast<int4&>(batchInfo).x),
          "=r"(reinterpret_cast<int4&>(batchInfo).y),
          "=r"(reinterpret_cast<int4&>(batchInfo).z),
          "=r"(reinterpret_cast<int4&>(batchInfo).w)
        : "l"(batchInfos + ctaIdZ));
    uint32_t batchId = batchInfo.batchId;
    uint32_t m = batchInfo.m;
    uint32_t COffset = batchInfo.COffset;
    uint32_t ctaIdXY = vCtaId - batchInfo.ctaOffset;

    __syncthreads();  // solve shared memory WAR conflict

    if (m > 96) {
      // m_tile 128, n_tile 256, k_tile 64x4, warpgroup tile 64x256

      // tile remap to improve L2 hit rate
      TileRemap tileRemap;  // 32 Byte = 8 regs
      uint32_t* tileRemapReg = reinterpret_cast<uint32_t*>(&tileRemap);
      asm("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%8];\n"
          "ld.global.nc.v4.b32 {%4, %5, %6, %7}, [%8 + 16];"
          : "=r"(tileRemapReg[0]), "=r"(tileRemapReg[1]), "=r"(tileRemapReg[2]),
            "=r"(tileRemapReg[3]), "=r"(tileRemapReg[4]), "=r"(tileRemapReg[5]),
            "=r"(tileRemapReg[6]), "=r"(tileRemapReg[7])
          : "l"(tileRemaps + ctaIdZ));
      uint32_t ctaIdX, ctaIdY;
      tileRemap.remap(ctaIdXY, &ctaIdX, &ctaIdY);

      hgemm_f32_m128n256_k64x4_hgmma64x256_ldg8_loop<T16>(
          A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
          smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
    } else {
      auto gridXDm = gridXDivMod.divmod(ctaIdXY);
      uint32_t ctaIdX = gridXDm.mod;
      uint32_t ctaIdY = gridXDm.div;

      if (m > 64) {
        // m_tile 96, n_tile 256, k_tile 64x5, warpgroup tile 96x128
        hgemm_f32_m96n256_k64x5_hgmma64x96_ldg8_loop<T16>(
            A, B + batchId * k * n, matARowIdx + batchId * matARows,
            C + COffset, smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
      } else if (m > 48) {
        // m_tile 64, n_tile 256, k_tile 64x5, warpgroup tile 64x128
        hgemm_f32_m64n256_k64x5_hgmma64x128_ldg8_loop<T16>(
            A, B + batchId * k * n, matARowIdx + batchId * matARows,
            C + COffset, smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
      } else if (m > 32) {
        // m_tile 48, n_tile 256, k_tile 64x5, warpgroup tile 48x128
        hgemm_f32_m48n256_k64x5_hgmma64x48_ldg8_loop<T16>(
            A, B + batchId * k * n, matARowIdx + batchId * matARows,
            C + COffset, smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
      } else if (m > 16) {
        // m_tile 32, n_tile 256, k_tile 64x6, warpgroup tile 32x128
        hgemm_f32_m32n256_k64x6_hgmma64x32_ldg8_loop<T16>(
            A, B + batchId * k * n, matARowIdx + batchId * matARows,
            C + COffset, smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
      } else {
        // m_tile 16, n_tile 256, k_tile 64x6, warpgroup tile 16x128
        hgemm_f32_m16n256_k64x6_hgmma64x16_ldg8_loop<T16>(
            A, B + batchId * k * n, matARowIdx + batchId * matARows,
            C + COffset, smem, m, n, k, ctaIdX, ctaIdY, BLdgStep, BTileStep);
      }
    }
  }
}

template <int CTA>
__global__ void matA_row_idx_kernel(
    const uint32_t* matBIndices, uint32_t* matARowIndices,
    uint32_t* batchedGemmM, uint32_t* matCRowBatchOffset,
    uint32_t size,              // m * nMatBPerMatARow
    uint32_t matARowIdxRShift,  // log(nMatBPerMatARow)
    uint32_t m, uint32_t nMatB) {
  __shared__ uint32_t smem[CTA];
  smem[threadIdx.x] = 0;
  __syncthreads();

  uint32_t matARowIdx, matBIdx, stgOffset;
  uint32_t idx = blockIdx.x * CTA + threadIdx.x;
  if (idx < size) {
    matARowIdx = idx >> matARowIdxRShift;
    matBIdx = matBIndices[idx];
    if (matBIdx != 0xffffffff) {
      stgOffset = atomicAdd(smem + matBIdx, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x < nMatB) {
    int ctaMatBCount = smem[threadIdx.x];
    if (ctaMatBCount != 0) {
      smem[threadIdx.x] = atomicAdd(batchedGemmM + threadIdx.x, ctaMatBCount);
    }
  }
  __syncthreads();

  if (idx < size) {
    if (matBIdx != 0xffffffff) {
      stgOffset += smem[matBIdx];
      matCRowBatchOffset[idx] = stgOffset;
      matARowIndices[matBIdx * m + stgOffset] = matARowIdx;
    } else {
      matCRowBatchOffset[idx] = 0xffffffff;
    }
  }
}

template <int CTA, int MAX_NMATB>
__global__ void prepare_moe_parameters_kernel(
    const uint32_t* matBIndices,  // {m, nMatBPerMatARow}
    uint32_t* matCRowIndices,     // {m, nMatBPerMatARow}, input/output
    uint32_t* batchedGemmM,       // {nMatB + 1}, input/workspace
    uint32_t* ctaIdBarriers,      // {MAX_NMATB}
    BatchInfo* batchInfos,        // {nMatB}
    TileRemap* tileRemaps,        // {nMatB}
    uint32_t nMatB, uint32_t vGridX, uint32_t n, float tilesPerWave,
    uint32_t size) {  // m * nMatBPerMatARow
  __shared__ uint32_t smem0[32];
  __shared__ uint32_t smem1[32];
  __shared__ uint32_t __align__(8) ctaIdBarrierBuffer[2];

  if (blockIdx.x == 0) {
    // CTA0: prepare GEMM m prefix sum and batched GEMM kernel
    // parameters (ctaIdBarriers, batchInfos, tileRemaps)
    uint32_t m = threadIdx.x < nMatB ? batchedGemmM[threadIdx.x] : 0;

    // m exclusive prefix sum
    uint32_t mPrefixSum = m;
#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      warp_prefix_scan(0xffffffff, mPrefixSum, i, 32);
    }
    uint32_t laneId = threadIdx.x % 32;
    uint32_t warpId = threadIdx.x / 32;
    if (laneId == 31) {
      smem0[warpId] = mPrefixSum;
    }
    __syncthreads();

    if (warpId == 0) {
      uint32_t ctaMSum = 0, warpMPrefix = 0;
#pragma unroll
      for (int i = 0; i < (MAX_NMATB + 31) / 32; ++i) {
        if (laneId == i) {
          warpMPrefix = ctaMSum;
        }
        ctaMSum += smem0[i];
      }
      smem0[laneId] = warpMPrefix;
    }
    __syncthreads();
    uint32_t mExclusivePrefixSum = mPrefixSum + smem0[warpId] - m;

    if (threadIdx.x < nMatB) {
      asm volatile(
          "st.global.cg.b32 [%0], %1;\n"
          "membar.gl;"
          :
          : "l"(batchedGemmM + threadIdx.x), "r"(mExclusivePrefixSum));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int READY_FLAG = 1;
      asm volatile(
          "{.reg .b32 r0;\n"
          " mov.b32 r0, %1;\n"
          " st.global.cg.b32 [%0], r0;}"
          :
          : "l"(batchedGemmM + nMatB), "n"(READY_FLAG));
    }

    // pseudo code:
    // tileY = m > 96 ? 128 :
    //         m > 64 ? 96 :
    //         m > 48 ? 64 :
    //         m > 32 ? 48 :
    //         m > 16 ? 32 : 16;
    uint32_t tileY;
    asm("{.reg .pred p0, p1, p2, p3, p4;\n"
        " .reg .b32 r;\n"
        " setp.le.u32 p0, %1, 96;\n"
        " setp.le.u32 p1, %1, 64;\n"
        " setp.le.u32 p2, %1, 48;\n"
        " setp.le.u32 p3, %1, 32;\n"
        " setp.le.u32 p4, %1, 16;\n"
        " mov.u32 r, 128;\n"
        " selp.u32 r, 96, r, p0;\n"
        " selp.u32 r, 64, r, p1;\n"
        " selp.u32 r, 48, r, p2;\n"
        " selp.u32 r, 32, r, p3;\n"
        " selp.u32 r, 16, r, p4;\n"
        " mov.u32 %0, r;}"
        : "=r"(tileY)
        : "r"(m));
    uint32_t vGridY = (m + tileY - 1) / tileY;
    uint32_t vGrid = vGridX * vGridY;

    // vGrid and validGemm prefix sum
    uint32_t validGemm = m > 0 ? 1 : 0;
    uint32_t vGridPrefix = vGrid;
    uint32_t validGemmPrefix = validGemm;
#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      warp_prefix_scan(0xffffffff, vGridPrefix, i, 32);
      warp_prefix_scan(0xffffffff, validGemmPrefix, i, 32);
    }
    if (laneId == 31) {
      smem0[warpId] = vGridPrefix;
      smem1[warpId] = validGemmPrefix;
    }
    __syncthreads();

    if (warpId == 0) {
      uint32_t ctaVGridSum = 0, warpVGridPrefix = 0;
      uint32_t ctaValidGemmSum = 0, warpValidGemmPrefix = 0;
#pragma unroll
      for (int i = 0; i < (MAX_NMATB + 31) / 32; ++i) {
        if (laneId == i) {
          warpVGridPrefix = ctaVGridSum;
          warpValidGemmPrefix = ctaValidGemmSum;
        }
        ctaVGridSum += smem0[i];
        ctaValidGemmSum += smem1[i];
      }
      smem0[laneId] = warpVGridPrefix;
      smem1[laneId] = warpValidGemmPrefix;
      ctaIdBarrierBuffer[0] = ctaVGridSum;
      ctaIdBarrierBuffer[1] = ctaValidGemmSum;
    }
    __syncthreads();

    if (m != 0) {
      uint32_t vGridInclusivePrefixSum = vGridPrefix + smem0[warpId];
      uint32_t batchInfoIdx = validGemmPrefix + smem1[warpId] - validGemm;
      BatchInfo batchInfo;
      batchInfo.batchId = threadIdx.x;
      batchInfo.m = m;
      batchInfo.ctaOffset = vGridInclusivePrefixSum - vGrid;
      batchInfo.COffset = mExclusivePrefixSum * n;
      TileRemap tileRemap(vGridY, tileY, vGridX, 256, tilesPerWave);

      ctaIdBarriers[batchInfoIdx] = vGridInclusivePrefixSum;
      batchInfos[batchInfoIdx] = batchInfo;
      tileRemaps[batchInfoIdx] = tileRemap;
    }

    uint32_t ctaIdPaddingVal = ctaIdBarrierBuffer[0];
    uint32_t ctaIdPaddingIdx = ctaIdBarrierBuffer[1];
    if (threadIdx.x >= ctaIdPaddingIdx && threadIdx.x < MAX_NMATB) {
      ctaIdBarriers[threadIdx.x] = ctaIdPaddingVal;
    }
  } else {
    // CTA1~...: update matCRowIndices (matCRowBatchOffset + gemmMPrefixSum)
    const uint32_t* gemmMPrefixSum = batchedGemmM;

    // waiting for the gemmMPrefixSum readyy
    const uint32_t* flagPtr = gemmMPrefixSum + nMatB;
    uint32_t gemmMPrefixReady;
    do {
      asm volatile(
          "nanosleep.u32 100;"
          "ld.global.cg.b32 %0, [%1];"
          : "=r"(gemmMPrefixReady)
          : "l"(flagPtr));
    } while (gemmMPrefixReady == 0);

    uint32_t idx = (blockIdx.x - 1) * CTA + threadIdx.x;
    if (idx >= size) {
      return;
    }

    uint32_t matBIdx = matBIndices[idx];
    if (matBIdx != 0xffffffff) {
      asm volatile(
          "{.reg .b32 r0, r1;\n"
          " .reg .b64 r2;\n"
          " ld.global.cg.b32 r0, [%2];\n"
          " cvta.to.global.u64 r2, %1;\n"
          " mad.wide.u32 r2, %0, %3, r2;\n"
          " ld.global.nc.b32 r1, [r2];\n"
          " add.u32 r0, r0, r1;\n"
          " st.global.b32 [%2], r0;}"
          :
          : "r"(matBIdx), "l"(gemmMPrefixSum), "l"(matCRowIndices + idx),
            "n"(sizeof(uint32_t)));
    }
  }
}

template <int MAX_NMATB>
void MoeBatchedGemmPreprocess(
    const uint32_t* matBIndices,  // {m, nMatBPerMatARow}
    uint32_t* matARowIndices,     // {nMatB, m}
    uint32_t* matCRowIndices,     // {m, nMatBPerMatARow}
    uint32_t* batchedGemmM,       // {nMatB + 1}, workspace
    uint32_t* ctaIdBarriers,      // {MAX_NMATB}
    BatchInfo* batchInfos,        // {nMatB}
    TileRemap* tileRemaps,        // {nMatB}
    uint32_t m, uint32_t n, uint32_t nMatB, uint32_t nMatBPerMatARow,
    uint32_t vGridX, uint32_t tilesPerWave, cudaStream_t stream) {
  const int CTA = 256;
  static_assert(MAX_NMATB <= CTA, "");
  cudaMemsetAsync(batchedGemmM, 0, (nMatB + 1) * sizeof(uint32_t), stream);

  uint32_t size = m * nMatBPerMatARow;
  int grid = (size + CTA - 1) / CTA;

  // matA row index preprocess
  {
    if ((nMatBPerMatARow & (nMatBPerMatARow - 1)) != 0) {
      // nMatBPerMatARow must be power of 2
      return;
    }

#ifdef __GNUC__
    uint32_t matARowIdxRShift = __builtin_ctz(nMatBPerMatARow);
#else
    uint32_t matARowIdxRShift;
    for (int i = 0; i < 32; ++i) {
      if (nMatBPerMatARow >> i == 1) {
        matARowIdxRShift = i;
      }
    }
#endif
    matA_row_idx_kernel<CTA><<<grid, CTA, 0, stream>>>(
        matBIndices, matARowIndices, batchedGemmM, matCRowIndices, size,
        matARowIdxRShift, m, nMatB);
  }

  // prepare GEMM kernel parameters and update matCRowIndices
  prepare_moe_parameters_kernel<CTA, MAX_NMATB><<<grid + 1, CTA, 0, stream>>>(
      matBIndices, matCRowIndices, batchedGemmM, ctaIdBarriers, batchInfos,
      tileRemaps, nMatB, vGridX, n, tilesPerWave, size);
}

// 128Byte aligned memory size
size_t AlignedMemSize(size_t requestedSize) {
  return (requestedSize + 127) / 128 * 128;
}

template <int MAX_NMATB>
size_t GetWorkspaceSize(uint32_t matARows, uint32_t nMatB) {
  size_t ctaIdBarrierSize = AlignedMemSize(MAX_NMATB * sizeof(uint32_t));
  size_t batchInfoSize = AlignedMemSize(nMatB * sizeof(BatchInfo));
  size_t tileRemapSize = AlignedMemSize(nMatB * sizeof(TileRemap));
  size_t batchedGemmMSize = AlignedMemSize((nMatB + 1) * sizeof(uint32_t));
  size_t matARowIdxSize = AlignedMemSize(nMatB * matARows * sizeof(uint32_t));

  size_t ws_size = ctaIdBarrierSize + batchInfoSize + tileRemapSize +
                   batchedGemmMSize + matARowIdxSize;
  return ws_size;
}

template <typename T16, int MAX_NMATB>
void MoeBatchedGemm(const T16* A, const T16* B, const uint32_t* matBIndices,
                    T16* C, uint32_t* matCRowIndices, void* ws, size_t wsSize,
                    uint32_t matARows, uint32_t n, uint32_t k, uint32_t nMatB,
                    uint32_t nMatBPerMatARow, cudaStream_t stream) {
  if (nMatB > MAX_NMATB) {
    // invalid nMatB
    return;
  }

  size_t ctaIdBarrierSize = AlignedMemSize(MAX_NMATB * sizeof(uint32_t));
  size_t batchInfoSize = AlignedMemSize(nMatB * sizeof(BatchInfo));
  size_t tileRemapSize = AlignedMemSize(nMatB * sizeof(TileRemap));
  size_t batchedGemmMSize = AlignedMemSize((nMatB + 1) * sizeof(uint32_t));
  size_t matARowIdxSize = AlignedMemSize(nMatB * matARows * sizeof(uint32_t));
  if (wsSize < ctaIdBarrierSize + batchInfoSize + tileRemapSize +
                   batchedGemmMSize + matARowIdxSize) {
    // invalid workspace size
    return;
  }

  char* wsPtr = static_cast<char*>(ws);
  uint32_t* ctaIdBarriers = reinterpret_cast<uint32_t*>(wsPtr);
  BatchInfo* batchInfos =
      reinterpret_cast<BatchInfo*>(wsPtr + ctaIdBarrierSize);
  TileRemap* tileRemaps =
      reinterpret_cast<TileRemap*>(wsPtr + ctaIdBarrierSize + batchInfoSize);
  uint32_t* batchedGemmM = reinterpret_cast<uint32_t*>(
      wsPtr + ctaIdBarrierSize + batchInfoSize + tileRemapSize);
  uint32_t* matARowIdx =
      reinterpret_cast<uint32_t*>(wsPtr + ctaIdBarrierSize + batchInfoSize +
                                  tileRemapSize + batchedGemmMSize);

  int deviceId, smCount;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId);
  uint32_t tilesPerWave = smCount;  // 1 cta per sm for GEMM kernel
  uint32_t gridX = (n + 255) / 256;

  MoeBatchedGemmPreprocess<MAX_NMATB>(
      matBIndices, matARowIdx, matCRowIndices, batchedGemmM, ctaIdBarriers,
      batchInfos, tileRemaps, matARows, n, nMatB, nMatBPerMatARow, gridX,
      tilesPerWave, stream);

  uint32_t smemSize = 220 * 1024;
  cudaFuncSetAttribute(hgemm_f32_n256k64_hgmma_ldg8_kernel<T16, MAX_NMATB>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);
  // the max m-tile is 128
  uint32_t gridY =
      (matARows * nMatBPerMatARow + 127) / 128 + matARows * nMatBPerMatARow <
              nMatB
          ? matARows * nMatBPerMatARow - 1
          : nMatB - 1;
  uint32_t grid = gridY * gridX < tilesPerWave ? gridY * gridX : tilesPerWave;
  static_assert(MAX_NMATB % 32 == 0, "");
  hgemm_f32_n256k64_hgmma_ldg8_kernel<T16, MAX_NMATB>
      <<<grid, 256, smemSize, stream>>>(
          A, B, C, ctaIdBarriers, batchInfos, tileRemaps, matARowIdx,
          U32DivMod(gridX), grid, matARows, n, k, 32 * n * sizeof(T16),
          64 * n * sizeof(T16));
}
template void MoeBatchedGemm<half, 64>(const half* A, const half* B,
                                       const uint32_t* matBIndices, half* C,
                                       uint32_t* matCRowIndices, void* deviceWs,
                                       size_t deviceWsSize, uint32_t matARows,
                                       uint32_t n, uint32_t k, uint32_t nMatB,
                                       uint32_t nMatBPerMatARow,
                                       cudaStream_t stream);
template void MoeBatchedGemm<half, 128>(
    const half* A, const half* B, const uint32_t* matBIndices, half* C,
    uint32_t* matCRowIndices, void* deviceWs, size_t deviceWsSize,
    uint32_t matARows, uint32_t n, uint32_t k, uint32_t nMatB,
    uint32_t nMatBPerMatARow, cudaStream_t stream);
template void MoeBatchedGemm<half, 256>(
    const half* A, const half* B, const uint32_t* matBIndices, half* C,
    uint32_t* matCRowIndices, void* deviceWs, size_t deviceWsSize,
    uint32_t matARows, uint32_t n, uint32_t k, uint32_t nMatB,
    uint32_t nMatBPerMatARow, cudaStream_t stream);
template void MoeBatchedGemm<__nv_bfloat16, 64>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, const uint32_t* matBIndices,
    __nv_bfloat16* C, uint32_t* matCRowIndices, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream);
template void MoeBatchedGemm<__nv_bfloat16, 128>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, const uint32_t* matBIndices,
    __nv_bfloat16* C, uint32_t* matCRowIndices, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream);
template void MoeBatchedGemm<__nv_bfloat16, 256>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, const uint32_t* matBIndices,
    __nv_bfloat16* C, uint32_t* matCRowIndices, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream);
template <>
void MoeBatchedGemmLauncher<float>(const float* A, const float* B,
                                   const uint32_t* matBIndices, float* C,
                                   uint32_t* matCRowIndices, void* deviceWs,
                                   size_t deviceWsSize, uint32_t matARows,
                                   uint32_t n, uint32_t k, uint32_t nMatB,
                                   uint32_t nMatBPerMatARow,
                                   cudaStream_t stream) {
  // TODO
  LOG(ERROR) << "DNN_MOE not support FP32 now";
}
#ifdef ENABLE_FP16
template <>
void MoeBatchedGemmLauncher<half>(const half* A, const half* B,
                                  const uint32_t* matBIndices, half* C,
                                  uint32_t* matCRowIndices, void* deviceWs,
                                  size_t deviceWsSize, uint32_t matARows,
                                  uint32_t n, uint32_t k, uint32_t nMatB,
                                  uint32_t nMatBPerMatARow,
                                  cudaStream_t stream) {
  if (nMatB <= 64) {
    MoeBatchedGemm<half, 64>(A, B, matBIndices, C, matCRowIndices, deviceWs,
                             deviceWsSize, matARows, n, k, nMatB,
                             nMatBPerMatARow, stream);
  } else if (nMatB <= 128) {
    MoeBatchedGemm<half, 128>(A, B, matBIndices, C, matCRowIndices, deviceWs,
                              deviceWsSize, matARows, n, k, nMatB,
                              nMatBPerMatARow, stream);
  } else if (nMatB <= 256) {
    MoeBatchedGemm<half, 256>(A, B, matBIndices, C, matCRowIndices, deviceWs,
                              deviceWsSize, matARows, n, k, nMatB,
                              nMatBPerMatARow, stream);
  }
}
#endif
#ifdef ENABLE_BF16
template <>
void MoeBatchedGemmLauncher<hie::bfloat16>(
    const hie::bfloat16* A, const hie::bfloat16* B, const uint32_t* matBIndices,
    hie::bfloat16* C, uint32_t* matCRowIndices, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  if (nMatB <= 64) {
    MoeBatchedGemm<__nv_bfloat16, 64>(
        (__nv_bfloat16*)A, (__nv_bfloat16*)B, matBIndices, (__nv_bfloat16*)C,
        matCRowIndices, deviceWs, deviceWsSize, matARows, n, k, nMatB,
        nMatBPerMatARow, stream);
  } else if (nMatB <= 128) {
    MoeBatchedGemm<__nv_bfloat16, 128>(
        (__nv_bfloat16*)A, (__nv_bfloat16*)B, matBIndices, (__nv_bfloat16*)C,
        matCRowIndices, deviceWs, deviceWsSize, matARows, n, k, nMatB,
        nMatBPerMatARow, stream);
  } else if (nMatB <= 256) {
    MoeBatchedGemm<__nv_bfloat16, 256>(
        (__nv_bfloat16*)A, (__nv_bfloat16*)B, matBIndices, (__nv_bfloat16*)C,
        matCRowIndices, deviceWs, deviceWsSize, matARows, n, k, nMatB,
        nMatBPerMatARow, stream);
  }
}
#endif
template size_t GetWorkspaceSize<64>(uint32_t, uint32_t);
template size_t GetWorkspaceSize<128>(uint32_t, uint32_t);
template size_t GetWorkspaceSize<256>(uint32_t, uint32_t);
size_t GetWorkspaceSizeLauncher(uint32_t matARows, uint32_t nMatB) {
  if (nMatB <= 64) {
    return GetWorkspaceSize<64>(matARows, nMatB);
  } else if (nMatB <= 128) {
    return GetWorkspaceSize<128>(matARows, nMatB);
  } else if (nMatB <= 256) {
    return GetWorkspaceSize<256>(matARows, nMatB);
  }
  return 0;
}
}  // namespace cuda
}  // namespace allspark
