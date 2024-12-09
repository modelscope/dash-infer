/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_ppu_kernel.cu
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>

#include "../cuda_kernel.h"
#include "allspark.pb.h"
#include "moe_ppu_kernel.h"
namespace allspark {
namespace cuda {
struct alignas(16) BatchInfo {
  uint32_t batchId;
  uint32_t m;
  uint32_t ctaYOffset;
  uint32_t COffset;
};

__device__ __forceinline__ uint32_t SmemU32Addr(const void* smemptr) {
  uint32_t u32addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr; }\n"
      : "=r"(u32addr)
      : "l"(smemptr));
  return u32addr;
}

__device__ __forceinline__ void LdgSts32(const uint32_t& smemAddr,
                                         const void* gmemPtr, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 4;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"((int)guard));
}

__device__ __forceinline__ void LdgSts32(const uint32_t& smemAddr,
                                         const void* gmemPtr,
                                         const uint32_t& srcSize, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 4, %2;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"(srcSize), "r"((int)guard));
}

__device__ __forceinline__ void LdgSts64(const uint32_t& smemAddr,
                                         const void* gmemPtr, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 8;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"((int)guard));
}

__device__ __forceinline__ void LdgSts64(const uint32_t& smemAddr,
                                         const void* gmemPtr,
                                         const uint32_t& srcSize, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 8, %2;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"(srcSize), "r"((int)guard));
}

__device__ __forceinline__ void LdgSts128(const uint32_t& smemAddr,
                                          const void* gmemPtr, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 16;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"((int)guard));
}

__device__ __forceinline__ void LdgSts128(const uint32_t& smemAddr,
                                          const void* gmemPtr,
                                          const uint32_t& srcSize, bool guard) {
  asm volatile(
      "{.reg.pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p cp.async.ca.shared.global [%0], [%1], 16, %2;}\n"
      :
      : "r"(smemAddr), "l"(gmemPtr), "r"(srcSize), "r"((int)guard));
}

__device__ __forceinline__ void LdgStsGroupCommit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void LdgStsGroupWait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

template <typename T>
__device__ __forceinline__ void Ldsm4(T& r0, T& r1, T& r2, T& r3,
                                      const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "Ldsm4: invalid T");
  asm volatile(
      "alippu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n"
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void Ldsm4Trans(T& r0, T& r1, T& r2, T& r3,
                                           const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "Ldsm4Trans: invalid T");
  asm volatile(
      "alippu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n"
      : "=r"(reinterpret_cast<uint32_t&>(r0)),
        "=r"(reinterpret_cast<uint32_t&>(r1)),
        "=r"(reinterpret_cast<uint32_t&>(r2)),
        "=r"(reinterpret_cast<uint32_t&>(r3))
      : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void Hmma161616F32(T (&d)[8], const T (&a)[4],
                                              const T (&b)[4]) {
  static_assert(sizeof(T) == 4, "Hmma161616F32: invalid T");
  asm volatile (
        "alippu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3, %4, %5, %6, %7},"
        " {%8, %9, %10, %11},"
        " {%12, %13, %14, %15},"
        " {%0, %1, %2, %3, %4, %5, %6, %7};\n"
        : "+r"(reinterpret_cast<uint32_t &>(d[0])),
          "+r"(reinterpret_cast<uint32_t &>(d[1]))
          "+r"(reinterpret_cast<uint32_t &>(d[2]))
          "+r"(reinterpret_cast<uint32_t &>(d[3]))
          "+r"(reinterpret_cast<uint32_t &>(d[4]))
          "+r"(reinterpret_cast<uint32_t &>(d[5]))
          "+r"(reinterpret_cast<uint32_t &>(d[6]))
          "+r"(reinterpret_cast<uint32_t &>(d[7]))
        : "r"(reinterpret_cast<const uint32_t &>(a[0])),
          "r"(reinterpret_cast<const uint32_t &>(a[1])),
          "r"(reinterpret_cast<const uint32_t &>(a[2])),
          "r"(reinterpret_cast<const uint32_t &>(a[3])),
          "r"(reinterpret_cast<const uint32_t &>(b[0])),
          "r"(reinterpret_cast<const uint32_t &>(b[1])),
          "r"(reinterpret_cast<const uint32_t &>(b[2])),
          "r"(reinterpret_cast<const uint32_t &>(b[3]))
    );
}

template <typename T>
__device__ __forceinline__ void Stg64(const T& r0, const T& r1, const void* ptr,
                                      bool guard) {
  static_assert(sizeof(T) == 4, "Stg64: invalid T");
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %1, 0;\n"
      " @p st.global.v2.b32 [%0], {%2, %3};}\n"
      :
      : "l"(ptr), "r"((int)guard), "r"(reinterpret_cast<const uint32_t&>(r0)),
        "r"(reinterpret_cast<const uint32_t&>(r1)));
}

/**
 * m_tile: 128
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 64x64
 * CTA: 2x4 warps
 * smem size: 120KB
 */
__device__ __forceinline__ void hgemm_f32_m128n256_k32x5_hmma161616_ldg8_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int mIdx = tileIdY * 128 + threadIdx.x / 4 + i * 64;
    if (mIdx < m) {
      asm("ld.global.ca.b32 %0, [%1];"
          : "=r"(matARowId[i])
          : "l"(matARowIdx + mIdx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matARowId[i] = 0;
    }
  }

  const char* ALdgPtr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    ALdgPtr[i] = reinterpret_cast<const char*>(A + matARowId[i] * k +
                                               threadIdx.x % 4 * 8);
  }
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 128 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 4) * (128 * 8) +
                      ((threadIdx.x / 4) ^ (threadIdx.x % 4 * 2)) * 8);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 128 * 8 +
                                              (warpId / 4) * 64 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int col = (laneId / 8 % 2 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId % 4) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 4 * 8 < firstKTile ? 16 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      LdgSts128(AStsAddr + i * 64 * 8 * sizeof(half), ALdgPtr[i], ASrcSize,
                true);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

// ldg pointer for the next tile
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ALdgPtr[i] += firstKTile * sizeof(half);
    }
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        LdgSts128(
            AStsAddr + prefetchIter * 1024 * 24 + i * 64 * 8 * sizeof(half),
            ALdgPtr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 24 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

// ldg pointer for the next tile
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        ALdgPtr[i] += 32 * sizeof(half);
      }
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 96 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][4][4];
  uint32_t BFrag[2][4][4];
  uint32_t CFrag[4][4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        CFrag[i][j][p] = 0;
      }
    }
  }

// load 1'st fragment
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Ldsm4(AFrag[0][i][0], AFrag[0][i][1], AFrag[0][i][2], AFrag[0][i][3],
          ALdsAddr[0] + ldsOffset + i * 16 * 8 * sizeof(half));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 96 * 1024 ? ldsOffset + 24 * 1024 : 0;
          stsOffset = stsOffset < 96 * 1024 ? stsOffset + 24 * 1024 : 0;

// ldg pointer for the next tile
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            LdgSts128(AStsAddr + stsOffset + i * 64 * 8 * sizeof(half),
                      ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 96 * 1024 ? ldsOffset + 24 * 1024 : 0;
          stsOffset = stsOffset < 96 * 1024 ? stsOffset + 24 * 1024 : 0;

// ldg pointer for the next tile
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            LdgSts128(AStsAddr + stsOffset + i * 64 * 8 * sizeof(half),
                      ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 96 * 1024 ? ldsOffset + 24 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

// load next A&B fragment from shared memory to register
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        Ldsm4(
            AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
            AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset + i * 16 * 8 * sizeof(half));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i], BFrag[kFrag % 2][j]);
        }
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId % 4 * 64 + laneId % 4;
  uint32_t CStsIdxY = warpId / 4 * 32 + laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 128 * 32 * 65 +
                          threadIdx.x % 128 / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx =
      tileIdY * 128 + threadIdx.x / 128 * 64 + threadIdx.x % 128 / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

#pragma unroll
  for (int stgIter = 0; stgIter < 2; ++stgIter) {
    // C_tile sts
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        CStsPtr[i * 16 * 260 + j * 16] = CFrag[stgIter * 2 + i][j][0];
        CStsPtr[i * 16 * 260 + j * 16 + 4] = CFrag[stgIter * 2 + i][j][1];
        CStsPtr[i * 16 * 260 + j * 16 + 8] = CFrag[stgIter * 2 + i][j][2];
        CStsPtr[i * 16 * 260 + j * 16 + 12] = CFrag[stgIter * 2 + i][j][3];

        CStsPtr[i * 16 * 260 + 8 * 260 + j * 16] = CFrag[stgIter * 2 + i][j][4];
        CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 4] =
            CFrag[stgIter * 2 + i][j][5];
        CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 8] =
            CFrag[stgIter * 2 + i][j][6];
        CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 12] =
            CFrag[stgIter * 2 + i][j][7];
      }
    }
    __syncthreads();

    // lds
    float4 CLdsReg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      CLdsReg[i] = CLdsPtr[i * 2 * 65];
    }

    half2 CStgReg[16][2];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      asm("{.reg .b16 h0, h1, h2, h3;\n"
          " cvt.rn.f16.f32 h0, %2;\n"
          " cvt.rn.f16.f32 h1, %3;\n"
          " cvt.rn.f16.f32 h2, %4;\n"
          " cvt.rn.f16.f32 h3, %5;\n"
          " mov.b32 %0, {h0, h1};\n"
          " mov.b32 %1, {h2, h3};}"
          : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
            "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
          : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
            "f"(CLdsReg[i].w));
    }

// C_tile stg
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + (stgIter * 32 + i * 2) * n,
            mIdx + stgIter * 32 + i * 2 < m && nGuard);
    }
  }
}

/**
 * m_tile: 96
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 48x64
 * CTA: 2x4 warps
 * smem size: 110KB
 */
__device__ __forceinline__ void hgemm_f32_m96n256_k32x5_hmma161616_ldg4_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    int mIdx = tileIdY * 96 + threadIdx.x / 8 + i * 32;
    if (mIdx < m) {
      asm("ld.global.ca.b32 %0, [%1];"
          : "=r"(matARowId[i])
          : "l"(matARowIdx + mIdx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matARowId[i] = 0;
    }
  }

  const char* ALdgPtr[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    ALdgPtr[i] = reinterpret_cast<const char*>(A + matARowId[i] * k +
                                               threadIdx.x % 8 * 4);
  }
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 96 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 8 / 2) * (96 * 8) +
                      ((threadIdx.x / 8) ^ (threadIdx.x % 8 / 2 * 2)) * 8 +
                      threadIdx.x % 2 * 4);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 96 * 8 +
                                              (warpId / 4) * 48 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int col = (laneId / 8 % 2 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId % 4) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 8 * 4 < firstKTile ? 8 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
      LdgSts64(AStsAddr + i * 32 * 8 * sizeof(half), ALdgPtr[i], ASrcSize,
               true);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

// ldg pointer for the the next tile
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      ALdgPtr[i] += firstKTile * sizeof(half);
    }
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        LdgSts64(
            AStsAddr + prefetchIter * 1024 * 22 + i * 32 * 8 * sizeof(half),
            ALdgPtr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 22 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

// ldg pointer for the the next tile
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        ALdgPtr[i] += 32 * sizeof(half);
      }
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 88 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][3][4];
  uint32_t BFrag[2][4][4];
  uint32_t CFrag[3][4][8];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        CFrag[i][j][p] = 0;
      }
    }
  }

// load 1'st fragment
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    Ldsm4(AFrag[0][i][0], AFrag[0][i][1], AFrag[0][i][2], AFrag[0][i][3],
          ALdsAddr[0] + ldsOffset + i * 16 * 8 * sizeof(half));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 88 * 1024 ? ldsOffset + 22 * 1024 : 0;
          stsOffset = stsOffset < 88 * 1024 ? stsOffset + 22 * 1024 : 0;

// ldg pointer for the next tile
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            LdgSts64(AStsAddr + stsOffset + i * 32 * 8 * sizeof(half),
                     ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 88 * 1024 ? ldsOffset + 22 * 1024 : 0;
          stsOffset = stsOffset < 88 * 1024 ? stsOffset + 22 * 1024 : 0;

// ldg pointer for next tile
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            LdgSts64(AStsAddr + stsOffset + i * 32 * 8 * sizeof(half),
                     ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 88 * 1024 ? ldsOffset + 22 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

// load next A&B fragment from shared memory to register
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        Ldsm4(
            AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
            AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset + i * 16 * 8 * sizeof(half));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i], BFrag[kFrag % 2][j]);
        }
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId % 4 * 64 + laneId % 4;
  uint32_t CStsIdxY = warpId / 4 * 48 + laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx = tileIdY * 96 + threadIdx.x / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      CStsPtr[i * 16 * 260 + j * 16] = CFrag[i][j][0];
      CStsPtr[i * 16 * 260 + j * 16 + 4] = CFrag[i][j][1];
      CStsPtr[i * 16 * 260 + j * 16 + 8] = CFrag[i][j][2];
      CStsPtr[i * 16 * 260 + j * 16 + 12] = CFrag[i][j][3];

      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16] = CFrag[i][j][4];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 4] = CFrag[i][j][5];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 8] = CFrag[i][j][6];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 12] = CFrag[i][j][7];
    }
  }
  __syncthreads();

  float4 CLdsReg[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    CLdsReg[i] = CLdsPtr[i * 4 * 65];
  }

  half2 CStgReg[24][2];
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    asm("{.reg .b16 h0, h1, h2, h3;\n"
        " cvt.rn.f16.f32 h0, %2;\n"
        " cvt.rn.f16.f32 h1, %3;\n"
        " cvt.rn.f16.f32 h2, %4;\n"
        " cvt.rn.f16.f32 h3, %5;\n"
        " mov.b32 %0, {h0, h1};\n"
        " mov.b32 %1, {h2, h3};}"
        : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
          "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
        : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
          "f"(CLdsReg[i].w));
  }

// C_tile stg
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + i * 4 * n,
          mIdx + i * 4 < m && nGuard);
  }
}

/**
 * m_tile: 64
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 32x64
 * CTA: 2x4 warps
 * smem size: 100KB
 */
__device__ __forceinline__ void hgemm_f32_m64n256_k32x5_hmma161616_ldg8_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId;
  if (tileIdY * 64 + threadIdx.x / 4 < m) {
    asm("ld.global.ca.b32 %0, [%1];"
        : "=r"(matARowId)
        : "l"(matARowIdx + tileIdY * 64 + threadIdx.x / 4));
  } else {
    // map the out-of-bound threads to row0 of matrixA,
    // to avoid predicated ld instructions
    matARowId = 0;
  }

  const char* ALdgPtr =
      reinterpret_cast<const char*>(A + matARowId * k + threadIdx.x % 4 * 8);
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 64 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 4) * (64 * 8) +
                      ((threadIdx.x / 4) ^ (threadIdx.x % 4 * 2)) * 8);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 64 * 8 +
                                              (warpId / 4) * 32 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int col = (laneId / 8 % 2 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId % 4) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 4 * 8 < firstKTile ? 16 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

    LdgSts128(AStsAddr, ALdgPtr, ASrcSize, true);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

    // ldg pointer for the next tile
    ALdgPtr += firstKTile * sizeof(half);
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
      LdgSts128(AStsAddr + prefetchIter * 1024 * 20, ALdgPtr, true);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 20 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

      // ldg pointer for the next tile
      ALdgPtr += 32 * sizeof(half);
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 80 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][2][4];
  uint32_t BFrag[2][4][4];
  uint32_t CFrag[2][4][8];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        CFrag[i][j][p] = 0;
      }
    }
  }

// load 1'st fragment
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    Ldsm4(AFrag[0][i][0], AFrag[0][i][1], AFrag[0][i][2], AFrag[0][i][3],
          ALdsAddr[0] + ldsOffset + i * 16 * 8 * sizeof(half));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 80 * 1024 ? ldsOffset + 20 * 1024 : 0;
          stsOffset = stsOffset < 80 * 1024 ? stsOffset + 20 * 1024 : 0;

          // ldg pointer for the next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts128(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 80 * 1024 ? ldsOffset + 20 * 1024 : 0;
          stsOffset = stsOffset < 80 * 1024 ? stsOffset + 20 * 1024 : 0;

          // ldg pointer for the next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts128(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 80 * 1024 ? ldsOffset + 20 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

// load next A&B fragment from shared memory to register
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Ldsm4(
            AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
            AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset + i * 16 * 8 * sizeof(half));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i], BFrag[kFrag % 2][j]);
        }
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId % 4 * 64 + laneId % 4;
  uint32_t CStsIdxY = warpId / 4 * 32 + laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx = tileIdY * 64 + threadIdx.x / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      CStsPtr[i * 16 * 260 + j * 16] = CFrag[i][j][0];
      CStsPtr[i * 16 * 260 + j * 16 + 4] = CFrag[i][j][1];
      CStsPtr[i * 16 * 260 + j * 16 + 8] = CFrag[i][j][2];
      CStsPtr[i * 16 * 260 + j * 16 + 12] = CFrag[i][j][3];

      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16] = CFrag[i][j][4];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 4] = CFrag[i][j][5];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 8] = CFrag[i][j][6];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 12] = CFrag[i][j][7];
    }
  }
  __syncthreads();

  float4 CLdsReg[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    CLdsReg[i] = CLdsPtr[i * 4 * 65];
  }

  half2 CStgReg[16][2];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    asm("{.reg .b16 h0, h1, h2, h3;\n"
        " cvt.rn.f16.f32 h0, %2;\n"
        " cvt.rn.f16.f32 h1, %3;\n"
        " cvt.rn.f16.f32 h2, %4;\n"
        " cvt.rn.f16.f32 h3, %5;\n"
        " mov.b32 %0, {h0, h1};\n"
        " mov.b32 %1, {h2, h3};}"
        : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
          "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
        : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
          "f"(CLdsReg[i].w));
  }

// C_tile stg
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + i * 4 * n,
          mIdx + i * 4 < m && nGuard);
  }
}

/**
 * m_tile: 48
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 48x32
 * CTA: 1x8 warps
 * smem size: 95KB
 */
__device__ __forceinline__ void hgemm_f32_m48n256_k32x5_hmma161616_ldg2_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    int mIdx = tileIdY * 48 + threadIdx.x / 16 + i * 16;
    if (mIdx < m) {
      asm("ld.global.ca.b32 %0, [%1];"
          : "=r"(matARowId[i])
          : "l"(matARowIdx + mIdx));
    } else {
      // map the out-of-bound threads to row0 of matrixA,
      // to avoid predicated ld instructions
      matARowId[i] = 0;
    }
  }

  const char* ALdgPtr[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    ALdgPtr[i] = reinterpret_cast<const char*>(A + matARowId[i] * k +
                                               threadIdx.x % 16 * 2);
  }
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 48 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 16 / 4) * (48 * 8) +
                      ((threadIdx.x / 16) ^ (threadIdx.x % 16 / 4 * 2)) * 8 +
                      threadIdx.x % 4 * 2);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 48 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = (laneId / 8 % 2 + warpId % 2 * 4 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId / 2) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 16 * 2 < firstKTile ? 4 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
      LdgSts32(AStsAddr + i * 16 * 8 * sizeof(half), ALdgPtr[i], ASrcSize,
               true);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

// ldg pointer for the the next tile
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      ALdgPtr[i] += firstKTile * sizeof(half);
    }
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        LdgSts32(
            AStsAddr + prefetchIter * 1024 * 19 + i * 16 * 8 * sizeof(half),
            ALdgPtr[i], true);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 19 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

// ldg pointer for the the next tile
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        ALdgPtr[i] += 32 * sizeof(half);
      }
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 76 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][3][4];
  uint32_t BFrag[2][2][4];
  uint32_t CFrag[3][2][8];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        CFrag[i][j][p] = 0;
      }
    }
  }

// load 1'st fragment
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    Ldsm4(AFrag[0][i][0], AFrag[0][i][1], AFrag[0][i][2], AFrag[0][i][3],
          ALdsAddr[0] + ldsOffset + i * 16 * 8 * sizeof(half));
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 76 * 1024 ? ldsOffset + 19 * 1024 : 0;
          stsOffset = stsOffset < 76 * 1024 ? stsOffset + 19 * 1024 : 0;

// ldg pointer for the next tile
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            LdgSts32(AStsAddr + stsOffset + i * 16 * 8 * sizeof(half),
                     ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 76 * 1024 ? ldsOffset + 19 * 1024 : 0;
          stsOffset = stsOffset < 76 * 1024 ? stsOffset + 19 * 1024 : 0;

// ldg pointer for next tile
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            ALdgPtr[i] += 32 * sizeof(half);
          }
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            LdgSts32(AStsAddr + stsOffset + i * 16 * 8 * sizeof(half),
                     ALdgPtr[i], true);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 76 * 1024 ? ldsOffset + 19 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

// load next A&B fragment from shared memory to register
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        Ldsm4(
            AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
            AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset + i * 16 * 8 * sizeof(half));
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i], BFrag[kFrag % 2][j]);
        }
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId * 32 + laneId % 4;
  uint32_t CStsIdxY = laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx = tileIdY * 48 + threadIdx.x / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      CStsPtr[i * 16 * 260 + j * 16] = CFrag[i][j][0];
      CStsPtr[i * 16 * 260 + j * 16 + 4] = CFrag[i][j][1];
      CStsPtr[i * 16 * 260 + j * 16 + 8] = CFrag[i][j][2];
      CStsPtr[i * 16 * 260 + j * 16 + 12] = CFrag[i][j][3];

      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16] = CFrag[i][j][4];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 4] = CFrag[i][j][5];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 8] = CFrag[i][j][6];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 12] = CFrag[i][j][7];
    }
  }
  __syncthreads();

  float4 CLdsReg[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    CLdsReg[i] = CLdsPtr[i * 4 * 65];
  }

  half2 CStgReg[12][2];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    asm("{.reg .b16 h0, h1, h2, h3;\n"
        " cvt.rn.f16.f32 h0, %2;\n"
        " cvt.rn.f16.f32 h1, %3;\n"
        " cvt.rn.f16.f32 h2, %4;\n"
        " cvt.rn.f16.f32 h3, %5;\n"
        " mov.b32 %0, {h0, h1};\n"
        " mov.b32 %1, {h2, h3};}"
        : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
          "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
        : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
          "f"(CLdsReg[i].w));
  }

// C_tile stg
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + i * 4 * n,
          mIdx + i * 4 < m && nGuard);
  }
}

/**
 * m_tile: 32
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 32x32
 * CTA: 1x8 warps
 * smem size: 90KB
 */
__device__ __forceinline__ void hgemm_f32_m32n256_k32x5_hmma161616_ldg4_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId;
  if (tileIdY * 32 + threadIdx.x / 8 < m) {
    asm("ld.global.ca.b32 %0, [%1];"
        : "=r"(matARowId)
        : "l"(matARowIdx + tileIdY * 32 + threadIdx.x / 8));
  } else {
    // map the out-of-bound threads to row0 of matrixA,
    // to avoid predicated ld instructions
    matARowId = 0;
  }

  const char* ALdgPtr =
      reinterpret_cast<const char*>(A + matARowId * k + threadIdx.x % 8 * 4);
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 32 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 8 / 2) * (32 * 8) +
                      ((threadIdx.x / 8) ^ (threadIdx.x % 8 / 2 * 2)) * 8 +
                      threadIdx.x % 2 * 4);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 32 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int col = (laneId / 8 % 2 + warpId % 2 * 4 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId / 2) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 8 * 4 < firstKTile ? 8 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

    LdgSts64(AStsAddr, ALdgPtr, ASrcSize, true);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

    // ldg pointer for the the next tile
    ALdgPtr += firstKTile * sizeof(half);
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
      LdgSts64(AStsAddr + prefetchIter * 1024 * 18, ALdgPtr, true);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 18 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

      // ldg pointer for the the next tile
      ALdgPtr += 32 * sizeof(half);
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 72 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][2][4];
  uint32_t BFrag[2][2][4];
  uint32_t CFrag[2][2][8];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        CFrag[i][j][p] = 0;
      }
    }
  }

// load 1'st fragment
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    Ldsm4(AFrag[0][i][0], AFrag[0][i][1], AFrag[0][i][2], AFrag[0][i][3],
          ALdsAddr[0] + ldsOffset + i * 16 * 8 * sizeof(half));
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 72 * 1024 ? ldsOffset + 18 * 1024 : 0;
          stsOffset = stsOffset < 72 * 1024 ? stsOffset + 18 * 1024 : 0;

          // ldg pointer for the next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts64(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 72 * 1024 ? ldsOffset + 18 * 1024 : 0;
          stsOffset = stsOffset < 72 * 1024 ? stsOffset + 18 * 1024 : 0;

          // ldg pointer for next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

// load next A&B fragment from shared memory to register
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4(AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
                AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
                ALdsAddr[(kFrag + 1) % 2] + ldsOffset +
                    i * 16 * 8 * sizeof(half));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i],
                          BFrag[kFrag % 2][j]);
          }
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts64(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 72 * 1024 ? ldsOffset + 18 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

// load next A&B fragment from shared memory to register
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Ldsm4(
            AFrag[(kFrag + 1) % 2][i][0], AFrag[(kFrag + 1) % 2][i][1],
            AFrag[(kFrag + 1) % 2][i][2], AFrag[(kFrag + 1) % 2][i][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset + i * 16 * 8 * sizeof(half));
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          Hmma161616F32(CFrag[i][j], AFrag[kFrag % 2][i], BFrag[kFrag % 2][j]);
        }
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId * 32 + laneId % 4;
  uint32_t CStsIdxY = laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx = tileIdY * 32 + threadIdx.x / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      CStsPtr[i * 16 * 260 + j * 16] = CFrag[i][j][0];
      CStsPtr[i * 16 * 260 + j * 16 + 4] = CFrag[i][j][1];
      CStsPtr[i * 16 * 260 + j * 16 + 8] = CFrag[i][j][2];
      CStsPtr[i * 16 * 260 + j * 16 + 12] = CFrag[i][j][3];

      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16] = CFrag[i][j][4];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 4] = CFrag[i][j][5];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 8] = CFrag[i][j][6];
      CStsPtr[i * 16 * 260 + 8 * 260 + j * 16 + 12] = CFrag[i][j][7];
    }
  }
  __syncthreads();

  float4 CLdsReg[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    CLdsReg[i] = CLdsPtr[i * 4 * 65];
  }

  half2 CStgReg[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    asm("{.reg .b16 h0, h1, h2, h3;\n"
        " cvt.rn.f16.f32 h0, %2;\n"
        " cvt.rn.f16.f32 h1, %3;\n"
        " cvt.rn.f16.f32 h2, %4;\n"
        " cvt.rn.f16.f32 h3, %5;\n"
        " mov.b32 %0, {h0, h1};\n"
        " mov.b32 %1, {h2, h3};}"
        : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
          "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
        : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
          "f"(CLdsReg[i].w));
  }

// C_tile stg
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + i * 4 * n,
          mIdx + i * 4 < m && nGuard);
  }
}

/**
 * m_tile: 16
 * n_tile: 256
 * k_tile: 32x5
 * warp_tile: 16x32
 * CTA: 1x8 warps
 * smem size: 85KB
 */
__device__ __forceinline__ void hgemm_f32_m16n256_k32x5_hmma161616_ldg2_loop(
    const half* A, const half* B, const uint32_t* matARowIdx, half* C,
    char* smem, const uint32_t& m, const uint32_t& n, const uint32_t& k,
    const uint32_t& tileIdX, const uint32_t& tileIdY,
    const uint32_t& BLdgStep) {
  uint32_t warpId = threadIdx.x / 32;
  uint32_t laneId = threadIdx.x % 32;

  uint32_t matARowId;
  if (tileIdY * 16 + threadIdx.x / 16 < m) {
    asm("ld.global.ca.b32 %0, [%1];"
        : "=r"(matARowId)
        : "l"(matARowIdx + tileIdY * 16 + threadIdx.x / 16));
  } else {
    // map the out-of-bound threads to row0 of matrixA,
    // to avoid predicated ld instructions
    matARowId = 0;
  }

  const char* ALdgPtr =
      reinterpret_cast<const char*>(A + matARowId * k + threadIdx.x % 16 * 2);
  const char* BLdgPtr = reinterpret_cast<const char*>(
      B + (threadIdx.x / 8) * n + tileIdX * 256 + (threadIdx.x % 8) * 8);

  // LdgGuard to avoid LDG out of bound
  uint32_t BLdgGuard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int nIdx = tileIdX * 256 + (threadIdx.x % 8) * 8 + i * 64;
    if (nIdx < n) {
      BLdgGuard |= (1U << i);
    }
  }

  uint32_t ASmemAddr = SmemU32Addr(smem);
  uint32_t BSmemAddr = SmemU32Addr(smem + 16 * 32 * sizeof(half));

  uint32_t AStsAddr =
      ASmemAddr +
      sizeof(half) * ((threadIdx.x % 16 / 4) * (16 * 8) +
                      ((threadIdx.x / 16) ^ (threadIdx.x % 16 / 4 * 2)) * 8 +
                      threadIdx.x % 4 * 2);
  uint32_t BStsAddr =
      BSmemAddr +
      sizeof(half) * ((threadIdx.x / 8) * 256 +
                      ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // ATile lds addr
  uint32_t ALdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = laneId / 8 % 2 + i * 2;
    int row = (laneId / 16 * 8 + laneId % 8) ^ (col * 2);
    ALdsAddr[i] = ASmemAddr + sizeof(half) * (col * 16 * 8 + row * 8);
  }

  // BTile lds addr
  uint32_t BLdsAddr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int col = (laneId / 8 % 2 + warpId % 2 * 4 + i * 2) ^ (laneId % 8);
    int row = laneId / 16 * 8 + laneId % 8;
    BLdsAddr[i] =
        BSmemAddr + sizeof(half) * (row * 256 + (warpId / 2) * 64 + col * 8);
  }

  uint32_t kTiles = (k + 31) / 32;

  // load 1'st tile to shared memory
  {
    uint32_t firstKTile = k - (kTiles * 32 - 32);
    uint32_t ASrcSize = threadIdx.x % 16 * 2 < firstKTile ? 4 : 0;
    uint32_t BSrcSize = threadIdx.x / 8 < firstKTile ? 16 : 0;

    LdgSts32(AStsAddr, ALdgPtr, ASrcSize, true);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      LdgSts128(BStsAddr + i * 64 * sizeof(half),
                BLdgPtr + i * 64 * sizeof(half), BSrcSize,
                (BLdgGuard & (1u << i)) != 0);
    }
    LdgStsGroupCommit();

    // ldg pointer for the the next tile
    ALdgPtr += firstKTile * sizeof(half);
    BLdgPtr += firstKTile * n * sizeof(half);
  }

// load 2'st to (N-stages - 1) tiles to shared memory
#pragma unroll
  for (int prefetchIter = 1; prefetchIter < 4; ++prefetchIter) {
    if (prefetchIter < kTiles) {
      LdgSts32(AStsAddr + prefetchIter * 1024 * 17, ALdgPtr, true);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        LdgSts128(BStsAddr + prefetchIter * 1024 * 17 + i * 64 * sizeof(half),
                  BLdgPtr + i * 64 * sizeof(half),
                  (BLdgGuard & (1u << i)) != 0);
      }

      // ldg pointer for the the next tile
      ALdgPtr += 32 * sizeof(half);
      BLdgPtr += BLdgStep;
    }
    LdgStsGroupCommit();
  }

  // wait for the 1'st tile
  LdgStsGroupWait<3>();
  __syncthreads();

  // smem double buffer offset
  uint32_t ldsOffset = 0;
  uint32_t stsOffset = 68 * 1024;

  // A, B and C register fragment
  uint32_t AFrag[2][4];
  uint32_t BFrag[2][2][4];
  uint32_t CFrag[2][8];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      CFrag[i][p] = 0;
    }
  }

  // load 1'st fragment
  Ldsm4(AFrag[0][0], AFrag[0][1], AFrag[0][2], AFrag[0][3],
        ALdsAddr[0] + ldsOffset);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    Ldsm4Trans(BFrag[0][i][0], BFrag[0][i][1], BFrag[0][i][2], BFrag[0][i][3],
               BLdsAddr[i] + ldsOffset);
  }

  if (tileIdX * 256 + 256 <= n) {
    // matrixB CTA tile is full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 68 * 1024 ? ldsOffset + 17 * 1024 : 0;
          stsOffset = stsOffset < 68 * 1024 ? stsOffset + 17 * 1024 : 0;

          // ldg pointer for the next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

        // load next A&B fragment from shared memory to register
        Ldsm4(AFrag[(kFrag + 1) % 2][0], AFrag[(kFrag + 1) % 2][1],
              AFrag[(kFrag + 1) % 2][2], AFrag[(kFrag + 1) % 2][3],
              ALdsAddr[(kFrag + 1) % 2] + ldsOffset);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Hmma161616F32(CFrag[i], AFrag[kFrag % 2], BFrag[kFrag % 2][i]);
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts32(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half), true);
          }
          LdgStsGroupCommit();
        }
      }
    }
  } else {
    // matrixB CTA tile is not full
    for (; kTiles > 4; --kTiles) {
#pragma unroll
      for (int kFrag = 0; kFrag < 2; ++kFrag) {
        // store next A&B tile to shared memory
        if (kFrag == 1) {
          // switch double buffer
          ldsOffset = ldsOffset < 68 * 1024 ? ldsOffset + 17 * 1024 : 0;
          stsOffset = stsOffset < 68 * 1024 ? stsOffset + 17 * 1024 : 0;

          // ldg pointer for next tile
          ALdgPtr += 32 * sizeof(half);
          BLdgPtr += BLdgStep;

          LdgStsGroupWait<3>();
          __syncthreads();
        }

        // load next A&B fragment from shared memory to register
        Ldsm4(AFrag[(kFrag + 1) % 2][0], AFrag[(kFrag + 1) % 2][1],
              AFrag[(kFrag + 1) % 2][2], AFrag[(kFrag + 1) % 2][3],
              ALdsAddr[(kFrag + 1) % 2] + ldsOffset);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                     BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                     BLdsAddr[i] + ldsOffset +
                         ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
        }

// HMMA loop
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          Hmma161616F32(CFrag[i], AFrag[kFrag % 2], BFrag[kFrag % 2][i]);
        }

        // tile prefetch
        if (kFrag == 0) {
          LdgSts32(AStsAddr + stsOffset, ALdgPtr, true);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            LdgSts128(BStsAddr + stsOffset + i * 64 * sizeof(half),
                      BLdgPtr + i * 64 * sizeof(half),
                      (BLdgGuard & (1U << i)) != 0);
          }
          LdgStsGroupCommit();
        }
      }
    }
  }

  // k-tiles loop without prefetch
  for (; kTiles > 0; --kTiles) {
#pragma unroll
    for (int kFrag = 0; kFrag < 2; ++kFrag) {
      // store next A&B tile to shared memory
      if (kFrag == 1) {
        // switch double buffer
        ldsOffset = ldsOffset < 68 * 1024 ? ldsOffset + 17 * 1024 : 0;

        LdgStsGroupWait<3>();
        __syncthreads();
      }

      // load next A&B fragment from shared memory to register
      Ldsm4(AFrag[(kFrag + 1) % 2][0], AFrag[(kFrag + 1) % 2][1],
            AFrag[(kFrag + 1) % 2][2], AFrag[(kFrag + 1) % 2][3],
            ALdsAddr[(kFrag + 1) % 2] + ldsOffset);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Ldsm4Trans(BFrag[(kFrag + 1) % 2][i][0], BFrag[(kFrag + 1) % 2][i][1],
                   BFrag[(kFrag + 1) % 2][i][2], BFrag[(kFrag + 1) % 2][i][3],
                   BLdsAddr[i] + ldsOffset +
                       ((kFrag + 1) % 2) * (16 * 256) * sizeof(half));
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Hmma161616F32(CFrag[i], AFrag[kFrag % 2], BFrag[kFrag % 2][i]);
      }

      // dummy LdgStsGroupCommit to make LdgStsGroupWait work
      if (kFrag == 0) {
        LdgStsGroupCommit();
      }
    }
  }

  uint32_t CStsIdxX = warpId * 32 + laneId % 4;
  uint32_t CStsIdxY = laneId / 4;
  uint32_t* CStsPtr =
      reinterpret_cast<uint32_t*>(smem) + CStsIdxY * 260 + CStsIdxX;
  const float4* CLdsPtr = reinterpret_cast<const float4*>(smem) +
                          threadIdx.x / 64 * 65 + threadIdx.x % 64;

  uint32_t mIdx = tileIdY * 16 + threadIdx.x / 64;
  uint32_t nIdx = tileIdX * 256 + threadIdx.x % 64 * 4;

  half* CStgPtr = C + mIdx * n + nIdx;
  bool nGuard = nIdx < n;

  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    CStsPtr[i * 16] = CFrag[i][0];
    CStsPtr[i * 16 + 4] = CFrag[i][1];
    CStsPtr[i * 16 + 8] = CFrag[i][2];
    CStsPtr[i * 16 + 12] = CFrag[i][3];

    CStsPtr[8 * 260 + i * 16] = CFrag[i][4];
    CStsPtr[8 * 260 + i * 16 + 4] = CFrag[i][5];
    CStsPtr[8 * 260 + i * 16 + 8] = CFrag[i][6];
    CStsPtr[8 * 260 + i * 16 + 12] = CFrag[i][7];
  }
  __syncthreads();

  float4 CLdsReg[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    CLdsReg[i] = CLdsPtr[i * 4 * 65];
  }

  half2 CStgReg[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm("{.reg .b16 h0, h1, h2, h3;\n"
        " cvt.rn.f16.f32 h0, %2;\n"
        " cvt.rn.f16.f32 h1, %3;\n"
        " cvt.rn.f16.f32 h2, %4;\n"
        " cvt.rn.f16.f32 h3, %5;\n"
        " mov.b32 %0, {h0, h1};\n"
        " mov.b32 %1, {h2, h3};}"
        : "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][0])),
          "=r"(reinterpret_cast<uint32_t&>(CStgReg[i][1]))
        : "f"(CLdsReg[i].x), "f"(CLdsReg[i].y), "f"(CLdsReg[i].z),
          "f"(CLdsReg[i].w));
  }

// C_tile stg
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Stg64(CStgReg[i][0], CStgReg[i][1], CStgPtr + i * 4 * n,
          mIdx + i * 4 < m && nGuard);
  }
}

__global__
    __launch_bounds__(256) void hgemm_f32_n256_k32x5_hmma161616_ldg8_kernel(
        const half* A, const half* B, half* C, const uint32_t* ctaIdYBarrier,
        const BatchInfo* batchInfos, const uint32_t* matARowIdx,
        uint32_t matARows, uint32_t n, uint32_t k,
        uint32_t BLdgStep) {  // 32 * n * sizeof(half)
  /**
   * CTA Tile Configuration:
   *
   * n_tile: 256, k_tile: 32x5
   * ---------------------------
   * m_tile 128: 64x64 warp tile, 2x4 warps
   * m_tile 96: 48x64 warp tile, 2x4 warps
   * m_tile 64: 32x64 warp tile, 2x4 warps
   * m_tile 48: 48x32 warp tile, 1x8 warps
   * m_tile 32: 32x32 warp tile, 1x8 warps
   * m_tile 16: 16x32 warp tile, 1x8 warps
   */

  // 24KB*5=120KB smem
  extern __shared__ char smem[];

  uint32_t ctaIdZ;
  uint32_t laneId = threadIdx.x % 32;
  asm(
      // for nMatB <= 64
      "{.reg .b32 r0, r1;\n"
      " .reg .pred p0, p1;\n"
      " ld.global.ca.b32 r0, [%1];\n"
      " ld.global.ca.b32 r1, [%1 + 128];\n"
      " setp.ge.u32 p0, %%ctaid.y, r0;\n"
      " setp.ge.u32 p1, %%ctaid.y, r1;\n"
      " vote.sync.ballot.b32 r0, p0, 0xffffffff;\n"
      " vote.sync.ballot.b32 r1, p1, 0xffffffff;\n"
      " popc.b32 r0, r0;\n"
      " popc.b32 r1, r1;\n"
      " add.u32 %0, r0, r1;}\n"
      : "=r"(ctaIdZ)
      : "l"(ctaIdYBarrier + laneId));

  // GEMM tile info
  BatchInfo batchInfo;
  asm("ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(reinterpret_cast<int4&>(batchInfo).x),
        "=r"(reinterpret_cast<int4&>(batchInfo).y),
        "=r"(reinterpret_cast<int4&>(batchInfo).z),
        "=r"(reinterpret_cast<int4&>(batchInfo).w),
      : "l"(batchInfos + ctaIdZ));
  uint32_t batchId = batchInfo.batchId;
  uint32_t m = batchInfo.m;
  uint32_t COffset = batchInfo.COffset;
  uint32_t ctaIdX = blockIdx.x;
  uint32_t ctaIdY = blockIdx.y - batchInfo.ctaYOffset;

  if (m > 96) {
    // m_tile 128, n_tile 256, k_tile 32x5, warp_tile 64x64, 2x4 warps
    hgemm_f32_m128n256_k32x5_hmma161616_ldg8_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
  } else if (m > 64) {
    // m_tile 96, n_tile 256, k_tile 32x5, warp_tile 48x64, 2x4 warps
    hgemm_f32_m96n256_k32x5_hmma161616_ldg4_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
  } else if (m > 48) {
    // m_tile 64, n_tile 256, k_tile 32x5, warp_tile 32x64, 2x4 warps
    hgemm_f32_m64n256_k32x5_hmma161616_ldg8_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
  } else if (m > 32) {
    // m_tile 48, n_tile 256, k_tile 32x5, warp_tile 48x32, 1x8 warps
    hgemm_f32_m48n256_k32x5_hmma161616_ldg2_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
  } else if (m > 16) {
    // m_tile 32, n_tile 256, k_tile 32x5, warp_tile 32x32, 1x8 warps
    hgemm_f32_m32n256_k32x5_hmma161616_ldg4_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
  } else {
    // m_tile 16, n_tile 256, k_tile 32x5, warp_tile 16x32, 1x8 warps
    hgemm_f32_m16n256_k32x5_hmma161616_ldg2_loop(
        A, B + batchId * k * n, matARowIdx + batchId * matARows, C + COffset,
        smem, m, n, k, ctaIdX, ctaIdY, BLdgStep);
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
    stgOffset = atomicAdd(smem + matBIdx, 1);
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
    stgOffset += smem[matBIdx];
    matCRowBatchOffset[idx] = stgOffset;
    matARowIndices[matBIdx * m + stgOffset] = matARowIdx;
  }
}

void MatARowIndex(const uint32_t* matBIndices,   // {m, nMatBPerMatARow}
                  uint32_t* matARowIndices,      // {nMatB, m}
                  uint32_t* batchedGemmM,        // {nMatB}
                  uint32_t* matCRowBatchOffset,  // {m, nMatBPerMatARow}
                  uint32_t m, uint32_t nMatB, uint32_t nMatBPerMatARow,
                  cudaStream_t stream) {
  const int CTA = 256;
  if (nMatB > CTA || (nMatBPerMatARow & (nMatBPerMatARow - 1)) != 0) {
    // inavlid nMatB or nMatBPerMatARow.
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
  uint32_t size = m * nMatBPerMatARow;
  int grid = (size + CTA - 1) / CTA;
  cudaMemsetAsync(batchedGemmM, 0, nMatB * sizeof(uint32_t), stream);
  matA_row_idx_kernel<CTA><<<grid, CTA, 0, stream>>>(
      matBIndices, matARowIndices, batchedGemmM, matCRowBatchOffset, size,
      matARowIdxRShift, m, nMatB);
}

template <int CTA>
__global__ void update_matCRowIndices_kernel(
    const uint32_t* matBIndices,     // {m, nMatBPerMatARow}
    const uint32_t* gemmMPrefixSum,  // {nMatB}
    uint32_t* matCRowOffset,         // {m, nMatBPerMatARow}
    uint32_t size) {
  int idx = blockIdx.x * CTA + threadIdx.x;
  if (idx >= size) {
    return;
  }
  asm volatile(
      "{.reg .b32 r0, r1;\n"
      " .reg .b64 r2;\n"
      " ld.global.cg.b32 r0, [%0];\n"
      " ld.global.cg.b32 r1, [%2];\n"
      " cvta.to.global.u64 r2, %1;\n"
      " mad.wide.u32 r2, r0, %3, r2;\n"
      " ld.global.ca.b32 r0, [r2];\n"
      " add.u32 r1, r1, r0;\n"
      " st.global.b32 [%2], r1;}"
      :
      : "l"(matBIndices + idx), "l"(gemmMPrefixSum), "l"(matCRowOffset + idx),
        "n"(sizeof(uint32_t)));
}

// 128Byte aligned memory size
size_t AlignedMemSize(size_t requestedSize) {
  return (requestedSize + 127) / 128 * 128;
}

void GetWorkspaceSize(size_t* hostWsSize, size_t* deviceWsSize, uint32_t m,
                      uint32_t nMatB) {
  size_t ctaIdYBarrierSize = AlignedMemSize(nMatB * sizeof(uint32_t));
  size_t batchInfoSize = AlignedMemSize(nMatB * sizeof(BatchInfo));
  size_t batchedGemmMSize = AlignedMemSize(nMatB * sizeof(uint32_t));
  size_t matARowIdxSize = AlignedMemSize(nMatB * m * sizeof(uint32_t));
  *hostWsSize = ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize;
  *deviceWsSize =
      ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize + matARowIdxSize;
}

void MoeBatchedGemm(const half* A, const half* B, const uint32_t* matBIndices,
                    half* C, uint32_t* matCRowIndices, void* hostWs,
                    size_t hostWsSize, void* deviceWs, size_t deviceWsSize,
                    uint32_t matARows, uint32_t n, uint32_t k, uint32_t nMatB,
                    uint32_t nMatBPerMatARow, cudaStream_t stream) {
  if (nMatB > 64) {
    // invalid nMatB
    return;
  }

  size_t ctaIdYBarrierSize = AlignedMemSize(nMatB * sizeof(uint32_t));
  size_t batchInfoSize = AlignedMemSize(nMatB * sizeof(BatchInfo));
  size_t batchedGemmMSize = AlignedMemSize(nMatB * sizeof(uint32_t));
  size_t matARowIdxSize = AlignedMemSize(nMatB * matARows * sizeof(uint32_t));

  if (hostWsSize < ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize ||
      deviceWsSize < ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize +
                         matARowIdxSize) {
    // invalid workspace size
    return;
  }
  // workspace:
  // host:   ctaIdYBarrier, batchInfos, batchedGemmM
  // device: ctaIdYBarrier, batchInfos, batchedGemmM, matARowIdx
  char* hWs = static_cast<char*>(hostWs);
  char* dWs = static_cast<char*>(deviceWs);
  uint32_t* hCtaIdYBarrier = reinterpret_cast<uint32_t*>(hWs);
  uint32_t* dCtaIdYBarrier = reinterpret_cast<uint32_t*>(dWs);
  BatchInfo* hBatchInfos =
      reinterpret_cast<BatchInfo*>(hWs + ctaIdYBarrierSize);
  BatchInfo* dBatchInfos =
      reinterpret_cast<BatchInfo*>(dWs + ctaIdYBarrierSize);
  uint32_t* hBatchedGemmM =
      reinterpret_cast<uint32_t*>(hWs + ctaIdYBarrierSize + batchInfoSize);
  uint32_t* dBatchedGemmM =
      reinterpret_cast<uint32_t*>(dWs + ctaIdYBarrierSize + batchInfoSize);
  uint32_t* dMatARowIdx = reinterpret_cast<uint32_t*>(
      dWs + ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize);

  // preprocess
  // matCRowBatchOffset: reuse matCRowIndices
  MatARowIndex(matBIndices, dMatARowIdx, dBatchedGemmM, matCRowIndices,
               matARows, nMatB, nMatBPerMatARow, stream);

  cudaMemcpyAsync(hBatchedGemmM, dBatchedGemmM, batchedGemmMSize,
                  cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  uint32_t gemmBatchIt = 0;
  uint32_t mAcc = 0;
  uint32_t gridYAcc = 0;
  for (uint32_t matBIt = 0; matBIt < nMatB; ++matBIt) {
    if (hBatchedGemmM[matBIt] != 0) {
      hBatchInfos[gemmBatchIt].batchId = matBIt;
      hBatchInfos[gemmBatchIt].m = hBatchedGemmM[matBIt];
      hBatchInfos[gemmBatchIt].ctaYOffset = gridYAcc;
      hBatchInfos[gemmBatchIt].COffset = mAcc * n;

      uint32_t tileY = hBatchedGemmM[matBIt] > 96   ? 128
                       : hBatchedGemmM[matBIt] > 64 ? 96
                       : hBatchedGemmM[matBIt] > 48 ? 64
                       : hBatchedGemmM[matBIt] > 32 ? 48
                       : hBatchedGemmM[matBIt] > 16 ? 32
                                                    : 16;
      uint32_t gridY = (hBatchedGemmM[matBIt] + tileY - 1) / tileY;
      gridYAcc += gridY;
      mAcc += hBatchedGemmM[matBIt];
      hCtaIdYBarrier[gemmBatchIt] = gridYAcc;
      ++gemmBatchIt;
    }
  }
  for (uint32_t i = gemmBatchIt; i < 64; ++i) {
    hCtaIdYBarrier[i] = gridYAcc;
  }

  // m exclusive prefix sum for the postprocess, reuse batchedGemmM
  for (uint32_t i = 0, mPrefix = 0; i < nMatB; ++i) {
    uint32_t m = hBatchedGemmM[i];
    hBatchedGemmM[i] = mPrefix;
    mPrefix += m;
  }

  // H2D copy: ctaIdYBarrier, batchInfos, batchedGemmM (gemmMPrefixSum)
  cudaMemcpyAsync(dWs, hWs,
                  ctaIdYBarrierSize + batchInfoSize + batchedGemmMSize,
                  cudaMemcpyDefault, stream);

  uint32_t smemSize = 120 * 1024;
  dim3 grid((n + 255) / 256, gridYAcc);
  hgemm_f32_n256_k32x5_hmma161616_ldg8_kernel<<<grid, 256, smemSize, stream>>>(
      A, B, C, dCtaIdYBarrier, dBatchInfos, dMatARowIdx, matARows, n, k,
      32 * n * sizeof(half));

  // postprocess
  update_matCRowIndices_kernel<256>
      <<<(matARows * nMatBPerMatARow + 255) / 256, 256, 0, stream>>>(
          matBIndices, dBatchedGemmM, matCRowIndices,
          matARows * nMatBPerMatARow);
}
template <>
void MoeBatchedGemmLauncher<float>(
    const float* A, const float* B, const uint32_t* matBIndices, float* C,
    uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  // TODO
}
#ifdef ENABLE_FP16
template <>
void MoeBatchedGemmLauncher<half>(
    const half* A, const half* B, const uint32_t* matBIndices, half* C,
    uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  MoeBatchedGemm(A, B, matBIndices, C, matCRowIndices, hostWs, hostWsSize,
                 deviceWs, deviceWsSize, matARows, n, k, nMatB, nMatBPerMatARow,
                 stream);
}
#endif
#ifdef ENABLE_BF16
template <>
void MoeBatchedGemmLauncher<hie::bfloat16>(
    const hie::bfloat16* A, const hie::bfloat16* B, const uint32_t* matBIndices,
    hie::bfloat16* C, uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize,
    void* deviceWs, size_t deviceWsSize, uint32_t matARows, uint32_t n,
    uint32_t k, uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  // TODO
}
#endif
}  // namespace cuda
}  // namespace allspark