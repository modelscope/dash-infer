/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_256x128x64.cu
 */

#include "cuda/hie/cuda_activation.hpp"
#include "cuda/hie/cuda_intdivider.hpp"
#include "fused_matmul_256x128x64.hpp"
// #include "hie_cmath_wrapper.hpp"
#include <utility>
namespace hie {

namespace dynamic_quant_matmul_fused {

using std::int32_t;
using std::int8_t;
using std::uint32_t;

#if __CUDA_ARCH__ >= 720 && CUDART_VERSION >= 10020

__device__ __forceinline__ void ldg64_0(uint32_t& r0, uint32_t& r1,
                                        const void* ptr, bool guard) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    !defined(__HGGCCC__)
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @p ld.global.cg.L2::128B.v2.u32 {%0, %1}, [%2];}\n"
      : "=r"(r0), "=r"(r1)
      : "l"(ptr), "r"((int)guard));
#else
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @p ld.global.cg.v2.u32 {%0, %1}, [%2];}\n"
      : "=r"(r0), "=r"(r1)
      : "l"(ptr), "r"((int)guard));
#endif
}

__device__ __forceinline__ void ldg64(uint32_t& r0, uint32_t& r1,
                                      const void* ptr, bool guard) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    !defined(__HGGCCC__)
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p ld.global.cg.L2::128B.v2.u32 {%0, %1}, [%2];}\n"
      : "=r"(r0), "=r"(r1)
      : "l"(ptr), "r"((int)guard));
#else
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p ld.global.cg.v2.u32 {%0, %1}, [%2];}\n"
      : "=r"(r0), "=r"(r1)
      : "l"(ptr), "r"((int)guard));
#endif
}

__device__ __forceinline__ void ldg128_0(uint32_t& r0, uint32_t& r1,
                                         uint32_t& r2, uint32_t& r3,
                                         const void* ptr, bool guard) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    !defined(__HGGCCC__)
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @!p mov.b32 %2, 0;\n"
      " @!p mov.b32 %3, 0;\n"
      " @p ld.global.cg.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(ptr), "r"((int)guard));
#else
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @!p mov.b32 %0, 0;\n"
      " @!p mov.b32 %1, 0;\n"
      " @!p mov.b32 %2, 0;\n"
      " @!p mov.b32 %3, 0;\n"
      " @p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(ptr), "r"((int)guard));
#endif
}

__device__ __forceinline__ void ldg128(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                       uint32_t& r3, const void* ptr,
                                       bool guard) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    !defined(__HGGCCC__)
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @p ld.global.cg.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(ptr), "r"((int)guard));
#else
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(ptr), "r"((int)guard));
#endif
}

__device__ __forceinline__ void stg32(const int32_t& reg, const void* ptr,
                                      bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p st.global.b32 [%0], %1;}\n"
      :
      : "l"(ptr), "r"(reg), "r"((int)guard));
}

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

template <typename T>
__device__ __forceinline__ void sts64(const T& r0, const T& r1,
                                      const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "sts64: invalid T");

  asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
               :
               : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(r0)),
                 "r"(reinterpret_cast<const uint32_t&>(r1)));
}

template <typename T>
__device__ __forceinline__ void sts128(const T& r0, const T& r1, const T& r2,
                                       const T& r3, const uint32_t& addr) {
  static_assert(sizeof(T) == 4, "sts128: invalid T");

  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(r0)),
                 "r"(reinterpret_cast<const uint32_t&>(r1)),
                 "r"(reinterpret_cast<const uint32_t&>(r2)),
                 "r"(reinterpret_cast<const uint32_t&>(r3)));
}

__device__ __forceinline__ void ldsm4(int32_t& r0, int32_t& r1, int32_t& r2,
                                      int32_t& r3, const uint32_t& addr) {
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
#endif
}

__device__ __forceinline__ void ldsm4_transpose(int32_t& r0, int32_t& r1,
                                                int32_t& r2, int32_t& r3,
                                                const uint32_t& addr) {
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
#endif
}

__device__ __forceinline__ void ldg_transpose(uint32_t (&d)[4],
                                              const uint32_t (&s0)[2],
                                              const uint32_t (&s1)[2]) {
  asm("prmt.b32 %0, %4, %6, 0x00005140;\n"
      "prmt.b32 %1, %4, %6, 0x00007362;\n"
      "prmt.b32 %2, %5, %7, 0x00005140;\n"
      "prmt.b32 %3, %5, %7, 0x00007362;\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(s0[0]), "r"(s0[1]), "r"(s1[0]), "r"(s1[1]));
}

__device__ __forceinline__ void mma8816(const int32_t& a, const int32_t& b,
                                        const int32_t& c0, const int32_t& c1,
                                        int32_t& d0, int32_t& d1) {
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, "
      "{%4, %5};"
      : "=r"(d0), "=r"(d1)
      : "r"(a), "r"(b), "r"(c0), "r"(c1));
#endif
}

struct gemm_proc_mma {
  __device__ __forceinline__ void init(const int8_t* asmem, const int8_t* bsmem,
                                       uint32_t warpid, uint32_t laneid) {
    // clear c for mma
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        creg[i][j][0] = 0;
        creg[i][j][1] = 0;
      }
    }
// ldsm offset
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      aaddr[i] = smem_u32addr(asmem + i * 256 * 16 + (warpid / 2) * 64 * 16 +
                              (laneid / 8 * 8 + (laneid % 8 + i * 2) % 8) * 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      baddr[i] = smem_u32addr(bsmem + (laneid % 8) * 256 + (warpid % 2) * 128 +
                              (laneid % 8 + laneid / 8 + i * 4) % 8 * 16);
    }
    // ldsm tictoc offset
    alds_offset = 0;
    blds_offset = 0;
  }

  __device__ __forceinline__ void alds(uint32_t pipe) {
    ldsm4(areg[(pipe + 1) % 2][0], areg[(pipe + 1) % 2][1],
          areg[(pipe + 1) % 2][2], areg[(pipe + 1) % 2][3],
          aaddr[(pipe + 1) % 4] + alds_offset);
    ldsm4(areg[(pipe + 1) % 2][4], areg[(pipe + 1) % 2][5],
          areg[(pipe + 1) % 2][6], areg[(pipe + 1) % 2][7],
          aaddr[(pipe + 1) % 4] + alds_offset + 32 * 16 * sizeof(int8_t));
  }

  __device__ __forceinline__ void blds(uint32_t pipe) {
    ldsm4_transpose(breg[(pipe + 1) % 2][0], breg[(pipe + 1) % 2][1],
                    breg[(pipe + 1) % 2][2], breg[(pipe + 1) % 2][3],
                    baddr[0] + blds_offset + (pipe + 1) % 4 * (8 * 256));
    ldsm4_transpose(breg[(pipe + 1) % 2][4], breg[(pipe + 1) % 2][5],
                    breg[(pipe + 1) % 2][6], breg[(pipe + 1) % 2][7],
                    baddr[1] + blds_offset + (pipe + 1) % 4 * (8 * 256));
  }

  __device__ __forceinline__ void csts(int32_t cit, uint32_t caddr) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sts64(creg[cit * 2 + i][j][0], creg[cit * 2 + i][j][1],
              caddr + (i * 8 * 72 + j * 8) * sizeof(int32_t));
      }
    }
  }

  __device__ __forceinline__ void csts_d12(int32_t d0, uint32_t caddr) {
    constexpr int32_t max_d1 = 8;  // regs [8][8][2]
    constexpr int32_t max_d2 = 2;
    constexpr int32_t max_tx = 4;  // mma_tid_x max = 4;
#pragma unroll
    for (int32_t d1 = 0; d1 < max_d1; d1++) {
      sts64(creg[d0][d1][0], creg[d0][d1][1],
            caddr + d1 * max_tx * max_d2 * sizeof(int32_t));
    }
  }

  __device__ __forceinline__ void tictoc() {
    // switch lds-ptr.
    alds_offset ^= 0x4000;
    blds_offset ^= 0x2000;
  }

  __device__ __forceinline__ void mma(uint32_t pipe) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mma8816(areg[pipe % 2][i], breg[pipe % 2][j], creg[i][j][0],
                creg[i][j][1], creg[i][j][0], creg[i][j][1]);
      }
    }
  }

  int32_t areg[2][8], breg[2][8], creg[8][8][2];
  uint32_t aaddr[4], baddr[2];
  uint32_t alds_offset, blds_offset;
};

struct gemm_proc_lda {
  __device__ __forceinline__ void init(const int8_t* a, uint32_t tiley,
                                       int32_t m, int32_t k) {
    uint32_t warpid = threadIdx.x / 32;
    uint32_t mhead = tiley * 256 + threadIdx.x / 4;
    aldg = mhead * k + (threadIdx.x % 4) * 16;
    asmem = smem_u32addr(
        a + (threadIdx.x % 4) * (256 * 16) +
        (warpid * 8 + (threadIdx.x / 4 + threadIdx.x % 4 * 2) % 8) * 16);
    aguard = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t m_idx = mhead + i * 64;
      if (m_idx < m) aguard |= 1u << i;
    }
  }

  __device__ __forceinline__ void sts() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts128(areg[i][0], areg[i][1], areg[i][2], areg[i][3],
             asmem + i * 64 * 16 * sizeof(int8_t));
    }
  }

  __device__ __forceinline__ void ldg0(const int8_t* a, int32_t m, int32_t k,
                                       uint32_t kfirst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bool guard = (aguard & (1u << i)) != 0 && (threadIdx.x % 4) * 16 < kfirst;
      ldg128_0(areg[i][0], areg[i][1], areg[i][2], areg[i][3],
               a + aldg + i * (64 * k), guard);
    }
  }

  __device__ __forceinline__ void ldg(const int8_t* a, int32_t m, int32_t k,
                                      uint32_t kid) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg128(areg[i][0], areg[i][1], areg[i][2], areg[i][3],
             a + aldg + kid * 1 + i * (64 * k), (aguard & (1u << i)) != 0);
    }
  }

  __device__ __forceinline__ void tictoc() { asmem ^= 0x4000; }

  uint32_t aguard, aldg, asmem;
  uint32_t areg[4][4];
};

struct gemm_proc_ldb {
  __device__ __forceinline__ void init(const int8_t* b, uint32_t tilex,
                                       int32_t k, int32_t n) {
    uint32_t nhead = tilex * 128 + threadIdx.x % 16 * 8;
    bldg = (threadIdx.x / 16 * 2) * n + nhead;
    bsmem = smem_u32addr(b + (threadIdx.x / 16) * 256 +
                         (threadIdx.x / 16 % 8 + threadIdx.x % 8) % 8 * 16 +
                         (threadIdx.x % 16) / 8 * 128);
    bguard = nhead < n;
  }

  __device__ __forceinline__ void sts() {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg_transpose(bsts[i], breg[i][0], breg[i][1]);
      sts128(bsts[i][0], bsts[i][1], bsts[i][2], bsts[i][3],
             bsmem + i * 16 * 256 * sizeof(int8_t));
    }
  }

  __device__ __forceinline__ void ldg0(const int8_t* b, int32_t k, int32_t n,
                                       uint32_t kfirst) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg64_0(breg[i][0][0], breg[i][0][1], b + bldg + i * 32 * n,
              bguard && threadIdx.x / 16 * 2 + i * 32 < kfirst);
      ldg64_0(breg[i][1][0], breg[i][1][1], b + bldg + i * 32 * n + n,
              bguard && threadIdx.x / 16 * 2 + i * 32 + 1 < kfirst);
    }
  }

  __device__ __forceinline__ void ldg(const int8_t* b, int32_t k, int32_t n,
                                      uint32_t kid) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg64(breg[i][0][0], breg[i][0][1], b + bldg + kid * n + (32 * n) * i,
            bguard);
      ldg64(breg[i][1][0], breg[i][1][1], b + bldg + kid * n + (32 * n) * i + n,
            bguard);
    }
  }

  __device__ __forceinline__ void tictoc() { bsmem ^= 0x2000; }

  bool bguard;
  uint32_t bldg, bsmem;
  uint32_t breg[2][2][2];
  uint32_t bsts[2][4];
};

template <typename FT>
struct gemm_proc_stc {
  __device__ __forceinline__ void init(uint32_t tilex, uint32_t tiley,
                                       uint32_t warpid, uint32_t laneid) {
    midx = static_cast<int32_t>(tiley * 256 + (warpid / 2) * 64);
    nidx = static_cast<int32_t>(tilex * 128 + (warpid % 2) * 64 + laneid);
  }

  __device__ __forceinline__ void load(int32_t n, float alpha, float beta,
                                       const int8_t* bzero,
                                       const int32_t* breduce,
                                       const float* bscale, const FT* bias) {
#pragma unroll
    for (int nloop = 0; nloop < 2; nloop++) {
      int32_t ncurr = nidx + nloop * 32;
      if (ncurr < n) {
        qbias[nloop] = bias == nullptr || beta == 0.f
                           ? 0.f
                           : beta * static_cast<float>(bias[ncurr]);
        qbzero[nloop] = static_cast<int32_t>(bzero[ncurr]);
        qbrsum[nloop] = breduce[ncurr];
        qbscale[nloop] = alpha * bscale[ncurr];
      }
    }
  }

  __device__ __forceinline__ void stg(FT* c, int32_t cit, int32_t m, int32_t n,
                                      int32_t k, const int32_t* csmem,
                                      const int8_t* azero,
                                      const int32_t* areduce,
                                      const float* ascale) {
    // #pragma unroll
    for (int32_t mm = 0; mm < 16; ++mm) {
      int32_t mcurr = midx + cit * 16 + mm;
      if (mcurr < m) {
        int32_t qazero = static_cast<int32_t>(azero[mcurr]);
        int32_t qarsum = areduce[mcurr];
        float qascale = ascale[mcurr];

        float facc[2];
        facc[0] = static_cast<float>(csmem[mm * 72] + k * qazero * qbzero[0] -
                                     qarsum * qbzero[0] - qbrsum[0] * qazero);
        facc[1] =
            static_cast<float>(csmem[mm * 72 + 32] + k * qazero * qbzero[1] -
                               qarsum * qbzero[1] - qbrsum[1] * qazero);

        if (nidx < n) {
          facc[0] *= qbscale[0] * qascale;
          facc[0] += qbias[0];
          c[mcurr * n + nidx] = static_cast<FT>(facc[0]);
        }

        if (nidx + 32 < n) {
          facc[1] *= qbscale[1] * qascale;
          facc[1] += qbias[1];
          c[mcurr * n + nidx + 32] = static_cast<FT>(facc[1]);
        }
      }
    }
  }

  int32_t midx, nidx;
  float qbias[2];
  float qbscale[2];
  int32_t qbzero[2];
  int32_t qbrsum[2];
};

// -------------------------------------------------------------------
// ldg16, 256x128_64x64 tile
// n % 8 == 0, k % 16 == 0
// M_TILE = 256, N_TILE = 128, K_TILE = 64, blockDim = (256, 1, 1)
// -------------------------------------------------------------------
template <typename FT, template <class> class ACT>
__global__
__launch_bounds__(256) void gemm_nn_256x128x64_activation_fused_kernel(
    const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bqptr, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, int32_t gemm_m, int32_t gemm_n,
    int32_t gemm_k, float alpha, float beta, uint32_t full_wave_blocks,
    uint32_t wave_y, hie::internal::IntDivModer<uint32_t> wave_size_divmod,
    hie::internal::IntDivModer<uint32_t> wave_y_divmod,
    hie::internal::IntDivModer<uint32_t> last_wave_y_divmod, const FT* bias,
    FT* cgptr) {
  __shared__ char __align__(32 * 1024) smem[48 * 1024];
  int8_t* asptr = reinterpret_cast<int8_t*>(smem);
  int8_t* bsptr = reinterpret_cast<int8_t*>(smem + 32 * 1024);

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // register fragment
  gemm_proc_mma ab2c;
  ab2c.init(asptr, bsptr, warp_id, lane_id);

  // remap thread block to matrix_C tile to increase L2 cache hit rate
  auto wave_size_dm = wave_size_divmod.divmod(blockIdx.x);
  auto wave_y_dm = blockIdx.x < full_wave_blocks
                       ? wave_y_divmod.divmod(wave_size_dm.mod)
                       : last_wave_y_divmod.divmod(wave_size_dm.mod);
  uint32_t tile_x = wave_y_dm.div;
  uint32_t tile_y = wave_size_dm.div * wave_y + wave_y_dm.mod;

  // ldg register buffer
  gemm_proc_lda g2a;
  g2a.init(asptr, tile_y, gemm_m, gemm_k);
  gemm_proc_ldb g2b;
  g2b.init(bsptr, tile_x, gemm_k, gemm_n);

  constexpr uint32_t k_step = 64;
  uint32_t k_tiles = (gemm_k + k_step - 1) / k_step - 1;
  uint32_t k_first = gemm_k - k_tiles * k_step;

  // load 1'st A_tile & B_tile
  g2a.ldg0(aqptr, gemm_m, gemm_k, k_first);
  g2a.sts();
  g2a.tictoc();
  g2b.ldg0(bqptr, gemm_k, gemm_n, k_first);
  g2b.sts();
  g2b.tictoc();
  __syncthreads();
  // load 1'st warp tile
  ab2c.alds(3);
  ab2c.blds(3);

  // GEMM main loop over K
  for (uint32_t k_index = k_first; k_index < gemm_k; k_index += k_step) {
#pragma unroll
    for (uint32_t tt = 0; tt < 4; ++tt) {
      // store A_ldg_reg & B_ldg_reg to smem
      if (tt == 3) {
        g2a.sts();
        g2a.tictoc();
        g2b.sts();
        g2b.tictoc();
        ab2c.tictoc();
        __syncthreads();
      }
      // load warp tile from smem
      ab2c.alds(tt);
      ab2c.blds(tt);
      // load A_tile & B_tile from gmem
      if (tt == 0) {
        g2a.ldg(aqptr, gemm_m, gemm_k, k_index);
        g2b.ldg(bqptr, gemm_k, gemm_n, k_index);
      }
      // mma
      ab2c.mma(tt);
    }
  }

// GEMM loop for final tile
#pragma unroll
  for (uint32_t tt = 0; tt < 4; ++tt) {
    // load warp tile from smem
    if (tt < 3) {
      ab2c.alds(tt);
      ab2c.blds(tt);
    }
    // mma
    ab2c.mma(tt);
  }

  /* mma results in regs.
   *
   * for 256 threads.
   *      - [4]. warps-m        m - wpm
   *          [2]. warps-n      n - wpn
   *            [8]. tid-y      m - ty
   *              [4]. tid-x    n - tx
   * for single threads. creg[8][8][2]
   *      - [8]. regD0          m - d0
   *          [8]. regD1        n - d1
   *            [2]. regD2      n - d2
   *
   * for M-256 / N-128 block
   *           64     128     192     256
   *  + - - - + - - - + - - - + - - - + > M
   *  | wp0   | wp2   | wp4   | wp6   |
   *  |       |       |       |       |
   *  + - - - + - - - + - - - + - - - +  64
   *  | wp1   | wp3   | wp5   | wp7   |
   *  |       |       |       |       |
   *  + - - - + - - - + - - - + - - - + 128
   *  v
   *  N
   *
   * for single wp, eg. wp0
   *  mma-tid-y 0-8
   *  |<--->|
   *  d0[0] d0[1] d0[2] d0[3] d0[4] d0[5] d0[6] d0[7] d0[8]
   *         8    16    24    32    40    48    56    64
   *  + --- + mma-tid-x[0] -- d2[0-1] --- + --- + --- + > M
   *  + --- + mma-tid-x[1]                ^           |
   *  + --- + mma-tid-x[2]                | d1[0]     +   4
   *  + --- + mma-tid-x[3]                v           |
   *  + --- + --- + --- + --- + --- + --- + --- + --- +   8
   *  |                                   ^           |
   *  +                                   | d1[2]     +  12
   *  |                                   v           |
   *  + --- + --- + --- + --- + --- + --- + --- + --- +  16
   *  |                                   ^           |
   *  +                                   | d1[3]     +  20
   *  |                                               |
   *     ...                                 ...
   *  |                                               |
   *  +                                   | d1[7]     +  60
   *  |                                   v           |
   *  + --- + --- + --- + --- + --- + --- + --- + --- +  64
   *  v
   *  N
   */

  // C_tile write back
  int32_t* csptr = (int32_t*)smem + warp_id * 16 * 72;
  uint32_t mma_tidx = lane_id % 4;
  uint32_t mma_tidy = lane_id / 4;

  constexpr int32_t warpn = 32;
  constexpr int32_t max_d0 = 8;
  // constexpr int32_t max_d1 = 8;
  constexpr int32_t max_d2 = 2;
  // constexpr int32_t max_tidx = 4;
  constexpr int32_t max_tidy = 8;
  constexpr int32_t stride_dimm = 72;
  constexpr int32_t stride_dimn = 1;
  uint32_t csts_warp_offset = smem_u32addr(csptr + mma_tidy * stride_dimm +
                                           mma_tidx * max_d2 * stride_dimn);

  int32_t mhead = tile_y * 256 + (warp_id / 2) * 64;
  int32_t nhead = tile_x * 128 + (warp_id % 2) * 64;

  int32_t b_quant_zero[2];
  int32_t b_quant_rsum[2];
  float b_quant_scales[2];
  float fbias[2];
#pragma unroll
  for (int32_t id = 0; id < 2; id++) {
    int32_t nidx = nhead + (id * warpn + lane_id);
    b_quant_zero[id] = nidx < gemm_n ? static_cast<int32_t>(bzero[nidx]) : 0;
    b_quant_rsum[id] = nidx < gemm_n ? breduce[nidx] : 0;
    b_quant_scales[id] = nidx < gemm_n ? bscale[nidx] : 1.f;
    fbias[id] =
        nidx < gemm_n && bias != nullptr ? static_cast<float>(bias[nidx]) : 0.f;
  }

  // #pragma unroll
  for (int32_t d0 = 0; d0 < max_d0; d0++) {
    __syncthreads();
    // mma result to shared
    ab2c.csts_d12(d0, csts_warp_offset);

    __syncthreads();
    // load to reg for compute
    for (int ty = 0; ty < max_tidy; ty++) {
      int32_t midx = mhead + d0 * max_tidy + ty;
      if (midx < gemm_m) {
        int32_t a_quant_zero = static_cast<int32_t>(azero[midx]);
        int32_t a_quant_rsum = areduce[midx];
        float a_quant_scales = ascale[midx];

        float fact[2];
#pragma unroll
        for (int32_t id = 0; id < 2; id++) {
          int32_t cint =
              csptr[ty * stride_dimm + (id * warpn + lane_id) * stride_dimn];
          cint += gemm_k * a_quant_zero * b_quant_zero[id] -
                  a_quant_rsum * b_quant_zero[id] -
                  a_quant_zero * b_quant_rsum[id];
          fact[id] = fbias[id] + a_quant_scales * b_quant_scales[id] *
                                     static_cast<float>(cint);
          fact[id] = ACT<float>::Op(fact[id]);
        }

#pragma unroll
        for (int32_t id = 0; id < 2; id++) {
          int32_t nidx = nhead + (id * warpn + lane_id);
          if (nidx < gemm_n) {
            cgptr[midx * gemm_n + nidx] = static_cast<FT>(fact[id]);
          }
        }
      }
    }
  }
}

// none-activation.
template <typename FT>
__global__ __launch_bounds__(256) void gemm_nn_256x128x64_fused_kernel(
    const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bqptr, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const FT* bias, FT* cgptr,
    int32_t gemm_m, int32_t gemm_n, int32_t gemm_k, float alpha, float beta,
    uint32_t full_wave_blocks, uint32_t wave_y,
    hie::internal::IntDivModer<uint32_t> wave_size_divmod,
    hie::internal::IntDivModer<uint32_t> wave_y_divmod,
    hie::internal::IntDivModer<uint32_t> last_wave_y_divmod) {
  __shared__ char __align__(32 * 1024) smem[48 * 1024];
  int8_t* asptr = reinterpret_cast<int8_t*>(smem);
  int8_t* bsptr = reinterpret_cast<int8_t*>(smem + 32 * 1024);

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // register fragment
  gemm_proc_mma ab2c;
  ab2c.init(asptr, bsptr, warp_id, lane_id);

  // remap thread block to matrix_C tile to increase L2 cache hit rate
  auto wave_size_dm = wave_size_divmod.divmod(blockIdx.x);
  auto wave_y_dm = blockIdx.x < full_wave_blocks
                       ? wave_y_divmod.divmod(wave_size_dm.mod)
                       : last_wave_y_divmod.divmod(wave_size_dm.mod);
  uint32_t tile_x = wave_y_dm.div;
  uint32_t tile_y = wave_size_dm.div * wave_y + wave_y_dm.mod;

  // ldg register buffer
  gemm_proc_lda g2a;
  g2a.init(asptr, tile_y, gemm_m, gemm_k);
  gemm_proc_ldb g2b;
  g2b.init(bsptr, tile_x, gemm_k, gemm_n);

  constexpr uint32_t k_step = 64;
  uint32_t k_tiles = (gemm_k + k_step - 1) / k_step - 1;
  uint32_t k_first = gemm_k - k_tiles * k_step;

  // load 1'st A_tile & B_tile
  g2a.ldg0(aqptr, gemm_m, gemm_k, k_first);
  g2a.sts();
  g2a.tictoc();
  g2b.ldg0(bqptr, gemm_k, gemm_n, k_first);
  g2b.sts();
  g2b.tictoc();
  __syncthreads();
  // load 1'st warp tile
  ab2c.alds(3);
  ab2c.blds(3);

  // GEMM main loop over K
  for (uint32_t k_index = k_first; k_index < gemm_k; k_index += k_step) {
#pragma unroll
    for (uint32_t tt = 0; tt < 4; ++tt) {
      // store A_ldg_reg & B_ldg_reg to smem
      if (tt == 3) {
        g2a.sts();
        g2a.tictoc();
        g2b.sts();
        g2b.tictoc();
        ab2c.tictoc();
        __syncthreads();
      }
      // load warp tile from smem
      ab2c.alds(tt);
      ab2c.blds(tt);
      // load A_tile & B_tile from gmem
      if (tt == 0) {
        g2a.ldg(aqptr, gemm_m, gemm_k, k_index);
        g2b.ldg(bqptr, gemm_k, gemm_n, k_index);
      }
      // mma
      ab2c.mma(tt);
    }
  }

// GEMM loop for final tile
#pragma unroll
  for (uint32_t tt = 0; tt < 4; ++tt) {
    // load warp tile from smem
    if (tt < 3) {
      ab2c.alds(tt);
      ab2c.blds(tt);
    }
    // mma
    ab2c.mma(tt);
  }
  // C_tile write back
  int32_t* csptr = (int32_t*)smem + warp_id * 16 * 72;
  uint32_t mma_tidx = lane_id % 4;
  uint32_t mma_tidy = lane_id / 4;

  uint32_t csts_addr = smem_u32addr(csptr + mma_tidy * 72 + mma_tidx * 2);
  const int32_t* clds_ptr = csptr + lane_id;  // lane_id = threadIdx.x % 32;

  gemm_proc_stc<FT> c2g;
  c2g.init(tile_x, tile_y, warp_id, lane_id);
  c2g.load(gemm_n, alpha, beta, bzero, breduce, bscale, bias);

#pragma unroll
  for (int32_t cit = 0; cit < 4; ++cit) {
    __syncthreads();
    ab2c.csts(cit, csts_addr);

    __syncthreads();
    c2g.stg(cgptr, cit, gemm_m, gemm_n, gemm_k, clds_ptr, azero, areduce,
            ascale);
  }
}

#else  // __CUDA_ARCH__ >= 720 && CUDART_VERSION >= 10020

template <typename FT, template <class> class ACT>
__global__
__launch_bounds__(256) void gemm_nn_256x128x64_activation_fused_kernel(
    const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bqptr, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, int32_t gemm_m, int32_t gemm_n,
    int32_t gemm_k, float alpha, float beta, uint32_t full_wave_blocks,
    uint32_t wave_y, hie::internal::IntDivModer<uint32_t> wave_size_divmod,
    hie::internal::IntDivModer<uint32_t> wave_y_divmod,
    hie::internal::IntDivModer<uint32_t> last_wave_y_divmod, const FT* bias,
    FT* cgptr) {}

// bias only
template <typename FT>
__global__ __launch_bounds__(256) void gemm_nn_256x128x64_fused_kernel(
    const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bqptr, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const FT* bias, FT* cgptr,
    int32_t gemm_m, int32_t gemm_n, int32_t gemm_k, float alpha, float beta,
    uint32_t full_wave_blocks, uint32_t wave_y,
    hie::internal::IntDivModer<uint32_t> wave_size_divmod,
    hie::internal::IntDivModer<uint32_t> wave_y_divmod,
    hie::internal::IntDivModer<uint32_t> last_wave_y_divmod) {}

#endif  // __CUDA_ARCH__ >= 720 && CUDART_VERSION >= 10020

// launch
template <typename FT>
struct gemm_nn_256x128x64_activation_fused_impl {
  static int calws(std::string act_string, int smv, int m, int n, int k) {
#if CUDART_VERSION >= 10020
    // n, k need aligned.
    if (smv < 0x0702) {
      return -1;
    }
    if (n % 8 == 0 && k % 16 == 0) {
      return 0;
    }
#endif  // CUDART_VERSION >= 10020
    return -1;
  }

  void operator()(cudaStream_t stream, int smcount, std::string act_string,
                  const int8_t* aquant, const int8_t* azero,
                  const int32_t* areduce, const float* ascale,
                  const int8_t* bquant, const int8_t* bzero,
                  const int32_t* breduce, const float* bscale, const void* bias,
                  int m, int n, int k, float alpha, float beta, void* c) {
    if (act_string == "NONE") {
      non_activation(stream, smcount, aquant, azero, areduce, ascale, bquant,
                     bzero, breduce, bscale, bias, m, n, k, alpha, beta, c);
    } else if (act_string == "RELU") {
      gemm_nn_256x128x64_activation_fused_inner<activation::Relu>()(
          stream, smcount, aquant, azero, areduce, ascale, bquant, bzero,
          breduce, bscale, bias, m, n, k, alpha, beta, c);
    } else if (act_string == "TANH") {
      gemm_nn_256x128x64_activation_fused_inner<activation::Tanh>()(
          stream, smcount, aquant, azero, areduce, ascale, bquant, bzero,
          breduce, bscale, bias, m, n, k, alpha, beta, c);
    } else if (act_string == "GELU") {
      gemm_nn_256x128x64_activation_fused_inner<activation::Gelu>()(
          stream, smcount, aquant, azero, areduce, ascale, bquant, bzero,
          breduce, bscale, bias, m, n, k, alpha, beta, c);
    } else if (act_string == "GELU_TANH") {
      gemm_nn_256x128x64_activation_fused_inner<activation::GeluTanh>()(
          stream, smcount, aquant, azero, areduce, ascale, bquant, bzero,
          breduce, bscale, bias, m, n, k, alpha, beta, c);
    } else {
      LOG(ERROR) << "ACTIVE FAIL. current act = " << act_string;
    }
  }

  template <template <class> class ACT>
  struct gemm_nn_256x128x64_activation_fused_inner {
    void operator()(cudaStream_t stream, int smcount, const int8_t* aquant,
                    const int8_t* azero, const int32_t* areduce,
                    const float* ascale, const int8_t* bquant,
                    const int8_t* bzero, const int32_t* breduce,
                    const float* bscale, const void* bias, int m, int n, int k,
                    float alpha, float beta, void* c) {
      uint32_t grid_x = (n + 127) / 128;
      uint32_t grid_y = (m + 255) / 256;
      uint32_t grid = grid_x * grid_y;

      // only 1 thread block per sm, limited by register and shared memory
      uint32_t blocks_per_wave = smcount;

      // wave_y * (wave_y * 2) == blocks_per_wave
      uint32_t wave_y = static_cast<uint32_t>(sqrt(float(blocks_per_wave / 2)));
      if (wave_y > grid_y) wave_y = grid_y;
      uint32_t last_wave_y = grid_y % wave_y == 0 ? 1 : grid_y % wave_y;

      uint32_t wave_size = wave_y * grid_x;
      uint32_t full_wave_blocks = grid - grid % wave_size;
      internal::IntDivModer<uint32_t> wave_size_divmod(wave_size);
      internal::IntDivModer<uint32_t> wave_y_divmod(wave_y);
      internal::IntDivModer<uint32_t> last_wave_y_divmod(last_wave_y);

      const FT* tbias = static_cast<const FT*>(bias);
      FT* tc = static_cast<FT*>(c);

      gemm_nn_256x128x64_activation_fused_kernel<FT, ACT>
          <<<grid, 256, 0, stream>>>(
              aquant, azero, areduce, ascale, bquant, bzero, breduce, bscale, m,
              n, k, alpha, beta, full_wave_blocks, wave_y, wave_size_divmod,
              wave_y_divmod, last_wave_y_divmod, tbias, tc);
    }
  };

  void non_activation(cudaStream_t stream, int smcount, const int8_t* aquant,
                      const int8_t* azero, const int32_t* areduce,
                      const float* ascale, const int8_t* bquant,
                      const int8_t* bzero, const int32_t* breduce,
                      const float* bscale, const void* bias, int m, int n,
                      int k, float alpha, float beta, void* c) {
    uint32_t grid_x = (n + 127) / 128;
    uint32_t grid_y = (m + 255) / 256;
    uint32_t grid = grid_x * grid_y;

    // only 1 thread block per sm, limited by register and shared memory
    uint32_t blocks_per_wave = smcount;

    // wave_y * (wave_y * 2) == blocks_per_wave
    uint32_t wave_y = static_cast<uint32_t>(sqrt(float(blocks_per_wave / 2)));
    if (wave_y > grid_y) wave_y = grid_y;
    uint32_t last_wave_y = grid_y % wave_y == 0 ? 1 : grid_y % wave_y;

    uint32_t wave_size = wave_y * grid_x;
    uint32_t full_wave_blocks = grid - grid % wave_size;
    internal::IntDivModer<uint32_t> wave_size_divmod(wave_size);
    internal::IntDivModer<uint32_t> wave_y_divmod(wave_y);
    internal::IntDivModer<uint32_t> last_wave_y_divmod(last_wave_y);

    const FT* tbias = static_cast<const FT*>(bias);
    FT* tc = static_cast<FT*>(c);

    gemm_nn_256x128x64_fused_kernel<FT><<<grid, 256, 0, stream>>>(
        aquant, azero, areduce, ascale, bquant, bzero, breduce, bscale, tbias,
        tc, m, n, k, alpha, beta, full_wave_blocks, wave_y, wave_size_divmod,
        wave_y_divmod, last_wave_y_divmod);
  }
};

}  // namespace dynamic_quant_matmul_fused

int64_t dynamicQuantMatMulActivationFused256x128x64WorkSpace(
    hie::DataType dtype, std::string act, int sm_ver, int sm_cnt, int m, int n,
    int k) {
  if (dtype == hie::DataType::FLOAT) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_256x128x64_activation_fused_impl<
            float>::calws(act, sm_ver, m, n, k));
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_256x128x64_activation_fused_impl<
            half>::calws(act, sm_ver, m, n, k));
  }
#endif

  return -1;
}

void dynamicQuantMatMulActivationFused256x128x64Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c) {
  if (dtype == hie::DataType::FLOAT) {
    dynamic_quant_matmul_fused::gemm_nn_256x128x64_activation_fused_impl<
        float>()(stream, sm_cnt, act, aquant, azero, areduce, ascale, bquant,
                 bzero, breduce, bscale, bias, m, n, k, alpha, beta, c);
    return;
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    dynamic_quant_matmul_fused::gemm_nn_256x128x64_activation_fused_impl<
        half>()(stream, sm_cnt, act, aquant, azero, areduce, ascale, bquant,
                bzero, breduce, bscale, bias, m, n, k, alpha, beta, c);
    return;
  }
#endif

  return;
}

}  // namespace hie
