/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_HIE.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {
// #if __CUDA_ARCH__ >= 720 && CUDART_VERSION >= 10020
__device__ __forceinline__ void ldg_8(uint2& dst, const void* __restrict__ ptr,
                                      bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %3, 0;\n"
      " @p ld.global.nc.v2.u32 {%0, %1}, [%2];}\n"
      : "=r"(dst.x), "=r"(dst.y)
      : "l"(ptr), "r"((int)guard));
}

__device__ __forceinline__ void ldg_16(uint4& dst, const void* __restrict__ ptr,
                                       bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %5, 0;\n"
      " @p ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];}\n"
      : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
      : "l"(ptr), "r"((int)guard));
}

// ldg8, sts16
__device__ __forceinline__ void ldg_trans(uint4& d, const uint2& s0,
                                          const uint2& s1) {
  asm("prmt.b32 %0, %4, %6, 0x00005140;\n"
      "prmt.b32 %1, %4, %6, 0x00007362;\n"
      "prmt.b32 %2, %5, %7, 0x00005140;\n"
      "prmt.b32 %3, %5, %7, 0x00007362;\n"
      : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
      : "r"(s0.x), "r"(s0.y), "r"(s1.x), "r"(s1.y));
}

__device__ __forceinline__ uint32_t get_u32_smem_ptr(const void* smemptr) {
  uint32_t u32_ptr;
  asm("{ .reg .u64 smem_ptr;\n"
      "  cvta.to.shared.u64 smem_ptr, %1;\n"
      "  cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(u32_ptr)
      : "l"(smemptr));

  return u32_ptr;
}

__device__ __forceinline__ void ldsm_4(int32_t& m0, int32_t& m1, int32_t& m2,
                                       int32_t& m3, const void* smemptr) {
  uint32_t u32_smem_ptr = get_u32_smem_ptr(smemptr);
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(m0), "=r"(m1), "=r"(m2), "=r"(m3)
      : "r"(u32_smem_ptr));
#endif
}

__device__ __forceinline__ void ldsm_trans_4(int32_t& m0, int32_t& m1,
                                             int32_t& m2, int32_t& m3,
                                             const void* smemptr) {
  uint32_t u32_smem_ptr = get_u32_smem_ptr(smemptr);
#if (__CUDA_ARCH__ >= 750) && (CUDA_VERSION >= 11040) && !defined(__HGGCCC__)
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(m0), "=r"(m1), "=r"(m2), "=r"(m3)
      : "r"(u32_smem_ptr));
#endif
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

// -------------------------------------------------------------------
// A_ldg16, B_ldg8, ldsm, 256x128_64x64 tile
// k % 16 == 0, n % 2 == 0
// M_TILE = 256, N_TILE = 128, K_TILE = 64, blockDim = (256, 1, 1)
// -------------------------------------------------------------------
__global__
__launch_bounds__(256) void s8gemm_nn_256x128_mma8816_ldsm4_crosswise_smem_ldg16_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ C, int m, int n, int k) {
  // shared memory buffer, each line for a mma8816 fragment
  __shared__ int8_t __align__(16) A_smem[2 * 256 * 64];
  // 64x128 B_tile transposed to 32x256, means 4x16 x (8x16 fragment)
  __shared__ int8_t __align__(16) B_smem[2 * 32 * 256];

  // register fragment
  int32_t A_frag[2][8];
  int32_t B_frag[2][8];
  int2 C_frag[8][8];

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      C_frag[i][j] = make_int2(0, 0);
    }
  }

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // init matrix_A & matrix_B ldg pointer
  const int8_t* __restrict__ A_ldg_ptr =
      A + (blockIdx.y * 256 + warp_id * 4 * 8 + lane_id / 4) * k +
      (lane_id % 4) * 16;
  const int8_t* __restrict__ B_ldg_ptr = B + blockIdx.x * 128 +
                                         (threadIdx.x % 16) * 8 +
                                         (threadIdx.x / 16 * 2 * n);

  // init matrix_A & matrix_B sts pointer
  int8_t* A_sts_ptr = A_smem + (lane_id % 4) * (256 * 16) +
                      (lane_id % 4 + lane_id / 4) % 8 * 16 +
                      warp_id * (4 * 8 * 16);
  int8_t* B_sts_ptr = B_smem + (threadIdx.x / 16 * 256) +
                      ((threadIdx.x / 16 % 8) * 4 + threadIdx.x % 16) % 16 * 16;

  // init matrix_A & matrix_B ldsm pointer
  // 4x2 warps for mma
  const int ldsm_row = lane_id % 8;
  const int8_t* A_ldsm_ptr =
      A_smem + (warp_id / 2) * (8 * 8 * 16) + (lane_id / 8) * 8 * 16;
  const int8_t* B_ldsm_ptr_0 =
      B_smem + ldsm_row * 256 +
      (ldsm_row * 4 + warp_id % 2 * 8 + lane_id / 8) % 16 * 16;
  const int8_t* B_ldsm_ptr_1 =
      B_smem + ldsm_row * 256 +
      (ldsm_row * 4 + warp_id % 2 * 8 + lane_id / 8 + 4) % 16 * 16;

  // avoid matrix A ldg out of bound
  uint32_t A_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = blockIdx.y * 256 + warp_id * 4 * 8 + i * 8 + lane_id / 4;
    if (m_idx < m) A_ldg_guard |= 1u << i;
  }

  // avoid matrix B ldg out of bound
  bool B_ldg_guard = blockIdx.x * 128 + threadIdx.x % 16 * 8 < n;

  // ldg register buffer
  uint4 A_ldg_reg[4];
  for (int i = 0; i < 4; ++i) {
    A_ldg_reg[i] = make_uint4(0, 0, 0, 0);
  }

  uint4 B_ldg_reg[2];
  uint2 B_ldg_0[2];
  uint2 B_ldg_1[2];
  for (int i = 0; i < 2; ++i) {
    B_ldg_0[i] = make_uint2(0, 0);
    B_ldg_1[i] = make_uint2(0, 0);
  }

  const int k_loops = (k + 63) / 64 - 1;

  // load 1'st A_tile & B_tile
  {
    // load A_tile
    int A_remained = (k - k_loops * 64) / 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(uint4*)(A_sts_ptr + 8 * 16 * i) =
          (A_ldg_guard & (1u << i)) != 0 && lane_id % 4 < A_remained
              ? *(const uint4*)(A_ldg_ptr + 8 * k * i)
              : make_uint4(0, 0, 0, 0);
    }
    A_ldg_ptr += A_remained * 16;

    // load B_tile
    int B_remained = k - k_loops * 64;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg_8(B_ldg_0[i], B_ldg_ptr + i * 16 * 2 * n,
            (i * 16 + threadIdx.x / 16) * 2 < B_remained && B_ldg_guard);
      ldg_8(B_ldg_1[i], B_ldg_ptr + i * 16 * 2 * n + n,
            (i * 16 + threadIdx.x / 16) * 2 + 1 < B_remained && B_ldg_guard);
      ldg_trans(B_ldg_reg[i], B_ldg_0[i], B_ldg_1[i]);
      *(uint4*)(B_sts_ptr + i * 16 * 256) = B_ldg_reg[i];
    }
    B_ldg_ptr += B_remained * n;

    __syncthreads();
    A_sts_ptr += 256 * 64;
    B_sts_ptr += 32 * 256;
  }

  // load 1'st warp tile
  ldsm_4(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
         A_ldsm_ptr + ldsm_row * 16);
  ldsm_4(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
         A_ldsm_ptr + ldsm_row * 16 + 4 * 8 * 16);
  ldsm_trans_4(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
               B_ldsm_ptr_0);
  ldsm_trans_4(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
               B_ldsm_ptr_1);

  // shared memory double buffer flag, switch between 1 and -1
  int smem_switch = 1;

  // GEMM main loop over K
  for (int k_loop = 0; k_loop < k_loops; ++k_loop) {
#pragma unroll
    for (int k_frag = 0; k_frag < 4; ++k_frag) {
      // store A_ldg_reg & B_ldg_reg to smem
      if (k_frag == 3) {
// A_tile sts
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          *(uint4*)(A_sts_ptr + 8 * 16 * i) = A_ldg_reg[i];
        }

// B_tile sts
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          *(uint4*)(B_sts_ptr + i * 16 * 256) = B_ldg_reg[i];
        }

        __syncthreads();

        // switch double buffer
        A_sts_ptr += smem_switch * (-256 * 64);
        B_sts_ptr += smem_switch * (-32 * 256);
        A_ldsm_ptr += smem_switch * (256 * 64);
        B_ldsm_ptr_0 += smem_switch * (32 * 256);
        B_ldsm_ptr_1 += smem_switch * (32 * 256);

        smem_switch ^= 0xfffffffe;
      }

      // load warp tile from smem
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_ldsm_ptr + ((k_frag + 1) % 4) * 256 * 16 +
                 ((k_frag + 1) % 4 + ldsm_row) % 8 * 16);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_ldsm_ptr + ((k_frag + 1) % 4) * 256 * 16 +
                 ((k_frag + 1) % 4 + ldsm_row) % 8 * 16 + 4 * 8 * 16);

      ldsm_trans_4(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_ldsm_ptr_0 + ((k_frag + 1) % 4) * 8 * 256);
      ldsm_trans_4(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_ldsm_ptr_1 + ((k_frag + 1) % 4) * 8 * 256);

      // load A_tile & B_tile from gmem
      if (k_frag == 0) {
// load A_tile
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ldg_16(A_ldg_reg[i], A_ldg_ptr + 8 * k * i,
                 (A_ldg_guard & (1u << i)) != 0);
        }
        A_ldg_ptr += 64;

// load B_tile
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          ldg_8(B_ldg_0[i], B_ldg_ptr + i * 16 * 2 * n, B_ldg_guard);
          ldg_8(B_ldg_1[i], B_ldg_ptr + i * 16 * 2 * n + n, B_ldg_guard);
          ldg_trans(B_ldg_reg[i], B_ldg_0[i], B_ldg_1[i]);
        }
        B_ldg_ptr += 64 * n;
      }

// mma
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          mma8816(A_frag[k_frag % 2][i], B_frag[k_frag % 2][j], C_frag[i][j].x,
                  C_frag[i][j].y, C_frag[i][j].x, C_frag[i][j].y);
        }
      }
    }
  }

// GEMM loop for final tile
#pragma unroll
  for (int k_frag = 0; k_frag < 4; ++k_frag) {
    // load warp tile from smem
    if (k_frag < 3) {
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_ldsm_ptr + ((k_frag + 1) % 4) * 256 * 16 +
                 ((k_frag + 1) % 4 + ldsm_row) % 8 * 16);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_ldsm_ptr + ((k_frag + 1) % 4) * 256 * 16 +
                 ((k_frag + 1) % 4 + ldsm_row) % 8 * 16 + 4 * 8 * 16);

      ldsm_trans_4(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_ldsm_ptr_0 + ((k_frag + 1) % 4) * 8 * 256);
      ldsm_trans_4(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_ldsm_ptr_1 + ((k_frag + 1) % 4) * 8 * 256);
    }

// mma
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mma8816(A_frag[k_frag % 2][i], B_frag[k_frag % 2][j], C_frag[i][j].x,
                C_frag[i][j].y, C_frag[i][j].x, C_frag[i][j].y);
      }
    }
  }

  __syncthreads();

  // C_tile write back, reuse A_smem
  // 8 int32 matrix, 8x4 int2 for each, double buffered
  int2* C_smem = (int2*)(A_smem + warp_id * 2 * 8 * 8 * 4 * sizeof(int2));
  int mma_tid_x = lane_id % 4;
  int mma_tid_y = lane_id / 4;

// sts C_frag[0][0]~C_frag[0][7]
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(C_smem + mma_tid_y * (8 * 4) + ((mma_tid_y + i) % 8) * 4 + mma_tid_x) =
        C_frag[0][i];
  }

  // C_smem double buffer switch
  int C_smem_switch = 1;
  int2* C_lds_ptr = C_smem;
  int2* C_sts_ptr = C_smem + 8 * 8 * 4;

// write back loop
#pragma unroll
  for (int C_row = 0; C_row < 8; ++C_row) {
    __syncthreads();

    // stg
    int n_idx = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id * 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int m_idx = blockIdx.y * 256 + (warp_id / 2) * 64 + C_row * 8 + i;
      if (m_idx < m && n_idx < n) {
        *(int2*)(C + m_idx * n + n_idx) =
            *(C_lds_ptr + i * 8 * 4 + (i * 4 + lane_id) % (8 * 4));
      }
    }

    // sts
    if (C_row < 7) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        *(C_sts_ptr + mma_tid_y * (8 * 4) + ((mma_tid_y + i) % 8) * 4 +
          mma_tid_x) = C_frag[C_row + 1][i];
      }
    }

    C_sts_ptr += C_smem_switch * (-8 * 8 * 4);
    C_lds_ptr += C_smem_switch * (8 * 8 * 4);
    C_smem_switch ^= 0xfffffffe;
  }
}
void GemmHIEInt8(int32_t* matrix_C, const int8_t* matrix_A,
                 const int8_t* matrix_B, int M_, int N_, int K_,
                 cudaStream_t stream) {
  int grid_x = (N_ + 127) / 128;
  int grid_y = (M_ + 255) / 256;
  dim3 block(256);
  dim3 grid(grid_x, grid_y);
  s8gemm_nn_256x128_mma8816_ldsm4_crosswise_smem_ldg16_kernel<<<grid, block, 0,
                                                                stream>>>(
      matrix_A, matrix_B, matrix_C, M_, N_, K_);
}
// #endif
}  // namespace cuda
}  // namespace allspark
