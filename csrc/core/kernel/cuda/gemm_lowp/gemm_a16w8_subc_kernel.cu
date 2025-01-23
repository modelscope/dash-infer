/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_subc_kernel.cu
 */

#include <cuda/std/type_traits>

#include "gemm_a16w8_kernel.h"
#include "gemm_lowp_utils.cuh"

namespace allspark {
namespace cuda {

/*
 * matrix_A/B/C: row_major
 * k % 8 == 0, n % 8 == 0
 * accumulator precision: FP16 or FP32
 * output datatype: FP16
 *
 * m128n128k32 thread block tile
 * m64n64k32 warp tile
 */
template <typename QT, typename AccumulatorType>
__global__ void __launch_bounds__(128, 2)
    hgemm_A16W8_subc_128x128x32_hmma1688_ldg8_kernel(
        const half* A, const QT* B, const half* B_scale, const half* B_zero,
        half* C, uint32_t m, uint32_t n, uint32_t k, uint32_t kGroupSize,
        uint32_t kGroupCnt,
        uint32_t A_ldg_step,    // k * sizeof(half) * 32
        uint32_t B_ldg_step) {  // n * sizeof(int8_t) * 16
  static_assert(::cuda::std::is_same_v<AccumulatorType, half> ||
                    ::cuda::std::is_same_v<AccumulatorType, float>,
                "");
  constexpr int PACK = ::cuda::std::is_same_v<AccumulatorType, half> ? 2 : 1;
  /*
   * matrix A & B thread block tile shared memory (double buffer)
   * matrix A: 128 * 32 * 2Byte/item * double buffer = 8KB * 2
   * matrix B: 128 * 32 * 2Byte/item * double buffer = 8KB * 2
   */
  __shared__ __align__(1024 * 16) char smem[1024 * 32];
  half* A_smem = reinterpret_cast<half*>(smem);
  half* B_smem = reinterpret_cast<half*>(smem + 1024 * 8);

  // A, B and C register fragment
  uint32_t A_frag[2][8];
  uint32_t B_frag[2][8];
  uint32_t C_frag[8][16 / PACK];

  if (::cuda::std::is_same_v<AccumulatorType, half>) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        C_frag[i][j] = 0;
      }
    }
  } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        C_frag[i][j] = 0;
      }
    }
  }

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;

  /*
   * A_tile & B_tile ldg pointer.
   * A_tile: 32x4 thread, 4 LDG each thread (16 register, 4 predicate)
   * B_tile: 16x8 thread, 2x2 LDG each thread (16 register, 2 predicate)
   */
  const char* A_ldg_ptr = reinterpret_cast<const char*>(
      A + (blockIdx.x * 128 + threadIdx.x / 4) * k + (threadIdx.x % 4) * 8);
  const int B_n_idx = blockIdx.y * 128 + (threadIdx.x % 8) * 8;
  const char* B_ldg_ptr =
      reinterpret_cast<const char*>(B + (threadIdx.x / 8) * n + B_n_idx);

  /*
   * A_tile & B_tile sts addr.
   * using uint32_t pointer for faster double buffer switch
   */
  uint32_t A_sts_addr =
      smem_u32addr(A_smem + (threadIdx.x % 4) * (128 * 8) +
                   ((threadIdx.x / 4) ^ (threadIdx.x % 4 * 2)) * 8);
  uint32_t B_sts_addr =
      smem_u32addr(B_smem + (threadIdx.x / 8) * 128 +
                   ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // A_tile lds addr, 1 addr for 1 k_frag
  uint32_t A_lds_addr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    A_lds_addr[i] = smem_u32addr(A_smem + i * 128 * 8 + (warp_id / 2) * 64 * 8 +
                                 (lane_id ^ (i * 2)) * 8);
  }

  // B_tile lds addr
  uint32_t B_lds_addr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    B_lds_addr[i] =
        smem_u32addr(B_smem + (lane_id % 8) * 128 + (warp_id % 2) * 64 +
                     ((lane_id / 8 + i * 4) ^ (lane_id % 8)) * 8);
  }

  // ldg_guard to avoid LDG out of bound
  bool A_ldg_guard[4]{};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = blockIdx.x * 128 + threadIdx.x / 4 + i * 32;
    if (m_idx < m) {
      A_ldg_guard[i] = true;
    }
  }

  bool B_ldg_guard[2]{};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int n_idx = blockIdx.y * 128 + (threadIdx.x % 8) * 8 + i * 64;
    if (n_idx < n) {
      B_ldg_guard[i] = true;
    }
  }

  uint32_t A_ldg_reg[4][4];
  QT B_ldg_reg_q[4][8];
  __half2 B_scale_reg[2][4];
  __half2 B_zero_reg[2][4];
  __half2 B_ldg_reg[4][4];

  // 1'st A&B tile loaded before the k_tile loop
  uint32_t k_tiles = (k + 31) / 32 - 1;

  int B_k_idx = 0;
  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - k_tiles * 32;

    const char* first_A_ldg_ptr = A_ldg_ptr + (k - first_k_tile) * sizeof(half);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bool guard = A_ldg_guard[i] && (threadIdx.x % 4) * 8 < first_k_tile;
      ldg128_cg_0(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                  A_ldg_reg[i][3], first_A_ldg_ptr + i * A_ldg_step, guard);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts128(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2], A_ldg_reg[i][3],
             A_sts_addr + i * 32 * 8 * sizeof(half));
    }

    // load B, B_scale, B_zero
    const char* first_B_ldg_ptr =
        B_ldg_ptr + (k - first_k_tile) * n * sizeof(QT);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        bool guard = B_ldg_guard[j] && threadIdx.x / 8 + i * 16 < first_k_tile;
        ldg64_nc_0(
            *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j]),
            *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j] + 4),
            first_B_ldg_ptr + i * 16 * n * sizeof(QT) + j * 64 * sizeof(QT),
            guard);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg128_ca_0(B_scale_reg[i][0], B_scale_reg[i][1], B_scale_reg[i][2],
                  B_scale_reg[i][3],
                  B_scale + (kGroupCnt - 1) * n + B_n_idx + i * 64,
                  B_ldg_guard[i]);
      ldg128_ca_0(B_zero_reg[i][0], B_zero_reg[i][1], B_zero_reg[i][2],
                  B_zero_reg[i][3],
                  B_zero + (kGroupCnt - 1) * n + B_n_idx + i * 64,
                  B_ldg_guard[i]);
    }

// dequant B from int8 -> fp16
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          B_ldg_reg[i * 2 + j][k].x =
              __hmul(__hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k]),
                            B_zero_reg[j][k].x),
                     B_scale_reg[j][k].x);
          B_ldg_reg[i * 2 + j][k].y = __hmul(
              __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k + 1]),
                     B_zero_reg[j][k].y),
              B_scale_reg[j][k].y);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        sts128(
            B_ldg_reg[i * 2 + j][0], B_ldg_reg[i * 2 + j][1],
            B_ldg_reg[i * 2 + j][2], B_ldg_reg[i * 2 + j][3],
            B_sts_addr + i * 16 * 128 * sizeof(half) + j * 64 * sizeof(half));
      }
    }

    __syncthreads();
  }

  // smem double buffer offset
  uint32_t lds_offset = 0;
  uint32_t sts_offset = 16 * 1024;

  // load 1'st fragment
  ldsm_4(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
         A_lds_addr[0] + lds_offset);
  ldsm_4(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
         A_lds_addr[0] + lds_offset + 32 * 8 * sizeof(half));
  ldsm_4_trans(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
               B_lds_addr[0] + lds_offset);
  ldsm_4_trans(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
               B_lds_addr[1] + lds_offset);

  // k_tiles loop
  for (; k_tiles > 0; --k_tiles) {
#pragma unroll
    for (int k_frag = 0; k_frag < 4; ++k_frag) {
      // store next A&B tile to shared memory
      if (k_frag == 3) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sts128(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                 A_ldg_reg[i][3],
                 A_sts_addr + sts_offset + i * 32 * 8 * sizeof(half));
        }

// dequant B from int8 -> fp16
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              B_ldg_reg[i * 2 + j][k].x = __hmul(
                  __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k]),
                         B_zero_reg[j][k].x),
                  B_scale_reg[j][k].x);
              B_ldg_reg[i * 2 + j][k].y = __hmul(
                  __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k + 1]),
                         B_zero_reg[j][k].y),
                  B_scale_reg[j][k].y);
            }
          }
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            sts128(B_ldg_reg[i * 2 + j][0], B_ldg_reg[i * 2 + j][1],
                   B_ldg_reg[i * 2 + j][2], B_ldg_reg[i * 2 + j][3],
                   B_sts_addr + sts_offset + i * 16 * 128 * sizeof(half) +
                       j * 64 * sizeof(half));
          }
        }

        __syncthreads();

        // switch double buffer
        lds_offset ^= 16 * 1024;
        sts_offset ^= 16 * 1024;
      }

      // load next A&B fragment from shared memory to register
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset + 32 * 8 * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr[0] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr[1] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));

      // load next A&B tile
      if (k_frag == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ldg128_cache_hint(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                            A_ldg_reg[i][3], A_ldg_ptr + i * A_ldg_step,
                            A_ldg_guard[i], LoadCacheOperator::CG);
        }

        A_ldg_ptr += 32 * sizeof(half);

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            ldg64_nc(*reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j]),
                     *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j] + 4),
                     B_ldg_ptr + j * 64 * sizeof(QT), B_ldg_guard[j]);
          }
          B_ldg_ptr += B_ldg_step;
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
          ldg128_cache_hint(
              B_scale_reg[i][0], B_scale_reg[i][1], B_scale_reg[i][2],
              B_scale_reg[i][3],
              B_scale + (B_k_idx >> kGroupSize) * n + B_n_idx + i * 64,
              B_ldg_guard[i], LoadCacheOperator::CA);
          ldg128_cache_hint(
              B_zero_reg[i][0], B_zero_reg[i][1], B_zero_reg[i][2],
              B_zero_reg[i][3],
              B_zero + (B_k_idx >> kGroupSize) * n + B_n_idx + i * 64,
              B_ldg_guard[i], LoadCacheOperator::CA);
        }
        B_k_idx += 32;
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (::cuda::std::is_same_v<AccumulatorType, half>) {
            hmma1688_f16(C_frag[i * 2][j], C_frag[i * 2 + 1][j],
                         A_frag[k_frag % 2][i * 2],
                         A_frag[k_frag % 2][i * 2 + 1], B_frag[k_frag % 2][j]);
          } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
            hmma1688_f32(C_frag[i * 2][j * 2], C_frag[i * 2][j * 2 + 1],
                         C_frag[i * 2 + 1][j * 2], C_frag[i * 2 + 1][j * 2 + 1],
                         A_frag[k_frag % 2][i * 2],
                         A_frag[k_frag % 2][i * 2 + 1], B_frag[k_frag % 2][j]);
          }
        }
      }
    }
  }

// FFMA for the last tile
#pragma unroll
  for (int k_frag = 0; k_frag < 4; ++k_frag) {
    if (k_frag < 3) {
      // load next A&B fragment from shared memory to register
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset + 32 * 8 * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr[0] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr[1] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
    }

// HMMA loop
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (::cuda::std::is_same_v<AccumulatorType, half>) {
          hmma1688_f16(C_frag[i * 2][j], C_frag[i * 2 + 1][j],
                       A_frag[k_frag % 2][i * 2], A_frag[k_frag % 2][i * 2 + 1],
                       B_frag[k_frag % 2][j]);
        } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
          hmma1688_f32(C_frag[i * 2][j * 2], C_frag[i * 2][j * 2 + 1],
                       C_frag[i * 2 + 1][j * 2], C_frag[i * 2 + 1][j * 2 + 1],
                       A_frag[k_frag % 2][i * 2], A_frag[k_frag % 2][i * 2 + 1],
                       B_frag[k_frag % 2][j]);
        }
      }
    }
  }

  /*
   * C_tile write back, reuse A&B tile shared memory buffer
   * 32x64 (32x72 with padding) FP16 C_tile_writeback
   * padding 8 items each row to avoid bank conflict.
   *     =>32x32 (32x36 with padding) uint32_t writeback tile
   *
   * 4x8 thread STG.128
   */
  uint32_t C_sts_size = 32 * 72 * sizeof(half);
  uint32_t mma_tid_x = lane_id % 4;
  uint32_t mma_tid_y = lane_id / 4;
  uint32_t* C_sts_ptr =
      reinterpret_cast<uint32_t*>(smem + warp_id * C_sts_size) +
      mma_tid_y * 36 + mma_tid_x;
  uint4* C_lds_ptr = reinterpret_cast<uint4*>(smem + warp_id * C_sts_size) +
                     lane_id / 8 * 9 + lane_id % 8;

  uint32_t m_idx = blockIdx.x * 128 + warp_id / 2 * 64 + lane_id / 8;
  uint32_t n_idx = blockIdx.y * 128 + warp_id % 2 * 64 + lane_id % 8 * 8;

  half* C_stg_ptr = C + m_idx * n + n_idx;

  bool n_guard = n_idx < n;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    __syncthreads();

// C_tile sts
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        if (::cuda::std::is_same_v<AccumulatorType, half>) {
          C_sts_ptr[j * 8 * 36 + p * 4] = C_frag[i * 4 + j][p];
        } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
          half low16 = static_cast<half>(
              reinterpret_cast<float&>(C_frag[i * 4 + j][p * 2]));
          half high16 = static_cast<half>(
              reinterpret_cast<float&>(C_frag[i * 4 + j][p * 2 + 1]));
          C_sts_ptr[j * 8 * 36 + p * 4] =
              reinterpret_cast<uint32_t&>(low16) & 0xffff;
          C_sts_ptr[j * 8 * 36 + p * 4] |= reinterpret_cast<uint32_t&>(high16)
                                           << 16;
        }
      }
    }

    __syncthreads();

// C_tile stg
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint4 stg_reg = C_lds_ptr[j * 9 * 4];
      stg128(stg_reg.x, stg_reg.y, stg_reg.z, stg_reg.w,
             C_stg_ptr + (i * 32 + j * 4) * n,
             m_idx + i * 32 + j * 4 < m && n_guard);
    }
  }
}

uint32_t log2(uint32_t x) {
  uint32_t ret = 0;
  while (x != 1) {
    x /= 2;
    ret++;
  }
  return ret;
}

template <typename QT, template <class> class ActiveFunc>
void hgemm_A16W8_subc_128x128x32_hmma1688_ldg8(
    const half* A, const QT* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const float alpha, cudaStream_t stream) {
  {
    uint32_t grid_x = (M + 127) / 128;
    uint32_t grid_y = (N + 127) / 128;
    dim3 grid(grid_x, grid_y);
    uint32_t A_ldg_step = K * sizeof(half) * 32;
    uint32_t B_ldg_step = N * sizeof(QT) * 16;
    uint32_t GroupCnt = (K + GroupSize - 1) / GroupSize;

    hgemm_A16W8_subc_128x128x32_hmma1688_ldg8_kernel<QT, float>
        <<<grid, 128, 0, stream>>>(A, B, B_scale, B_zero, C, M, N, K,
                                   log2(GroupSize), GroupCnt, A_ldg_step,
                                   B_ldg_step);
  }

  { add_bias<half, ActiveFunc>(bias, C, M, N, alpha, stream); }
}

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 *
 */
template <uint32_t BLOCK_SIZE, uint32_t Bldg_pack_size, typename QType>
struct GmemTile_A16W8_32x128x16 {
  // element num loaded by per thread from Global Memory
  static constexpr int THREAD_ELEM_CNT_A = 32 * 16 / BLOCK_SIZE;   // 4
  static constexpr int THREAD_ELEM_CNT_B = 16 * 128 / BLOCK_SIZE;  // 16
  const int WARP_CNT = BLOCK_SIZE / 32;                            // 4

  const int ROW_OFFSET_OF_A_SMEM = 32 * 2 + 4;
  const int ROW_OFFSET_OF_B_SMEM = 128;

  __device__ GmemTile_A16W8_32x128x16(
      const GEMM_A16W8_Params<half, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    this_block_A_base_ptr =
        params.A_ptr + blockIdx.x * 32 * params.K + blockIdx.z * params.SplitK;
    this_block_B_base_ptr =
        params.B_ptr + blockIdx.y * 128 + blockIdx.z * params.SplitK * params.N;

    // per thread load one element of matrix A in each iter, total 4 iter
    const uint32_t load_a_row_base_idx = threadIdx.x / 16 * 2;
    load_a_col_idx = threadIdx.x % 16;
    load_a_base_offset = load_a_row_base_idx * params.K + load_a_col_idx;

    // per thread load Bldg_pack_size element of matrix B in each iter
    const uint32_t load_b_col_idx = (threadIdx.x * Bldg_pack_size) % 128;
    load_b_row_base_idx = (threadIdx.x * Bldg_pack_size) / 128;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_base_offset = (load_a_col_idx / 2) * ROW_OFFSET_OF_A_SMEM +
                          (load_a_col_idx % 2) * 32 + load_a_row_base_idx;
    store_b_base_offset =
        load_b_row_base_idx * ROW_OFFSET_OF_B_SMEM + load_b_col_idx;

#pragma unroll
    for (int i = 0; i < THREAD_ELEM_CNT_A; ++i) {
      A_ldg_guard[i] = false;
      int m_idx = blockIdx.x * 32 + load_a_row_base_idx + (i / 2) * 16 + i % 2;
      if (m_idx < params.M) {
        A_ldg_guard[i] = true;
      }
    }

    B_ldg_guard = false;
    n_idx = blockIdx.y * 128 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

    B_k_idx = blockIdx.z * params.SplitK;

// initialize a_regs all 0
#pragma unroll
    for (int i = 0; i < THREAD_ELEM_CNT_A; ++i) {
      a_regs[i] = half(0);
    }
  }

  __device__ void ldg_first_k_tile(const uint32_t first_k_tile,
                                   const uint32_t tb_k_slice) {
    const __half* this_A_ptr =
        this_block_A_base_ptr + tb_k_slice - first_k_tile;
#pragma unroll
    for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_A; load_iter++) {
      if (A_ldg_guard[load_iter] && load_a_col_idx < first_k_tile) {
        a_regs[load_iter] =
            *(this_A_ptr + load_a_base_offset +
              (load_iter / 2) * 16 * params.K + (load_iter % 2) * params.K);
      }
    }

    // load B scale and zero-point
    uint32_t k_idx = B_k_idx + tb_k_slice - first_k_tile;
    uint32_t scale_offset = (k_idx / params.GroupSize) * params.N + n_idx;
    const half* this_B_scale_ptr = params.B_scale_ptr + scale_offset;
    const half* this_B_zero_ptr = params.B_zero_ptr + scale_offset;
    load_gdata_cache_hint<half, Bldg_pack_size>(
        b_regs_scale, this_B_scale_ptr, B_ldg_guard, LoadCacheOperator::CA);
    load_gdata_cache_hint<half, Bldg_pack_size>(
        b_regs_zero, this_B_zero_ptr, B_ldg_guard, LoadCacheOperator::CA);

    // load B
    const QType* this_B_ptr =
        this_block_B_base_ptr + (tb_k_slice - first_k_tile) * params.N;
    const uint32_t b_rows_one_iter = (BLOCK_SIZE * Bldg_pack_size) / 128;
#pragma unroll
    for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_B / Bldg_pack_size;
         ++load_iter) {
      int load_b_offset =
          load_b_base_offset + load_iter * b_rows_one_iter * params.N;
      if ((load_b_row_base_idx + load_iter * b_rows_one_iter) < first_k_tile) {
        load_gdata_cache_hint<QType, Bldg_pack_size>(
            b_regs_q + load_iter * Bldg_pack_size, this_B_ptr + load_b_offset,
            B_ldg_guard, LoadCacheOperator::CS);
      }
    }
  }

  __device__ void load_from_gmem() {
#pragma unroll
    for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_A; load_iter++) {
      if (A_ldg_guard[load_iter]) {
        a_regs[load_iter] =
            *(this_block_A_base_ptr + load_a_base_offset +
              (load_iter / 2) * 16 * params.K + (load_iter % 2) * params.K);
      }
    }

    // load B scale and zero-point
    uint32_t scale_offset = (B_k_idx / params.GroupSize) * params.N + n_idx;
    const half* this_B_scale_ptr = params.B_scale_ptr + scale_offset;
    const half* this_B_zero_ptr = params.B_zero_ptr + scale_offset;
    load_gdata_cache_hint<half, Bldg_pack_size>(
        b_regs_scale, this_B_scale_ptr, B_ldg_guard, LoadCacheOperator::CA);
    load_gdata_cache_hint<half, Bldg_pack_size>(
        b_regs_zero, this_B_zero_ptr, B_ldg_guard, LoadCacheOperator::CA);

    // load B
    const uint32_t b_rows_one_iter = (BLOCK_SIZE * Bldg_pack_size) / 128;
#pragma unroll
    for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_B / Bldg_pack_size;
         ++load_iter) {
      int load_b_offset =
          load_b_base_offset + load_iter * b_rows_one_iter * params.N;
      load_gdata_cache_hint<QType, Bldg_pack_size>(
          b_regs_q + load_iter * Bldg_pack_size,
          this_block_B_base_ptr + load_b_offset, B_ldg_guard,
          LoadCacheOperator::CS);
    }

    // switch to next BLOCK_TILE_K
    this_block_A_base_ptr += 16;
    this_block_B_base_ptr += 16 * params.N;

    B_k_idx += 16;
  }

  __device__ void store_to_smem(const int buf_idx) {
    uint32_t A_smem_addr = A_smem_base_addr + A_smem_switch_offset * buf_idx;
    uint32_t B_smem_addr = B_smem_base_addr + B_smem_switch_offset * buf_idx;

#pragma unroll
    for (int store_iter = 0; store_iter < THREAD_ELEM_CNT_A / 2; store_iter++) {
      sts32(reinterpret_cast<uint32_t&>(a_regs[store_iter * 2]),
            A_smem_addr +
                (store_a_base_offset + store_iter * 16) * 2 /* ELEM SIZE */);
    }

    const uint32_t b_rows_one_iter = (BLOCK_SIZE * Bldg_pack_size) / 128;
#pragma unroll
    for (int store_iter = 0; store_iter < THREAD_ELEM_CNT_B / Bldg_pack_size;
         store_iter++) {
#pragma unroll
      for (int i = 0; i < Bldg_pack_size; ++i) {
        b_regs[store_iter * Bldg_pack_size + i] = __hmul(
            __hsub(static_cast<half>(b_regs_q[store_iter * Bldg_pack_size + i]),
                   b_regs_zero[i]),
            b_regs_scale[i]);
      }
      store_sdata<half, Bldg_pack_size>(
          b_regs + store_iter * Bldg_pack_size,
          B_smem_addr + (store_b_base_offset +
                         store_iter * b_rows_one_iter * ROW_OFFSET_OF_B_SMEM) *
                            2 /* ELEM SIZE */);
    }
  }

  const half* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;
  uint32_t load_a_base_offset;
  uint32_t load_b_base_offset;

  uint32_t load_a_col_idx;
  uint32_t load_b_row_base_idx;

  uint32_t store_a_base_offset;
  uint32_t store_b_base_offset;

  bool A_ldg_guard[4]{};
  bool B_ldg_guard{};

  half a_regs[THREAD_ELEM_CNT_A]{};
  QType b_regs_q[THREAD_ELEM_CNT_B]{};
  half b_regs_scale[Bldg_pack_size]{};
  half b_regs_zero[Bldg_pack_size]{};
  half b_regs[THREAD_ELEM_CNT_B]{};

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;

  uint32_t n_idx;
  uint32_t B_k_idx;
  const GEMM_A16W8_Params<half, QType>& params;
};

template <uint32_t Bstg_pack_size, typename QType>
struct ComputeTile_f16_32x128x16 {
  const int WARP_SIZE = 32;
  const int ROW_OFFSET_OF_A_SMEM = 32 * 2 + 4;
  const int ROW_OFFSET_OF_B_SMEM = 128;

  __device__ ComputeTile_f16_32x128x16(
      const GEMM_A16W8_Params<half, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    load_a_base_offset = lane_id % 8 * 4;
    load_b_base_offset = warp_id * 32 + lane_id / 8 * 8;
    store_c_row_base_idx = lane_id % 8 * 4;
    store_c_col_idx = warp_id * 32 + lane_id / 8 * 8;
    store_c_base_offset = store_c_row_base_idx * params.N + store_c_col_idx;

    n_idx = blockIdx.y * 128 + store_c_col_idx;
    m_idx = blockIdx.x * 32 + store_c_row_base_idx;

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        C_frag[i][j] = 0;
      }
    }

    this_block_C_base_ptr = params.C_split_ptr +
                            blockIdx.z * params.M * params.N +
                            blockIdx.x * 32 * params.N + blockIdx.y * 128;
  }
  __device__ void load_from_smem(const int smem_buf_idx, const int reg_idx,
                                 const int k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_switch_offset * smem_buf_idx;
    uint32_t B_smem_addr =
        B_smem_base_addr + B_smem_switch_offset * smem_buf_idx;

    const int load_a_offset = load_a_base_offset +
                              (k_phase_idx / 2) * ROW_OFFSET_OF_A_SMEM +
                              (k_phase_idx % 2) * 32;
    const int load_b_offset =
        load_b_base_offset + ROW_OFFSET_OF_B_SMEM * k_phase_idx;

    lds64(*reinterpret_cast<uint32_t*>(A_frag[reg_idx]),
          *reinterpret_cast<uint32_t*>(A_frag[reg_idx] + 2),
          A_smem_addr + load_a_offset * 2 /* ELEM SIZE */);
    lds128(*reinterpret_cast<uint32_t*>(B_frag[reg_idx]),
           *reinterpret_cast<uint32_t*>(B_frag[reg_idx] + 2),
           *reinterpret_cast<uint32_t*>(B_frag[reg_idx] + 4),
           *reinterpret_cast<uint32_t*>(B_frag[reg_idx] + 6),
           B_smem_addr + load_b_offset * 2 /* ELEM SIZE */);
  }

  __device__ void mma(const int reg_idx) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __half2 a = __half2half2(A_frag[reg_idx][i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 b =
            __halves2half2(B_frag[reg_idx][j * 2], B_frag[reg_idx][j * 2 + 1]);
        auto& acc_now = *reinterpret_cast<half2*>(C_frag[i] + j * 2);
        acc_now = __hfma2(a, b, acc_now);
      }
    }
  }

  __device__ void writeback_to_gmem() {
#pragma unroll
    for (int store_iter_m = 0; store_iter_m < 4; store_iter_m++) {
#pragma unroll
      for (int store_iter_n = 0; store_iter_n < 8 / Bstg_pack_size;
           ++store_iter_n) {
        bool guard = (n_idx + store_iter_n * Bstg_pack_size) < params.N &&
                     (m_idx + store_iter_m) < params.M;
        if (guard) {
          half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                        store_iter_m * params.N + store_iter_n * Bstg_pack_size;
          store_gdata<half, Bstg_pack_size>(
              C_frag[store_iter_m] + store_iter_n * Bstg_pack_size, C_ptr);
        }
      }
    }
  }

  const GEMM_A16W8_Params<half, QType>& params;

  uint32_t load_a_base_offset;
  uint32_t load_b_base_offset;
  uint32_t store_c_base_offset;

  uint32_t store_c_row_base_idx;
  uint32_t store_c_col_idx;

  half* this_block_C_base_ptr = nullptr;

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;

  half A_frag[2][4];
  half B_frag[2][8];
  half C_frag[4][8];

  uint32_t n_idx;
  uint32_t m_idx;
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: K x N, matrix C: M x N
 *  BLOCK_TILE: m32n128k16
 *  WARP_TILE: m32n32k16
 *  THREAD_TILE: m4n8k16
 *
 */
template <uint32_t Bldg_pack_size, typename QType>
__global__
__launch_bounds__(128) void gemm_A16W8_subc_32x128x16_nn_Aldg1_splitk_nonfused_kernel(
    const GEMM_A16W8_Params<half, QType> params) {
  // A smem size = (32 * 2 + 4/* padding */) * 16 / 2 * 2B/elem * 2(double
  // buffer) = 2.125KB B smem size = 16 * 128 * 2B/elem * 2(double biffer) =
  // 16KB
  static constexpr int SMEM_SIZE =
      (32 * 2 + 4) * (16 / 2) * 2 * 2 + 16 * 128 * 2 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + (32 * 2 + 4) * (16 / 2) * 2 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);
  uint32_t A_smem_switch_offset = (32 * 2 + 4) * (16 / 2) * 2;
  uint32_t B_smem_switch_offset = 16 * 128 * 2;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_A16W8_32x128x16<128, Bldg_pack_size, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  uint32_t tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                            ? params.SplitK
                            : params.K - blockIdx.z * params.SplitK;
  uint32_t k_main_loop = (tb_k_slice + 15) / 16 - 1;
  uint32_t first_k_tile = tb_k_slice - k_main_loop * 16;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_f16_32x128x16<Bldg_pack_size, QType> compute_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);
  gmem_tile.store_to_smem(write_smem_buf_idx);

  __syncthreads();
  compute_tile.load_from_smem(read_smem_buf_idx, 0, 0);
  int reg_buf_idx = 1;

#pragma unroll 1
  for (int k_tile_idx = 0; k_tile_idx < k_main_loop; k_tile_idx++) {
    write_smem_buf_idx ^= 1;
    gmem_tile.load_from_gmem();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 16; k_phase_idx++) {
      if (k_phase_idx == 12) {
        gmem_tile.store_to_smem(write_smem_buf_idx);
      }

      if (k_phase_idx == 15) {
        read_smem_buf_idx ^= 1;
        __syncthreads();
      }

      compute_tile.load_from_smem(read_smem_buf_idx, reg_buf_idx,
                                  (k_phase_idx + 1) % 16);
      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

// compute for the last tile
#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 16; k_phase_idx++) {
    if (k_phase_idx < 15) {
      compute_tile.load_from_smem(read_smem_buf_idx, reg_buf_idx,
                                  k_phase_idx + 1);
    }
    compute_tile.mma(reg_buf_idx ^ 1);
    reg_buf_idx ^= 1;
  }

  compute_tile.writeback_to_gmem();
}

template <typename QType, template <class> class ActiveFunc>
void hgemm_A16W8_subc_32x128x16_simt_Aldg1(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const int sm_count, const SplitKParams splitk_params, const float alpha,
    cudaStream_t stream) {
  half* C_split = static_cast<half*>(workspace);

  int grid_x = (M + 31) / 32;
  int grid_y = (N + 127) / 128;
  int grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  dim3 block(128);
  dim3 grid(grid_x, grid_y, grid_z);
  GEMM_A16W8_Params<half, QType> params{A,
                                        B,
                                        B_scale,
                                        B_zero,
                                        C,
                                        C_split,
                                        M,
                                        N,
                                        K,
                                        GroupSize,
                                        (uint32_t)splitk_params.SplitK};

  uint32_t Bldg_pack_size = 8;
  while (N % Bldg_pack_size != 0) {
    Bldg_pack_size /= 2;
  }

  switch (Bldg_pack_size) {
    case 8:
      gemm_A16W8_subc_32x128x16_nn_Aldg1_splitk_nonfused_kernel<8, QType>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 4:
      gemm_A16W8_subc_32x128x16_nn_Aldg1_splitk_nonfused_kernel<4, QType>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 2:
      gemm_A16W8_subc_32x128x16_nn_Aldg1_splitk_nonfused_kernel<2, QType>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 1:
      gemm_A16W8_subc_32x128x16_nn_Aldg1_splitk_nonfused_kernel<1, QType>
          <<<grid, block, 0, stream>>>(params);
      break;
    default:
      LOG(ERROR) << "Error B_ldg pack size!";
      break;
  }

  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                           grid_z, alpha, stream);
}

// [NumSplitK, M, N]
template <typename FT, template <class> class ActiveFunc>
__global__ void reduce_sum(const FT* data_in, const FT* bias, FT* data_out,
                           const int M, const int N, const int K,
                           const int NumSplitK, const float alpha) {
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
    // C = alpha * A * B
    sum = static_cast<FT>(float(sum) * alpha);
    sum += bias_val;
    data_out_ptr[tid] = ActiveFunc<FT>::Op(sum);
  }
}

template <typename FT, typename QT, uint32_t UNROLL_N, uint32_t WARP_NUM,
          uint32_t MinGroupSize>
__global__ void gemv_a16w8_subc_splitk_m1_kernel(
    const FT* data_in, const QT* weight, const FT* scales, const FT* zeros,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t GroupSize, const uint32_t SplitK, FT* data_out) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int n_start = (blockIdx.x * 32 + lane_id) * UNROLL_N;
  const int k_start = blockIdx.y * SplitK;
  if (n_start >= N) return;

  float reg_c_f32[UNROLL_N];
#pragma unroll
  for (uint32_t i = 0; i < UNROLL_N; ++i) {
    reg_c_f32[i] = float(0);
  }

  const uint32_t depth = min(K - k_start, SplitK);
  const uint32_t k_main_loop = DivCeil(depth, MinGroupSize) - 1;

  {
    const uint32_t k_first = k_start + k_main_loop * MinGroupSize;
    const uint32_t group_idx = (k_first / GroupSize) * N + n_start;
    FT reg_s[UNROLL_N];
    FT reg_z[UNROLL_N];
    load_gdata<FT, UNROLL_N>(reg_s, scales + group_idx);
    load_gdata<FT, UNROLL_N>(reg_z, zeros + group_idx);

    FT reg_b[UNROLL_N];
    QT ldb[UNROLL_N];
    const FT* data_in_ptr = data_in + k_first;
    const QT* weight_ptr = weight + k_first * N + n_start;
    const uint32_t first_depth = depth - k_main_loop * MinGroupSize;
    for (int k_phase_idx = warp_id; k_phase_idx < first_depth;
         k_phase_idx += WARP_NUM) {
      FT reg_a = data_in_ptr[k_phase_idx];
      load_gdata<QT, UNROLL_N>(ldb, weight_ptr + k_phase_idx * N);
#pragma unroll
      for (int i = 0; i < UNROLL_N; ++i) {
        reg_b[i] = (FT(ldb[i]) - reg_z[i]) * reg_s[i];
      }
#pragma unroll
      for (int i = 0; i < UNROLL_N; ++i) {
        reg_c_f32[i] += static_cast<float>(reg_a * reg_b[i]);
      }
    }
  }

  FT reg_s[UNROLL_N];
  FT reg_z[UNROLL_N];
  for (int k_loop = 0; k_loop < k_main_loop; ++k_loop) {
    const uint32_t k_idx = k_start + k_loop * MinGroupSize;
    const uint32_t group_idx = (k_idx / GroupSize) * N + n_start;
    load_gdata<FT, UNROLL_N>(reg_s, scales + group_idx);
    load_gdata<FT, UNROLL_N>(reg_z, zeros + group_idx);

    FT reg_a;
    FT reg_b[UNROLL_N];
    QT ldb[UNROLL_N];
    const FT* data_in_ptr = data_in + k_idx;
    const QT* weight_ptr = weight + k_idx * N + n_start;
#pragma unroll
    for (int k_phase_idx = warp_id; k_phase_idx < MinGroupSize;
         k_phase_idx += WARP_NUM) {
      reg_a = data_in_ptr[k_phase_idx];
      load_gdata<QT, UNROLL_N>(ldb, weight_ptr + k_phase_idx * N);
#pragma unroll
      for (int i = 0; i < UNROLL_N; ++i) {
        reg_b[i] = (FT(ldb[i]) - reg_z[i]) * reg_s[i];
      }
#pragma unroll
      for (int i = 0; i < UNROLL_N; ++i) {
        reg_c_f32[i] += static_cast<float>(reg_a * reg_b[i]);
      }
    }
  }
  FT reg_c[UNROLL_N];
#pragma unroll
  for (int i = 0; i < UNROLL_N; ++i) {
    reg_c[i] = FT(reg_c_f32[i]);
  }

  if (WARP_NUM > 1) {
    __shared__ FT smem_reduce[WARP_NUM][UNROLL_N][32 * 2];
#pragma unroll
    for (int i = 0; i < UNROLL_N; ++i) {
      smem_reduce[warp_id][i][lane_id * 2] = reg_c[i];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
      for (int i = 0; i < UNROLL_N; ++i) {
        reg_c[i] = FT(0);
      }
#pragma unroll
      for (int wi = 0; wi < WARP_NUM; ++wi) {
#pragma unroll
        for (int i = 0; i < UNROLL_N; ++i) {
          reg_c[i] += smem_reduce[wi][i][lane_id * 2];
        }
      }
    }
  }

  if (warp_id == 0) {
    store_gdata<FT, UNROLL_N>(reg_c, data_out + blockIdx.y * M * N + n_start);
  }
}

template <typename FT, typename QT, template <class> class ActiveFunc>
void gemv_a16w8_subc_splitk(const FT* lhs, const QT* rhs, const FT* scales,
                            const FT* zeros, const FT* bias, FT* data_out,
                            const uint32_t M, const uint32_t N,
                            const uint32_t K, const uint32_t GroupSize,
                            void* workspace, const float alpha,
                            cudaStream_t stream) {
  const uint32_t SplitK = 512;
  {
    const uint32_t MinGroupSize = 64;
    const uint32_t UNROLL_N = 8;
    const uint32_t WARP_NUM = 8;
    const uint32_t THREADS_PER_BLOCK = 32 * WARP_NUM;
    const dim3 BLOCK_DIM =
        dim3(DivCeil(N, 32 * UNROLL_N), DivCeil(K, SplitK), 1);
    gemv_a16w8_subc_splitk_m1_kernel<FT, QT, UNROLL_N, WARP_NUM, MinGroupSize>
        <<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
            lhs, rhs, scales, zeros, M, N, K, GroupSize, SplitK,
            static_cast<FT*>(workspace));
  }
  {
    const uint32_t THREADS_PER_BLOCK = 128;
    const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), M, 1);
    reduce_sum<FT, ActiveFunc><<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<const FT*>(workspace), bias, data_out, M, N, K,
        DivCeil(K, SplitK), alpha);
  }
}

/**
 * @brief A16W8 GEMM 32x128x32 16816 TensorCore. Support SplitK.
 *
 * FP16 * FP16 += FP32
 * BF16 * BF16 += FP32
 *
 * K % 2 == 0
 * N % 4 == 0
 * b.x = (M + 32 - 1) /32
 * b.y = (N + 128 - 1) / 128
 * b.z = (K + SplitK - 1) / SplitK
 *
 */
template <typename FT, typename QT>
__global__ void hgemm_a16w8_32x128x32_16816_nn_splitk_kernel(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev, FT* Cdev,
    const int32_t M, const int32_t N, const int32_t K, const int32_t GroupSize,
    const uint32_t SplitK) {
  constexpr uint32_t BLOCK_TILE_M = 32;
  constexpr uint32_t BLOCK_TILE_N = 128;
  constexpr uint32_t BLOCK_TILE_K = 32;
  constexpr uint32_t WARP_TILE_M = 16;
  constexpr uint32_t WARP_TILE_K = 16;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // A_ldg_guard avoid LDG out of bound M
  bool A_ldg_guard[BLOCK_TILE_M / WARP_TILE_M]{};
#pragma unroll
  for (uint32_t i = 0; i < BLOCK_TILE_M / WARP_TILE_M; ++i) {
    const uint32_t m_idx = blockIdx.x * BLOCK_TILE_M + i * WARP_TILE_M +
                           warp_id * 2 + lane_id / 16;
    if (m_idx < M) {
      A_ldg_guard[i] = true;
    }
  }
  // B_ldg_guard avoid LDG out of bound N
  bool B_ldg_guard =
      (blockIdx.y * BLOCK_TILE_N + lane_id * 4) < N ? true : false;
  // QParam_ldg_guard avoid LDG out of bound N
  bool QParam_ldg_guard[2]{};
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    if (blockIdx.y * BLOCK_TILE_N + warp_id * 16 + lane_id / 4 + i * 8 < N) {
      QParam_ldg_guard[i] = true;
    }
  }

  // ShareMem
  // uint32_t Asmem[2][576]   (32 + 4) * 8 * 2 * 2(double_buffer) = 1152
  // uint32_t B_smem[2][1056] (32 + 1) * 16 * 2 * 2(double_buffer) = 2112
  __shared__ __align__(16 * 1024) uint32_t smem[2 * 576 + 2 * 1056];
  uint32_t* A_smem = smem;
  uint32_t* B_smem = smem + 2 * 576;
  uint32_t A_sts_addr =
      smem_u32addr(A_smem + ((lane_id / 4 % 4) * 2 + warp_id / 4) * 36 +
                   warp_id % 4 * 8 + lane_id / 16 * 4 + lane_id % 4);
  uint32_t A_lds_addr[2][4];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      A_lds_addr[i][j] = smem_u32addr(A_smem + i * (36 * 8) + 36 * j + lane_id);
    }
  }
  const uint32_t A_smem_buf_offset = 576;
  uint32_t B_sts_addr =
      smem_u32addr(B_smem + warp_id / 4 * (16 * 33) + lane_id / 2 * 33 +
                   lane_id % 2 * 16 + warp_id % 4);
  uint32_t B_lds_addr[2];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    B_lds_addr[i] = smem_u32addr(B_smem + (warp_id * 2 + i) * 33 + lane_id);
  }
  const uint32_t B_smem_buf_offset = 1056;

  const FT* Adev_ptr =
      Adev + blockIdx.x * BLOCK_TILE_M * K + blockIdx.z * SplitK;
  const QT* Bdev_ptr =
      Bdev + blockIdx.y * BLOCK_TILE_N + blockIdx.z * SplitK * N;
  const FT* BSdev_ptr = BSdev + blockIdx.z * SplitK / GroupSize * N +
                        blockIdx.y * BLOCK_TILE_N + warp_id * 16 + lane_id / 4;
  const FT* BZdev_ptr = BZdev + blockIdx.z * SplitK / GroupSize * N +
                        blockIdx.y * BLOCK_TILE_N + warp_id * 16 + lane_id / 4;

  uint32_t ld_smem_idx = 0;
  uint32_t st_smem_idx = 0;

  uint32_t lda[2] = {0};
  uint32_t ldb[4] = {0};
  uint32_t ldb_trans[4] = {0};

  FT scale[2];
  FT zero[2];
  const uint32_t KDepth = min(SplitK, K - blockIdx.z * SplitK);
  const uint32_t k_tile_num = (KDepth + BLOCK_TILE_K - 1) / BLOCK_TILE_K - 1;
  // First Tile
  {
    uint32_t first_k_tile = KDepth - k_tile_num * BLOCK_TILE_K;
    const FT* Adev_first_ptr = Adev_ptr + (KDepth - first_k_tile);
    const QT* Bdev_first_ptr = Bdev_ptr + (KDepth - first_k_tile) * N;
    const FT* BSdev_first_ptr =
        BSdev_ptr + (KDepth - first_k_tile) / GroupSize * N;
    const FT* BZdev_first_ptr =
        BZdev_ptr + (KDepth - first_k_tile) / GroupSize * N;
// Load A from gmem to reg.
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_TILE_M / WARP_TILE_M; ++i) {
      const uint32_t row = i * WARP_TILE_M + warp_id * 2 + lane_id / 16;
      const uint32_t col = lane_id % 16 * 2;
      lda[i] = A_ldg_guard[i] && col < first_k_tile
                   ? reinterpret_cast<const uint32_t*>(Adev_first_ptr +
                                                       row * K + col)[0]
                   : uint32_t(0);
    }
    // Load B from gmem to reg
    {
      const uint32_t row = 16 * (warp_id / 4) + (warp_id % 4) * 2;
      const uint32_t col = lane_id * 4;
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          ldb[i * 2 + j] =
              (row + i * 8 + j) < first_k_tile && B_ldg_guard
                  ? reinterpret_cast<const uint32_t*>(
                        Bdev_first_ptr + (row + i * 8 + j) * N + col)[0]
                  : uint32_t(0);
        }
      }
    }
// Load scale and zero
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t offset = i * 8;
      scale[i] = QParam_ldg_guard[i] ? BSdev_first_ptr[offset] : FT(1);
      zero[i] = QParam_ldg_guard[i] ? BZdev_first_ptr[offset] : FT(0);
    }
// Store AB from reg to smem
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_TILE_M / WARP_TILE_M; ++i) {
      sts32(lda[i],
            A_sts_addr + (st_smem_idx * A_smem_buf_offset + i * (36 * 8)) *
                             sizeof(uint32_t));
    }
    transpose4x4x8bit(ldb, ldb_trans);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      sts32(ldb_trans[i],
            B_sts_addr +
                (st_smem_idx * B_smem_buf_offset + i * 4) * sizeof(uint32_t));
    }
    ld_smem_idx = st_smem_idx;
    st_smem_idx ^= 1;
    __syncthreads();
  }

  hie::Array<FT, 8> FragmentA[2];
  hie::Array<FT, 4> FragmentB[2];
  hie::Array<float, 4> FragmentC[2][2];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      FragmentC[i][j][0] = float(0);
      FragmentC[i][j][1] = float(0);
      FragmentC[i][j][2] = float(0);
      FragmentC[i][j][3] = float(0);
    }
  }

  const FT* Adev_ldg_ptr =
      Adev_ptr + (warp_id * 2 + lane_id / 16) * K + lane_id % 16 * 2;
  const QT* Bdev_ldg_ptr =
      Bdev_ptr + (16 * (warp_id / 4) + (warp_id % 4) * 2) * N + lane_id * 4;
  for (uint32_t k_tile_idx = 0; k_tile_idx < k_tile_num; ++k_tile_idx) {
    // Load next gdata to reg
    {
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_TILE_M / WARP_TILE_M; ++i) {
        const uint32_t offset = i * WARP_TILE_M * K;
        lda[i] =
            A_ldg_guard[i]
                ? reinterpret_cast<const uint32_t*>(Adev_ldg_ptr + offset)[0]
                : uint32_t(0);
      }
      if (B_ldg_guard) {
        ldb[0] = reinterpret_cast<const uint32_t*>(Bdev_ldg_ptr + 0 * N)[0];
        ldb[1] = reinterpret_cast<const uint32_t*>(Bdev_ldg_ptr + 1 * N)[0];
        ldb[2] = reinterpret_cast<const uint32_t*>(Bdev_ldg_ptr + 8 * N)[0];
        ldb[3] = reinterpret_cast<const uint32_t*>(Bdev_ldg_ptr + 9 * N)[0];
      }
      Adev_ldg_ptr += BLOCK_TILE_K;
      Bdev_ldg_ptr += BLOCK_TILE_K * N;
    }

    uint32_t reg_b[2];
// Main : Loop
#pragma unroll
    for (uint32_t k_frag_idx = 0; k_frag_idx < BLOCK_TILE_K / WARP_TILE_K;
         ++k_frag_idx) {
// load AB from smem to reg
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
        lds32<uint32_t>(reg_b[i],
                        B_lds_addr[i] + (ld_smem_idx * B_smem_buf_offset +
                                         k_frag_idx * 16 * 33) *
                                            sizeof(uint32_t));
      }
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          lds32<uint32_t>(reinterpret_cast<uint32_t*>(FragmentA[i].data)[j],
                          A_lds_addr[i][j] + (ld_smem_idx * A_smem_buf_offset +
                                              k_frag_idx * 4 * 36) *
                                                 sizeof(uint32_t));
        }
      }
// TODO: OPT
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
        const QT* i8_ptr = reinterpret_cast<const QT*>(&reg_b[i]);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          FragmentB[i][j] = (FT(i8_ptr[j]) - zero[i]) * scale[i];
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          hmma_16816<FT, float>(FragmentA[i], FragmentB[j], FragmentC[i][j],
                                FragmentC[i][j]);
        }
      }
    }  // Main : Loop

// Load next scale and zero
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t offset = k_tile_idx * BLOCK_TILE_K / GroupSize * N + i * 8;
      scale[i] = QParam_ldg_guard[i] ? BSdev_ptr[offset] : FT(1);
      zero[i] = QParam_ldg_guard[i] ? BZdev_ptr[offset] : FT(0);
    }

    // Store reg data to smem
    {
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_TILE_M / WARP_TILE_M; ++i) {
        sts32<uint32_t>(lda[i], A_sts_addr + (st_smem_idx * A_smem_buf_offset +
                                              i * (36 * 8)) *
                                                 sizeof(uint32_t));
      }
      transpose4x4x8bit(ldb, ldb_trans);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        sts32(ldb_trans[i],
              B_sts_addr +
                  (st_smem_idx * B_smem_buf_offset + i * 4) * sizeof(uint32_t));
      }
      ld_smem_idx = st_smem_idx;
      st_smem_idx ^= 1;
      __syncthreads();
    }
  }
  // Compute last tile
  {
    uint32_t reg_b[2];
#pragma unroll
    for (uint32_t k_frag_idx = 0; k_frag_idx < BLOCK_TILE_K / WARP_TILE_K;
         ++k_frag_idx) {
// load AB from smem to reg
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
        lds32<uint32_t>(reg_b[i],
                        B_lds_addr[i] + (ld_smem_idx * B_smem_buf_offset +
                                         k_frag_idx * 16 * 33) *
                                            sizeof(uint32_t));
      }
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          lds32<uint32_t>(reinterpret_cast<uint32_t*>(FragmentA[i].data)[j],
                          A_lds_addr[i][j] + (ld_smem_idx * A_smem_buf_offset +
                                              k_frag_idx * 4 * 36) *
                                                 sizeof(uint32_t));
        }
      }
// TODO: OPT
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
        const QT* i8_ptr = reinterpret_cast<const QT*>(&reg_b[i]);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          FragmentB[i][j] = (FT(i8_ptr[j]) - zero[i]) * scale[i];
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          hmma_16816<FT, float>(FragmentA[i], FragmentB[j], FragmentC[i][j],
                                FragmentC[i][j]);
        }
      }
    }
  }

  hie::Array<FT, 4> FragmentC_ST[2][2];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      for (uint32_t t = 0; t < 4; ++t) {
        FragmentC_ST[i][j][t] = FT(FragmentC[i][j][t]);
      }
    }
  }

  __syncthreads();
  uint32_t* C_smem = smem;
  const uint32_t C_smem_offset[3] = {32 * 36, 36, 1};
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      C_smem[(warp_id / 4) * C_smem_offset[0] +
             (i * 16 + lane_id / 4) * C_smem_offset[1] +
             (warp_id % 4 * 8 + j * 4 + lane_id % 4) * C_smem_offset[2]] =
          reinterpret_cast<uint32_t*>(FragmentC_ST[i][j].data)[0];
      C_smem[(warp_id / 4) * C_smem_offset[0] +
             (i * 16 + lane_id / 4 + 8) * C_smem_offset[1] +
             (warp_id % 4 * 8 + j * 4 + lane_id % 4) * C_smem_offset[2]] =
          reinterpret_cast<uint32_t*>(FragmentC_ST[i][j].data)[1];
    }
  }
  __syncthreads();

  const uint32_t m_idx = blockIdx.x * BLOCK_TILE_M;
  uint32_t* Cdev_ptr = reinterpret_cast<uint32_t*>(
      Cdev + blockIdx.z * M * N + blockIdx.x * BLOCK_TILE_M * N +
      blockIdx.y * BLOCK_TILE_N);
#pragma unroll
  for (uint32_t mi = warp_id; mi < BLOCK_TILE_M; mi += 8) {
#pragma unroll
    for (uint32_t ni = lane_id; ni < BLOCK_TILE_N / 2; ni += 32) {
      if ((m_idx + mi) < M && (2 * ni + blockIdx.y * BLOCK_TILE_N) < N) {
        Cdev_ptr[mi * N / 2 + ni] =
            C_smem[(ni / 32) * C_smem_offset[0] + mi * C_smem_offset[1] +
                   (ni % 32) * C_smem_offset[2]];
      }
    }
  }
}

template <typename FT, typename QT, template <class> class ActiveFunc>
void hgemm_a16w8_subc_32x128x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const int sm_count, const float alpha, cudaStream_t stream) {
  constexpr float SPLIT_THRESHOLD = 4;

  constexpr uint32_t BLOCK_TILE_M = 32;
  constexpr uint32_t BLOCK_TILE_N = 128;

  uint32_t grid_x = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
  uint32_t grid_y = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
  uint32_t grid_z;

  uint32_t num_slice;
  for (num_slice = 1; num_slice < K / 256; ++num_slice) {
    uint32_t num_block = grid_x * grid_y * num_slice;
    if (num_block > sm_count * SPLIT_THRESHOLD &&
        (num_block % sm_count == 0 || num_block % sm_count >= sm_count / 2)) {
      break;
    }
  }
  const uint32_t SplitK =
      (K / num_slice) % GroupSize == 0
          ? (K / num_slice)
          : (K / num_slice / GroupSize * GroupSize + GroupSize);
  grid_z = (K + SplitK - 1) / SplitK;
  {
    dim3 block_size(256);
    dim3 grid_size(grid_x, grid_y, grid_z);
    hgemm_a16w8_32x128x32_16816_nn_splitk_kernel<FT, QT>
        <<<grid_size, block_size, 0, stream>>>(Adev, Bdev, BSdev, BZdev,
                                               static_cast<FT*>(workspace), M,
                                               N, K, GroupSize, SplitK);
  }
  {
    const uint32_t THREADS_PER_BLOCK = 128;
    const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), M, 1);
    reduce_sum<FT, ActiveFunc><<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<const FT*>(workspace), bias, Cdev, M, N, K,
        DivCeil(K, SplitK), alpha);
  }
}

/**********************/
/**********************/
template void
hgemm_A16W8_subc_32x128x16_simt_Aldg1<int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const SplitKParams, const float, cudaStream_t);

template void
hgemm_A16W8_subc_32x128x16_simt_Aldg1<int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const SplitKParams, const float, cudaStream_t);

template void
hgemm_A16W8_subc_32x128x16_simt_Aldg1<int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const SplitKParams, const float, cudaStream_t);

template void
hgemm_A16W8_subc_32x128x16_simt_Aldg1<int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const SplitKParams, const float, cudaStream_t);

template void
hgemm_A16W8_subc_32x128x16_simt_Aldg1<int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const SplitKParams, const float, cudaStream_t);

template void
hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void
hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void
hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void
hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void
hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void gemv_a16w8_subc_splitk<half, int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);
template void gemv_a16w8_subc_splitk<half, int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);
template void gemv_a16w8_subc_splitk<half, int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);
template void gemv_a16w8_subc_splitk<half, int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);
template void gemv_a16w8_subc_splitk<half, int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const float, cudaStream_t);

template void
gemv_a16w8_subc_splitk<hie::bfloat16, int8_t, hie::activation::Identity>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);
template void
gemv_a16w8_subc_splitk<hie::bfloat16, int8_t, hie::activation::Gelu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);
template void
gemv_a16w8_subc_splitk<hie::bfloat16, int8_t, hie::activation::GeluTanh>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);
template void
gemv_a16w8_subc_splitk<hie::bfloat16, int8_t, hie::activation::Relu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);
template void
gemv_a16w8_subc_splitk<hie::bfloat16, int8_t, hie::activation::Silu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

//
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<
    half, int8_t, hie::activation::Identity>(const half*, const int8_t*,
                                             const half*, const half*,
                                             const half*, half*, const uint32_t,
                                             const uint32_t, const uint32_t,
                                             const uint32_t, void*, const int,
                                             const float, cudaStream_t);
template void
hgemm_a16w8_subc_32x128x32_16816_nn_splitk<half, int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const float, cudaStream_t);
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<
    half, int8_t, hie::activation::GeluTanh>(const half*, const int8_t*,
                                             const half*, const half*,
                                             const half*, half*, const uint32_t,
                                             const uint32_t, const uint32_t,
                                             const uint32_t, void*, const int,
                                             const float, cudaStream_t);
template void
hgemm_a16w8_subc_32x128x32_16816_nn_splitk<half, int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const float, cudaStream_t);
template void
hgemm_a16w8_subc_32x128x32_16816_nn_splitk<half, int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, const uint32_t, void*,
    const int, const float, cudaStream_t);

template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<
    hie::bfloat16, int8_t, hie::activation::Identity>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const float, cudaStream_t);
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<hie::bfloat16, int8_t,
                                                         hie::activation::Gelu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const float, cudaStream_t);
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<
    hie::bfloat16, int8_t, hie::activation::GeluTanh>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const float, cudaStream_t);
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<hie::bfloat16, int8_t,
                                                         hie::activation::Relu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const float, cudaStream_t);
template void hgemm_a16w8_subc_32x128x32_16816_nn_splitk<hie::bfloat16, int8_t,
                                                         hie::activation::Silu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const float, cudaStream_t);

}  // namespace cuda
}  // namespace allspark