/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_subc_kernel.cu
 */

#include "../hie/cuda_activation.hpp"
#include "convert_4bit.h"
#include "gemm_a16w4_kernel.h"
#include "gemm_lowp_utils.cuh"

namespace allspark {
namespace cuda {

__device__ __forceinline__ int get_octid(int lane_id) {
  return lane_id % 16 / 4;
}

static constexpr int WARP_SIZE = 32;

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * QType is uint8, contains 2 4-bit unsigned
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_Subc_A16W4_32x256x32_SM70_SplitK {
  static_assert(std::is_same<QType, uint8_t>::value,
                "A16W4 GmemTile only support uint8");
  // element num loaded by a LDG inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 8;
  const int WARP_CNT = BLOCK_SIZE / WARP_SIZE;  // 4

  __device__ GmemTile_Subc_A16W4_32x256x32_SM70_SplitK(
      const SM70_GEMM_A16W4_Params<half, uint8_t>& k_params,
      const uint32_t& A_smem_base_addr, const uint32_t& B_smem_base_addr)
      : params(k_params),
        A_smem_addr(A_smem_base_addr),
        B_smem_addr(B_smem_base_addr) {
    this_block_A_base_ptr =
        params.A_ptr + blockIdx.x * 32 * params.K + blockIdx.z * params.SplitK;
    ;
    this_block_B_base_ptr =
        params.B_ptr +
        (blockIdx.y * 256 + blockIdx.z * params.SplitK * params.N) / 2;

    const int warp_id = threadIdx.x / WARP_SIZE;
    lane_id = threadIdx.x % WARP_SIZE;

    // For matrix A, a block load/store 32(row) x 32(col) elements in 1 iter
    // 8x4 warp load/store 8(row) x 32(col) elements
    const int load_a_row_idx = threadIdx.x / 4;
    load_a_col_idx = threadIdx.x % 4 * LDG_ELEMENT_CNT_A;
    load_a_offset = load_a_row_idx * params.K + load_a_col_idx;

    // For matrix B, a block load/store 32(row) * 256(col) elements in 8 iter
    // and a block load/store 4(row) * 256(col) elements per iter, a warp
    // load/store 1(row) * 256(col) per iter
    const int load_b_col_idx = lane_id * LDG_ELEMENT_CNT_B;
    load_b_row_base_idx = warp_id * 8;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_offset =
        threadIdx.x / 4 * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;

    store_b_base_offset = warp_id * 8 * 256;

    A_ldg_guard = false;
    int m_idx = blockIdx.x * 32 + load_a_row_idx;
    if (m_idx < params.M) {
      A_ldg_guard = true;
    }

    B_ldg_guard = false;
    n_idx = blockIdx.y * 256 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

    B_k_idx = blockIdx.z * params.SplitK;
  }

  __device__ void ldg_first_k_tile(const int& first_k_tile,
                                   const int tb_k_slice) {
    // load first k_tile
    const QType* this_B_ptr =
        this_block_B_base_ptr + (tb_k_slice - first_k_tile) * params.N / 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // Note: GroupSize must be multiples of 8, here a warp load continuous 8
      // rows, so can correspond to the same quantization parameters in the K
      // direction
      int load_b_offset = load_b_base_offset + i * params.N;
      bool guard = B_ldg_guard && (load_b_row_base_idx + i) < first_k_tile;
      ldg32_cs_0(b_regs_q[i], this_B_ptr + (load_b_offset / 2), guard);
    }

    int k_idx = B_k_idx + tb_k_slice - first_k_tile + load_b_row_base_idx;
    const half* this_B_scale_ptr =
        params.B_scale_ptr + (k_idx / params.GroupSize) * params.N + n_idx;
    const half* this_B_zero_ptr =
        params.B_zero_ptr + (k_idx / params.GroupSize) * params.N + n_idx;
    ldg128_ca_0(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
                b_regs_scale[3], this_B_scale_ptr, B_ldg_guard);
    ldg128_ca_0(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
                this_B_zero_ptr, B_ldg_guard);

    const half* this_A_ptr = this_block_A_base_ptr + tb_k_slice - first_k_tile;
    bool guard = A_ldg_guard && load_a_col_idx < first_k_tile;
    ldg128_cg_0(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                this_A_ptr + load_a_offset, guard);
  }

  __device__ void ldg() {
// load B
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * params.N;
      ldg32_cs(b_regs_q[i], this_block_B_base_ptr + (load_b_offset / 2),
               B_ldg_guard);
    }

    // load B_scale and B_zero
    const int k_idx = B_k_idx + load_b_row_base_idx;
    const half* this_B_scale_ptr =
        params.B_scale_ptr + (k_idx / params.GroupSize) * params.N + n_idx;
    const half* this_B_zero_ptr =
        params.B_zero_ptr + (k_idx / params.GroupSize) * params.N + n_idx;
    ldg128_ca(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
              b_regs_scale[3], this_B_scale_ptr, B_ldg_guard);
    ldg128_ca(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
              this_B_zero_ptr, B_ldg_guard);

    // load A
    ldg128_cg(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
              this_block_A_base_ptr + load_a_offset, A_ldg_guard);

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N / 2;
    B_k_idx += 32;
  }

  __device__ void dq() {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      cvt_u4x8_to_halfx8(b_regs_q[i], b_regs[i]);
      b_regs[i][0].x =
          __hmul(__hsub(b_regs[i][0].x, b_regs_zero[0].x), b_regs_scale[0].x);
      b_regs[i][0].y =
          __hmul(__hsub(b_regs[i][0].y, b_regs_zero[0].y), b_regs_scale[0].y);
      b_regs[i][1].x =
          __hmul(__hsub(b_regs[i][1].x, b_regs_zero[1].x), b_regs_scale[1].x);
      b_regs[i][1].y =
          __hmul(__hsub(b_regs[i][1].y, b_regs_zero[1].y), b_regs_scale[1].y);
      b_regs[i][2].x =
          __hmul(__hsub(b_regs[i][2].x, b_regs_zero[2].x), b_regs_scale[2].x);
      b_regs[i][2].y =
          __hmul(__hsub(b_regs[i][2].y, b_regs_zero[2].y), b_regs_scale[2].y);
      b_regs[i][3].x =
          __hmul(__hsub(b_regs[i][3].x, b_regs_zero[3].x), b_regs_scale[3].x);
      b_regs[i][3].y =
          __hmul(__hsub(b_regs[i][3].y, b_regs_zero[3].y), b_regs_scale[3].y);
    }
  }

  __device__ void sts() {
    // store A from reg to smem
    sts128(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
           A_smem_addr + store_a_offset * 2 /* ELEM_SIZE */);

// dequant and store B from reg to smem
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int store_b_offset = store_b_base_offset + i * 256 +
                                 (lane_id ^ ((i % 4) * 2)) * LDG_ELEMENT_CNT_B;
      sts128(b_regs[i][0], b_regs[i][1], b_regs[i][2], b_regs[i][3],
             B_smem_addr + store_b_offset * 2 /* ELEM_SIZE */);
    }
  }

  const half* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;
  int load_a_offset;
  int load_b_base_offset;
  int store_a_offset;
  int store_b_base_offset;

  int load_a_col_idx;
  int load_b_row_base_idx;

  int n_idx;
  int B_k_idx;

  int lane_id;

  bool A_ldg_guard;
  bool B_ldg_guard;

  uint32_t a_regs[4];

  __half2 b_regs[8][4];
  uint32_t b_regs_q[8];
  __half2 b_regs_scale[4];
  __half2 b_regs_zero[4];

  const uint32_t A_smem_addr, B_smem_addr;
  const SM70_GEMM_A16W4_Params<half, QType>& params;
};

/*
 * warp_tile : m32n64k32
 */
template <typename QType>
struct ComputeTile_f16_32x256x32_SM70_SplitK {
  __device__ ComputeTile_f16_32x256x32_SM70_SplitK(
      const SM70_GEMM_A16W4_Params<half, QType>& k_params,
      const uint32_t& A_smem_base_addr, const uint32_t& B_smem_base_addr)
      : params(k_params),
        A_smem_addr(A_smem_base_addr),
        B_smem_addr(B_smem_base_addr) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int oct_id = get_octid(lane_id);

    int row_idx_in_oct = lane_id / 16 * 4 + lane_id % 4;
    A_kphase_col_adjust = (row_idx_in_oct / 4) ^ (row_idx_in_oct % 4);
    load_a_base_offset = (oct_id / 2 * 8 + row_idx_in_oct) * 32;

    load_b_base_offset[0] =
        (lane_id % 4) * 256 + warp_id * 64 +
        (((oct_id % 2) * 2 + (lane_id / 16)) ^ ((lane_id % 4) * 4)) * 4;
    load_b_base_offset[1] =
        (lane_id % 4) * 256 + warp_id * 64 +
        (((oct_id % 2) * 2 + (lane_id / 16) + 4) ^ ((lane_id % 4) * 4)) * 4;
    load_b_base_offset[2] =
        (lane_id % 4) * 256 + warp_id * 64 +
        (((oct_id % 2) * 2 + (lane_id / 16) + 8) ^ ((lane_id % 4) * 4)) * 4;
    load_b_base_offset[3] =
        (lane_id % 4) * 256 + warp_id * 64 +
        (((oct_id % 2) * 2 + (lane_id / 16) + 12) ^ ((lane_id % 4) * 4)) * 4;
    store_c_row_base_idx = oct_id / 2 * 8 + lane_id / 16 * 4 + lane_id % 4;
    store_c_col_base_idx = oct_id % 2 * 8 + warp_id * 64;
    store_c_base_offset =
        store_c_row_base_idx * params.N + store_c_col_base_idx;

    this_block_C_base_ptr = params.C_split_ptr +
                            blockIdx.z * params.M * params.N +
                            blockIdx.x * 32 * params.N + blockIdx.y * 256;
  }

  // load 32 * 8 A elements per warp per k_phase
  __device__ void lds_A(const int k_phase_idx) {
    int load_a_offset =
        load_a_base_offset + (k_phase_idx ^ A_kphase_col_adjust) * 8;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds128(A_frag[i][0], A_frag[i][1], A_frag[i][2], A_frag[i][3],
             A_smem_addr + (load_a_offset + i * 16 * 32) * 2 /* ELEM SIZE */);
    }
  }

  // load 8 * 64 B elements per warp per k_phase
  __device__ void lds_B(const int k_phase_idx, const int step_idx) {
    int k_phase_offset = k_phase_idx * 8 * 256;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds64(B_frag[i][0], B_frag[i][1],
            B_smem_addr + (load_b_base_offset[step_idx * 2] + k_phase_offset +
                           i * 4 * 256) *
                              2 /* ELEM SIZE */);
      lds64(B_frag[i][2], B_frag[i][3],
            B_smem_addr + (load_b_base_offset[step_idx * 2 + 1] +
                           k_phase_offset + i * 4 * 256) *
                              2 /* ELEM SIZE */);
    }
  }

  __device__ void mma(const int step_idx) {
#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 2; ++n_idx) {
        mma_h884(A_frag[m_idx][0], A_frag[m_idx][1], B_frag[0][n_idx * 2],
                 B_frag[0][n_idx * 2 + 1],
                 C_frag[m_idx][n_idx + step_idx * 2][0],
                 C_frag[m_idx][n_idx + step_idx * 2][1],
                 C_frag[m_idx][n_idx + step_idx * 2][2],
                 C_frag[m_idx][n_idx + step_idx * 2][3]);
        mma_h884(A_frag[m_idx][2], A_frag[m_idx][3], B_frag[1][n_idx * 2],
                 B_frag[1][n_idx * 2 + 1],
                 C_frag[m_idx][n_idx + step_idx * 2][0],
                 C_frag[m_idx][n_idx + step_idx * 2][1],
                 C_frag[m_idx][n_idx + step_idx * 2][2],
                 C_frag[m_idx][n_idx + step_idx * 2][3]);
      }
    }
  }

  __device__ void stg() {
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 2; st_iter_m++) {
      for (int st_iter_n = 0; st_iter_n < 4; st_iter_n++) {
        half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                      st_iter_m * 16 * params.N + st_iter_n * 16;
        bool guard = (blockIdx.x * 32 + store_c_row_base_idx + st_iter_m * 16) <
                         params.M &&
                     (blockIdx.y * 256 + store_c_col_base_idx +
                      st_iter_n * 16) < params.N;
        stg128(C_frag[st_iter_m][st_iter_n][0], C_frag[st_iter_m][st_iter_n][1],
               C_frag[st_iter_m][st_iter_n][2], C_frag[st_iter_m][st_iter_n][3],
               C_ptr, guard);
      }
    }
  }

  const SM70_GEMM_A16W4_Params<half, QType>& params;

  int load_a_base_offset;
  int load_b_base_offset[4];
  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_base_idx;
  half* this_block_C_base_ptr = nullptr;

  const uint32_t A_smem_addr, B_smem_addr;

  // first 2 denotes M direction, second 4 denotes K direction
  uint32_t A_frag[2][4];
  // first 2 denotes K direction, second 4 denotes N direction
  uint32_t B_frag[2][4];
  // first 2 denotes M direction, second 4 denotes N direction
  uint32_t C_frag[2][4][4]{};

  int A_kphase_col_adjust;
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: K x N, matrix C: M x N (row-major)
 *  K % 8 == 0 && N % 8 == 0
 *  accumulator precision: FP16
 *  output datatype: FP16
 *
 *  BLOCK_TILE: m32n256k32
 *  BLOCK_SIZE: 128
 *  WARP_TILE:  m32n64k32
 *
 *  GroupSize must be multiples of 8 and greater than 32
 */
template <typename QType>
__global__ void __launch_bounds__(128, 4)
    volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk_kernel(
        const SM70_GEMM_A16W4_Params<half, QType> params) {
  // A smem size = 32 * 32 * 2B/elem = 4KB
  // B smem size = 256 * 32 * 2B/elem = 16KB
  static constexpr int SMEM_SIZE = 32 * 32 * 2 + 256 * 32 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + 32 * 32 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);

  // initialize the data move process from GM to SMEM for this block
  GmemTile_Subc_A16W4_32x256x32_SM70_SplitK<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr);

  int tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                       ? params.SplitK
                       : params.K - blockIdx.z * params.SplitK;
  int k_main_loop = (tb_k_slice + 31) / 32 - 1;
  int first_k_tile = tb_k_slice - k_main_loop * 32;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_f16_32x256x32_SM70_SplitK<QType> compute_tile(params, A_smem_addr,
                                                            B_smem_addr);
  gmem_tile.dq();
  gmem_tile.sts();

  for (int k_tile_idx = 0; k_tile_idx < k_main_loop; k_tile_idx++) {
    __syncthreads();
    gmem_tile.ldg();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
      compute_tile.lds_A(k_phase_idx);
      compute_tile.lds_B(k_phase_idx, 0);
      compute_tile.mma(0);

      compute_tile.lds_B(k_phase_idx, 1);
      compute_tile.mma(1);
    }

    gmem_tile.dq();
    __syncthreads();
    gmem_tile.sts();
  }

  __syncthreads();

#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
    compute_tile.lds_A(k_phase_idx);
    compute_tile.lds_B(k_phase_idx, 0);
    compute_tile.mma(0);

    compute_tile.lds_B(k_phase_idx, 1);
    compute_tile.mma(1);
  }

  compute_tile.stg();
}

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const int GroupSize, void* workspace,
    const SplitKParams splitk_params, cudaStream_t stream) {
  uint32_t grid_x = (M + 31) / 32;
  uint32_t grid_y = (N + 255) / 256;
  uint32_t grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  int GroupCnt = (K + GroupSize - 1) / GroupSize;
  half* C_split = static_cast<half*>(workspace);

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(128);

  SM70_GEMM_A16W4_Params<half, QType> params{
      A,        B,        B_scale, B_zero,  C,
      (int)M,   (int)N,   (int)K,  C_split, splitk_params.SplitK,
      GroupCnt, GroupSize};

  volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk_kernel<QType>
      <<<grid, block, 0, stream>>>(params);

  // SplitK reduce
  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                           grid_z, 1.0f, stream);
}

/**
 * @brief
 *
 * 32x256x32_32x32_16816_NN_SPLITK_A16W4_SUBC
 *
 * 256 threads
 *
 * 一个Warp处理2x4个16816块, 大小是[16x2]x[8x4]x16.
 *
 * K % 8 == 0
 * N % 8 == 0
 *
 * b.x = (M + 32 - 1) / 32
 * b.y = (N + 256 - 1) / 256
 * b.z = (K + SplitK - 1) / SplitK;
 *
 */

template <typename FT, typename QT>
__global__ void __launch_bounds__(256, 3)
    hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk_kernel(
        const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
        FT* Cdev, const uint32_t M, const uint32_t N, const uint32_t K,
        const uint32_t SplitK, U32DivMod groupsize_divmod) {
  constexpr uint32_t BLOCK_TILE_M = 32;
  constexpr uint32_t BLOCK_TILE_N = 256;
  constexpr uint32_t BLOCK_TILE_K = 32;

  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  // A_ldg_guard avoid LDG out of bound M
  const uint32_t m_idx = blockIdx.x * BLOCK_TILE_M + warp_id * 8 + lane_id / 4;
  ;
  bool A_ldg_guard = m_idx < M ? true : false;
  // B_ldg_guard avoid LDG out of bound N
  const uint32_t n_idx = blockIdx.y * BLOCK_TILE_N + lane_id * 8;
  bool B_ldg_guard = n_idx < N ? true : false;

  // A : 32 * 32 * 2 = 2048
  // B : 32 * 256 * 2 = 16384
  // Double buffer x2 = 18432 * 2 = 36K
  __shared__ __align__(1024) char smem[36864];
  FT* A_smem = reinterpret_cast<FT*>(smem);
  FT* B_smem = reinterpret_cast<FT*>(smem + 2048);

  uint32_t A_lds_addr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int lds_a_row = lane_id % 16 / 2;
    int lds_a_col = lane_id % 2 * 4 + ((lane_id % 8 / 2)) ^ (lane_id / 16);
    const uint32_t lds_a_offset = (lds_a_row * 8 + lds_a_col) ^ (i % 2 * 2);
    A_lds_addr[i] = __cvta_generic_to_shared(A_smem + lds_a_offset * 8);
  }
  uint32_t B_lds_addr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t lds_b_row = warp_id / 2 * 16 + lane_id % 16;
    const uint32_t lds_b_col =
        (lane_id % 8 ^ (lane_id / 16)) ^ (warp_id % 2 * 4) + i * 2;
    const uint32_t lds_b_offset = (lds_b_row * 8 + lds_b_col) * 8;
    B_lds_addr[i] = __cvta_generic_to_shared(B_smem + lds_b_offset);
  }

  const uint32_t A_sts_addr = __cvta_generic_to_shared(
      A_smem + warp_id * 4 * 8 * 8 + ((lane_id / 8) ^ (lane_id)) * 8);
  const uint32_t sts_b_row = lane_id / 8 * 16 + warp_id;
  const uint32_t sts_b_col = (lane_id % 8) ^ warp_id;
  const uint32_t B_sts_addr =
      __cvta_generic_to_shared(B_smem + (sts_b_row * 8 + sts_b_col) * 8);

  const FT* Adev_ptr =
      Adev + blockIdx.x * BLOCK_TILE_M * K + blockIdx.z * SplitK;
  ;
  const QT* Bdev_ptr =
      Bdev + (blockIdx.y * BLOCK_TILE_N + blockIdx.z * SplitK * N) / 2;
  const FT* BSdev_ptr = BSdev + blockIdx.y * BLOCK_TILE_N + lane_id * 8;
  const FT* BZdev_ptr = BZdev + blockIdx.y * BLOCK_TILE_N + lane_id * 8;

  int4 ldg_a_reg;
  uint32_t ldg_b_reg[4];
  typename HalfType<FT>::T2 sts_b_reg[4][4];
  typename HalfType<FT>::T2 BS[4];
  typename HalfType<FT>::T2 BZ[4];

  const uint32_t KDepth = min(SplitK, K - blockIdx.z * SplitK);
  uint32_t k_tile_num = (KDepth + BLOCK_TILE_K - 1) / BLOCK_TILE_K - 1;
  const FT* Adev_ldg_ptr =
      Adev_ptr + (warp_id * 8 + lane_id / 4) * K + lane_id % 4 * 8;
  const QT* Bdev_ldg_ptr = Bdev_ptr + (warp_id * N + lane_id * 8) / 2;
  // First
  {
    const uint32_t first_k_tile = KDepth - k_tile_num * BLOCK_TILE_K;
    const FT* Adev_first_ptr = Adev_ldg_ptr + (KDepth - first_k_tile);
    const QT* Bdev_first_ptr = Bdev_ldg_ptr + (KDepth - first_k_tile) * N / 2;

    // Load A from gmem to reg.
    if (warp_id < 4) {
      const uint32_t ldg_a_col = lane_id % 4 * 8;
      ldg128_cg_0<int32_t>(ldg_a_reg.x, ldg_a_reg.y, ldg_a_reg.z, ldg_a_reg.w,
                           Adev_first_ptr,
                           A_ldg_guard && ldg_a_col < first_k_tile);
    }

// Load B from gmem to reg.
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      ldg32_cg_0<uint32_t>(ldg_b_reg[i], Bdev_first_ptr + i * 8 * N / 2,
                           B_ldg_guard && (warp_id + 8 * i) < first_k_tile);
    }

    // Load Quantize params
    uint32_t this_k_idx = blockIdx.z * SplitK + KDepth - first_k_tile;
    ldg128_cg_0<typename HalfType<FT>::T2>(
        BS[0], BS[1], BS[2], BS[3],
        BSdev_ptr + groupsize_divmod.Div(this_k_idx) * N, B_ldg_guard);
    ldg128_cg_0<typename HalfType<FT>::T2>(
        BZ[0], BZ[1], BZ[2], BZ[3],
        BZdev_ptr + groupsize_divmod.Div(this_k_idx) * N, B_ldg_guard);

    // STS A from reg to smem
    if (warp_id < 4) {
      sts128<int32_t>(ldg_a_reg.x, ldg_a_reg.y, ldg_a_reg.z, ldg_a_reg.w,
                      A_sts_addr);
    }

// CVT + DQ
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      cvt_4bx8_to_16bx8_v2(ldg_b_reg[i], sts_b_reg[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sts_b_reg[i][j] = dequantize_func<typename HalfType<FT>::T2>(
            sts_b_reg[i][j], BS[j], BZ[j]);
      }
    }
// STS B from reg to smem
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t sts_b_row = i / 2 * 64 + i % 2 * 8;
      const uint32_t sts_b_offset = sts_b_row * 64 * sizeof(FT);
      sts128<typename HalfType<FT>::T2>(sts_b_reg[i][0], sts_b_reg[i][1],
                                        sts_b_reg[i][2], sts_b_reg[i][3],
                                        B_sts_addr + sts_b_offset);
    }
    __syncthreads();
  }

  const uint32_t kSmemSize = 2048 + 16384;
  uint32_t smem_ld_offset = 0;
  uint32_t smem_st_offset = kSmemSize;

  hie::Array<FT, 8> FragmentA[2];
  hie::Array<FT, 4> FragmentB[4];
  hie::Array<float, 4> FragmentC[2][4];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      FragmentC[i][j][0] = float(0);
      FragmentC[i][j][1] = float(0);
      FragmentC[i][j][2] = float(0);
      FragmentC[i][j][3] = float(0);
    }
  }

  // Main-Loop
  for (int k_tile_idx = 0; k_tile_idx < k_tile_num; ++k_tile_idx) {
    // Load A from gmem to reg.
    if (warp_id < 4) {
      ldg128_cg<int32_t>(ldg_a_reg.x, ldg_a_reg.y, ldg_a_reg.z, ldg_a_reg.w,
                         Adev_ldg_ptr, A_ldg_guard);
    }
// Load B from gmem to reg.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_cg<uint32_t>(ldg_b_reg[i], Bdev_ldg_ptr + i * 8 * N / 2,
                         B_ldg_guard);
    }
    Adev_ldg_ptr += BLOCK_TILE_K;
    Bdev_ldg_ptr += BLOCK_TILE_K * N / 2;

    // Load Quantize params
    uint32_t this_k_idx = blockIdx.z * SplitK + k_tile_idx * BLOCK_TILE_K;
    uint32_t last_k_idx = this_k_idx - BLOCK_TILE_K;
    if (k_tile_idx == 0 || (groupsize_divmod.Div(this_k_idx) !=
                            groupsize_divmod.Div(last_k_idx))) {
      ldg128_cg<typename HalfType<FT>::T2>(
          BS[0], BS[1], BS[2], BS[3],
          BSdev_ptr + groupsize_divmod.Div(this_k_idx) * N, B_ldg_guard);
      ldg128_cg<typename HalfType<FT>::T2>(
          BZ[0], BZ[1], BZ[2], BZ[3],
          BZdev_ptr + groupsize_divmod.Div(this_k_idx) * N, B_ldg_guard);
    }

// FMA
#pragma unroll
    for (uint32_t k_frag_idx = 0; k_frag_idx < 2; ++k_frag_idx) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        ldsm_4<uint32_t>(reinterpret_cast<uint32_t*>(FragmentA[i].data)[0],
                         reinterpret_cast<uint32_t*>(FragmentA[i].data)[1],
                         reinterpret_cast<uint32_t*>(FragmentA[i].data)[2],
                         reinterpret_cast<uint32_t*>(FragmentA[i].data)[3],
                         A_lds_addr[k_frag_idx % 2] + i * 64 * 8 * sizeof(FT) +
                             smem_ld_offset);
      }

#pragma unroll
      for (int i = 0; i < 2; ++i) {
        ldsm_4_trans<uint32_t>(
            reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 0].data)[0],
            reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 0].data)[1],
            reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 1].data)[0],
            reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 1].data)[1],
            B_lds_addr[i] + 64 * 64 * k_frag_idx * sizeof(FT) + smem_ld_offset);
      }

#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          hmma_16816<FT, float>(FragmentA[i], FragmentB[j], FragmentC[i][j],
                                FragmentC[i][j]);
        }
      }
    }

    // STS A from reg to smem
    if (warp_id < 4) {
      sts128<int32_t>(ldg_a_reg.x, ldg_a_reg.y, ldg_a_reg.z, ldg_a_reg.w,
                      A_sts_addr + smem_st_offset);
    }

// CVT + DQ
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      cvt_4bx8_to_16bx8_v2(ldg_b_reg[i], sts_b_reg[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sts_b_reg[i][j] = dequantize_func<typename HalfType<FT>::T2>(
            sts_b_reg[i][j], BS[j], BZ[j]);
      }
    }
// STS B from reg to smem
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t sts_b_row = i / 2 * 64 + i % 2 * 8;
      const uint32_t sts_b_offset = sts_b_row * 64;
      sts128<typename HalfType<FT>::T2>(
          sts_b_reg[i][0], sts_b_reg[i][1], sts_b_reg[i][2], sts_b_reg[i][3],
          B_sts_addr + sts_b_offset * sizeof(FT) + smem_st_offset);
    }

    smem_st_offset ^= kSmemSize;
    smem_ld_offset ^= kSmemSize;
    __syncthreads();
  }  // Main-Loop

  // Last Compute
  for (uint32_t k_frag_idx = 0; k_frag_idx < 2; ++k_frag_idx) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldsm_4<uint32_t>(reinterpret_cast<uint32_t*>(FragmentA[i].data)[0],
                       reinterpret_cast<uint32_t*>(FragmentA[i].data)[1],
                       reinterpret_cast<uint32_t*>(FragmentA[i].data)[2],
                       reinterpret_cast<uint32_t*>(FragmentA[i].data)[3],
                       A_lds_addr[k_frag_idx % 2] + i * 64 * 8 * sizeof(FT) +
                           smem_ld_offset);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldsm_4_trans<uint32_t>(
          reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 0].data)[0],
          reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 0].data)[1],
          reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 1].data)[0],
          reinterpret_cast<uint32_t*>(FragmentB[i * 2 + 1].data)[1],
          B_lds_addr[i] + 64 * 64 * k_frag_idx * sizeof(FT) + smem_ld_offset);
    }

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        hmma_16816<FT, float>(FragmentA[i], FragmentB[j], FragmentC[i][j],
                              FragmentC[i][j]);
      }
    }
  }
  hie::Array<FT, 4> FragmentC_ST[2][4];
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t t = 0; t < 4; ++t) {
        FragmentC_ST[i][j][t] = FT(FragmentC[i][j][t]);
      }
    }
  }
  __syncthreads();
  uint32_t* C_sts_smem = reinterpret_cast<uint32_t*>(smem);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t sts_c_row = i * 16 + warp_id / 2 * 32 + lane_id / 4;
      const uint32_t sts_c_col = warp_id % 2 * 16 + j * 4 + lane_id % 4;
      const uint32_t sts_c_offset = sts_c_row * 36 + sts_c_col;
      C_sts_smem[sts_c_offset] =
          reinterpret_cast<uint32_t*>(FragmentC_ST[i][j].data)[0];
      C_sts_smem[sts_c_offset + 8 * 36] =
          reinterpret_cast<uint32_t*>(FragmentC_ST[i][j].data)[1];
    }
  }
  __syncthreads();

  FT* Cdev_ptr = Cdev + blockIdx.z * M * N + blockIdx.x * BLOCK_TILE_M * N +
                 blockIdx.y * BLOCK_TILE_N + lane_id * 8;
  ;
  const uint32_t m_start = blockIdx.x * BLOCK_TILE_M;
  uint32_t* C_lds_smem = reinterpret_cast<uint32_t*>(smem) +
                         lane_id / 8 * 32 * 36 + lane_id % 8 * 4;
  for (uint32_t mi = warp_id; mi < BLOCK_TILE_M; mi += 8) {
    if (mi + m_start < M && B_ldg_guard) {
      reinterpret_cast<int4*>(Cdev_ptr + mi * N)[0] =
          reinterpret_cast<int4*>(C_lds_smem + mi * 36)[0];
    }
  }
}

template <typename FT, typename QT, template <class> class ActiveFunc>
void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, const int GroupSize, void* workspace,
    const SplitKParams splitk_params, cudaStream_t stream) {
  if (GroupSize % 32 != 0) {
    LOG(ERROR)
        << "A16W4 ampere+ subc kernel now only supports GroupSize % 32 = 0"
        << std::endl;
  }

  constexpr uint32_t BLOCK_TILE_M = 32;
  constexpr uint32_t BLOCK_TILE_N = 256;
  uint32_t grid_x = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
  uint32_t grid_y = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
  uint32_t grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  {
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(256);
    U32DivMod groupsize_divmod(GroupSize);
    hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk_kernel<FT, QT>
        <<<grid, block, 0, stream>>>(Adev, Bdev, BSdev, BZdev,
                                     static_cast<FT*>(workspace), M, N, K,
                                     splitk_params.SplitK, groupsize_divmod);
  }
  {
    const uint32_t THREADS_PER_BLOCK = 128;
    const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), M, 1);
    reduce_sum<FT, ActiveFunc><<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<const FT*>(workspace), bias, Cdev, M, N, K,
        DivCeil((int)K, splitk_params.SplitK));
  }
}
//-------------------
//-------------------
template void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Identity>(const half*, const uint8_t*,
                                        const half*, const half*, const half*,
                                        half*, const uint32_t, const uint32_t,
                                        const uint32_t, const int, void*,
                                        const SplitKParams, cudaStream_t);
template void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Gelu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, const int, void*,
                                    const SplitKParams, cudaStream_t);
template void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::GeluTanh>(const half*, const uint8_t*,
                                        const half*, const half*, const half*,
                                        half*, const uint32_t, const uint32_t,
                                        const uint32_t, const int, void*,
                                        const SplitKParams, cudaStream_t);
template void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Relu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, const int, void*,
                                    const SplitKParams, cudaStream_t);
template void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Silu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, const int, void*,
                                    const SplitKParams, cudaStream_t);

template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Identity>(const half*, const uint8_t*,
                                              const half*, const half*,
                                              const half*, half*,
                                              const uint32_t, const uint32_t,
                                              const uint32_t, const int, void*,
                                              const SplitKParams, cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Gelu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, const int, void*,
                                          const SplitKParams, cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::GeluTanh>(const half*, const uint8_t*,
                                              const half*, const half*,
                                              const half*, half*,
                                              const uint32_t, const uint32_t,
                                              const uint32_t, const int, void*,
                                              const SplitKParams, cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Relu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, const int, void*,
                                          const SplitKParams, cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Silu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, const int, void*,
                                          const SplitKParams, cudaStream_t);

template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Identity>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const int, void*, const SplitKParams,
    cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Gelu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const int, void*, const SplitKParams,
    cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::GeluTanh>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const int, void*, const SplitKParams,
    cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Relu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const int, void*, const SplitKParams,
    cudaStream_t);
template void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Silu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, const int, void*, const SplitKParams,
    cudaStream_t);

}  // namespace cuda
}  // namespace allspark