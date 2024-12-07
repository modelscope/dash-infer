/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    volta_gemm_a16w8_subc_kernel.cu
 */

#include "gemm_a16w8_kernel.h"
#include "gemm_lowp_utils.cuh"

namespace allspark {
namespace cuda {

__device__ __forceinline__ int get_octid(int lane_id) {
  return lane_id % 16 / 4;
}

static constexpr int WARP_SIZE = 32;

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * QType is int8 or uint8
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_A16W8_128x128x32_SM70 {
  // element num loaded by a LDG.128 inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 8;
  const int WARP_CNT = BLOCK_SIZE / 32;  // 4

  // For matrix A, a block load/store 128(row) * 32(col) elements,
  // a warp load/store 32(row) * 32(col) elements in 4 LDG.128 inst
  const int WARP_M_SIZE = 128 / WARP_CNT;  // 32
  // For matrix B, a block load/store 32(row) * 128(col) elements,
  // a warp load/store 32(row) * 32(col) elements in 4 LDG.64 inst
  const int WARP_N_SIZE = 128 / WARP_CNT;  // 32

  __device__ GmemTile_A16W8_128x128x32_SM70(
      const SM70_GEMM_A16W8_Params<half, int8_t>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset,
      uint32_t tile_x, uint32_t tile_y)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset),
        blk_x(tile_x),
        blk_y(tile_y) {
    this_block_A_base_ptr = params.A_ptr + blk_y * 128 * params.K;
    this_block_B_base_ptr = params.B_ptr + blk_x * 128;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int load_a_row_base_idx = warp_id * WARP_M_SIZE + lane_id / 4;
    load_a_col_idx = lane_id % 4 * LDG_ELEMENT_CNT_A;
    load_a_base_offset = load_a_row_base_idx * params.K + load_a_col_idx;

    const int load_b_col_idx =
        warp_id * WARP_N_SIZE + lane_id % 4 * LDG_ELEMENT_CNT_B;
    load_b_row_base_idx = lane_id / 4;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_base_offset =
        (warp_id * WARP_M_SIZE + lane_id / 4) * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;
    const int store_b_base_offset =
        warp_id * 32 * WARP_N_SIZE + (lane_id / 4) * WARP_N_SIZE +
        (lane_id % 2) * 2 * LDG_ELEMENT_CNT_B +
        ((lane_id / 2) % 2) * (LDG_ELEMENT_CNT_B / 2);
    store_b_base_offset_0 =
        store_b_base_offset + ((lane_id / 8) % 2) * LDG_ELEMENT_CNT_B;
    store_b_base_offset_1 =
        store_b_base_offset + ((lane_id / 8 + 1) % 2) * LDG_ELEMENT_CNT_B;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      A_ldg_guard[i] = false;
      int m_idx = blk_y * 128 + load_a_row_base_idx + i * WARP_M_SIZE / 4;
      if (m_idx < params.M) {
        A_ldg_guard[i] = true;
      }
    }

    B_ldg_guard = false;
    n_idx = blk_x * 128 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

    B_k_idx = 0;
  }

  __device__ void ldg_first_k_tile(const uint32_t& first_k_tile) {
    const half* this_A_ptr = this_block_A_base_ptr + params.K - first_k_tile;
// For A, a warp load 8(m) * 32(k) chunk per iter, totally load 32(m) * 32(k)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_a_offset = load_a_base_offset + i * WARP_M_SIZE / 4 * params.K;
      bool guard = A_ldg_guard[i] && load_a_col_idx < first_k_tile;
      ldg128_cg_0(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
                  this_A_ptr + load_a_offset, guard);
    }

    // load B scale and zero-point
    const half* this_B_scale_ptr =
        params.B_scale_ptr + (params.GroupCnt - 1) * params.N + n_idx;
    const half* this_B_zero_ptr =
        params.B_zero_ptr + (params.GroupCnt - 1) * params.N + n_idx;
    ldg128_ca_0(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
                b_regs_scale[3], this_B_scale_ptr, B_ldg_guard);
    ldg128_ca_0(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
                this_B_zero_ptr, B_ldg_guard);

    const QType* this_B_ptr =
        this_block_B_base_ptr + (params.K - first_k_tile) * params.N;
// For B, a warp load 8(k) * 32(n) chunk per iter, totally load 32(k) * 32(n)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_b_offset = load_b_base_offset + i * 32 / 4 * params.N;
      bool guard =
          B_ldg_guard && (load_b_row_base_idx + i * 32 / 4) < first_k_tile;
      ldg64_nc_0(b_regs_q[i][0], b_regs_q[i][1], this_B_ptr + load_b_offset,
                 guard);
    }
  }

  __device__ void ldg() {
// For A, a warp load 8(m) * 32(k) chunk per iter, totally load 32(m) * 32(k)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_a_offset = load_a_base_offset + i * WARP_M_SIZE / 4 * params.K;
      bool guard = A_ldg_guard[i];
      ldg128_cg(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
                this_block_A_base_ptr + load_a_offset, guard);
    }

    // load B scale and zero-point
    int B_scale_offset = (B_k_idx >> params.GroupSize) * params.N + n_idx;
    ldg128_ca(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
              b_regs_scale[3], params.B_scale_ptr + B_scale_offset,
              B_ldg_guard);
    ldg128_ca(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
              params.B_zero_ptr + B_scale_offset, B_ldg_guard);

// For B, a warp load 8(k) * 32(n) chunk per iter, totally load 32(k) * 32(n)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // load B
      int load_b_offset = load_b_base_offset + i * 32 / 4 * params.N;
      ldg64_nc(b_regs_q[i][0], b_regs_q[i][1],
               this_block_B_base_ptr + load_b_offset, B_ldg_guard);
    }

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N;

    B_k_idx += 32;
  }

  __device__ void sts(const int buf_idx) {
    uint32_t A_smem_addr = A_smem_base_addr + A_smem_switch_offset * buf_idx;
    uint32_t B_smem_addr = B_smem_base_addr + B_smem_switch_offset * buf_idx;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts128(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
             A_smem_addr + (store_a_base_offset + i * WARP_M_SIZE / 4 * 32) *
                               2 /* ELEM_SIZE */);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b_regs[i][0].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].x), b_regs_zero[0].x),
                 b_regs_scale[0].x);
      b_regs[i][0].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].y), b_regs_zero[0].y),
                 b_regs_scale[0].y);
      b_regs[i][1].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].z), b_regs_zero[1].x),
                 b_regs_scale[1].x);
      b_regs[i][1].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].w), b_regs_zero[1].y),
                 b_regs_scale[1].y);
      sts64(b_regs[i][0], b_regs[i][1],
            B_smem_addr + (store_b_base_offset_0 + i * 32 / 4 * WARP_N_SIZE) *
                              2 /* ELEM_SIZE */);

      b_regs[i][2].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].x), b_regs_zero[2].x),
                 b_regs_scale[2].x);
      b_regs[i][2].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].y), b_regs_zero[2].y),
                 b_regs_scale[2].y);
      b_regs[i][3].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].z), b_regs_zero[3].x),
                 b_regs_scale[3].x);
      b_regs[i][3].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].w), b_regs_zero[3].y),
                 b_regs_scale[3].y);
      sts64(b_regs[i][2], b_regs[i][3],
            B_smem_addr + (store_b_base_offset_1 + i * 32 / 4 * WARP_N_SIZE) *
                              2 /* ELEM_SIZE */);
    }
  }

  const half* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;
  int load_a_base_offset;
  int load_b_base_offset;
  int store_a_base_offset;
  int store_b_base_offset_0;
  int store_b_base_offset_1;

  int load_a_col_idx;
  int load_b_row_base_idx;

  int n_idx;
  int B_k_idx;

  bool A_ldg_guard[4]{};
  bool B_ldg_guard{};

  uint32_t a_regs[4][4];
  __half2 b_regs[4][4];
  char4 b_regs_q[4][2];
  __half2 b_regs_scale[4];
  __half2 b_regs_zero[4];

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;
  const uint32_t blk_x, blk_y;
  const SM70_GEMM_A16W8_Params<half, QType>& params;
};

/*
 * warp_tile : m64n64k32
 */
template <typename QType>
struct ComputeTile_f16_128x128x32_SM70 {
  const int N_PART_SIZE = 32;

  __device__ ComputeTile_f16_128x128x32_SM70(
      const SM70_GEMM_A16W8_Params<half, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset,
      const uint32_t& tile_x, const uint32_t& tile_y)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset),
        blk_x(tile_x),
        blk_y(tile_y) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int oct_id = get_octid(lane_id);

    int row_idx_in_oct = lane_id / 16 * 4 + lane_id % 4;
    A_kphase_col_adjust = (row_idx_in_oct / 4) ^ (row_idx_in_oct % 4);
    load_a_base_offset =
        (oct_id / 2 * 8 + row_idx_in_oct) * 32 + warp_id / 2 * 128 / 2 * 32;
    load_b_base_offset = (oct_id % 2) * N_PART_SIZE / 2 +
                         ((lane_id / 16) ^ ((lane_id % 4) / 2)) * 8 +
                         (lane_id % 4) * N_PART_SIZE +
                         (warp_id % 2) * 128 / 2 * 32;
    store_c_row_base_idx =
        oct_id / 2 * 8 + lane_id / 16 * 4 + lane_id % 4 + warp_id / 2 * 64;
    store_c_col_base_idx = oct_id % 2 * 8 + warp_id % 2 * 64;
    ;
    store_c_base_offset =
        store_c_row_base_idx * params.N + store_c_col_base_idx;

    __half2 zeros{};

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          reinterpret_cast<__half2&>(C_frag[i][j][k]) = zeros;
        }
      }
    }

    this_block_C_base_ptr = params.C_ptr + blk_y * 128 * params.N + blk_x * 128;
  }

  // load 64 * 8 A elements per warp per k_phase
  __device__ void lds_A(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_switch_offset * smem_buf_idx;

    int load_a_offset =
        load_a_base_offset + (k_phase_idx ^ A_kphase_col_adjust) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      lds128(A_frag[reg_buf_idx][i][0], A_frag[reg_buf_idx][i][1],
             A_frag[reg_buf_idx][i][2], A_frag[reg_buf_idx][i][3],
             A_smem_addr + (load_a_offset + i * 16 * 32) * 2 /* ELEM SIZE */);
    }
  }

  // load 8 * 64 B elements per warp per k_phase
  __device__ void lds_B(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t B_smem_addr =
        B_smem_base_addr + B_smem_switch_offset * smem_buf_idx;

    int load_b_offset = load_b_base_offset + k_phase_idx * 8 * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds128(B_frag[reg_buf_idx][i][0], B_frag[reg_buf_idx][i][1],
             B_frag[reg_buf_idx][i][2], B_frag[reg_buf_idx][i][3],
             B_smem_addr +
                 (load_b_offset + i * 4 * N_PART_SIZE) * 2 /* ELEM SIZE */);
      lds128(B_frag[reg_buf_idx][i][4], B_frag[reg_buf_idx][i][5],
             B_frag[reg_buf_idx][i][6], B_frag[reg_buf_idx][i][7],
             B_smem_addr +
                 (load_b_offset + i * 4 * N_PART_SIZE + 32 * N_PART_SIZE) *
                     2 /* ELEM SIZE */);
    }
  }

  __device__ void mma(const int reg_buf_idx) {
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 4; ++n_idx) {
        mma_h884(A_frag[reg_buf_idx][m_idx][0], A_frag[reg_buf_idx][m_idx][1],
                 B_frag[reg_buf_idx][0][n_idx * 2],
                 B_frag[reg_buf_idx][0][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
        mma_h884(A_frag[reg_buf_idx][m_idx][2], A_frag[reg_buf_idx][m_idx][3],
                 B_frag[reg_buf_idx][1][n_idx * 2],
                 B_frag[reg_buf_idx][1][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
      }
    }
  }

  __device__ void stg() {
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 4; st_iter_m++) {
      for (int st_iter_n = 0; st_iter_n < 4; st_iter_n++) {
        half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                      st_iter_m * 16 * params.N + st_iter_n * 16;
        bool guard =
            (blk_y * 128 + store_c_row_base_idx + st_iter_m * 16) < params.M &&
            (blk_x * 128 + store_c_col_base_idx + st_iter_n * 16) < params.N;
        stg128(C_frag[st_iter_m][st_iter_n][0], C_frag[st_iter_m][st_iter_n][1],
               C_frag[st_iter_m][st_iter_n][2], C_frag[st_iter_m][st_iter_n][3],
               C_ptr, guard);
      }
    }
  }

  const SM70_GEMM_A16W8_Params<half, QType>& params;

  int load_a_base_offset;
  int load_b_base_offset;
  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_base_idx;
  half* this_block_C_base_ptr = nullptr;

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;
  const uint32_t blk_x, blk_y;

  // 2 denotes double buffer, first 4 denotes M direction, second 4 denotes K
  // direction
  uint32_t A_frag[2][4][4];
  // 2 denotes double buffer, first 2 denotes K direction, second 8 denotes N
  // direction
  uint32_t B_frag[2][2][8];
  // first 4 denotes M direction, second 4 denotes N direction
  uint32_t C_frag[4][4][4];

  int A_kphase_col_adjust;
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: K x N, matrix C: M x N (row-major)
 *  K % 8 == 0 && N % 8 == 0
 *  accumulator precision: FP16
 *  output datatype: FP16
 *
 *  BLOCK_TILE: m128n128k32
 *  BLOCK_SIZE: 128
 *  WARP_TILE:  m64n64k32
 */
template <typename QType>
__global__ void __launch_bounds__(128, 2)
    volta_hgemm_A16W8_f16_f16_128x128x32_mma884_kernel(
        const SM70_GEMM_A16W8_Params<half, QType> params,
        uint32_t full_wave_blocks, uint32_t wave_y, U32DivMod wave_size_divmod,
        U32DivMod wave_y_divmod, U32DivMod last_wave_y_divmod) {
  // A smem size = 128 * 32 * 2B/elem * 2(double buffer) = 16KB
  // B smem size = 128 * 32 * 2B/elem * 2(double biffer) = 16KB
  static constexpr int SMEM_SIZE = 128 * 32 * 2 * 2 + 128 * 32 * 2 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + 128 * 32 * 2 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);
  uint32_t A_smem_switch_offset = 128 * 32 * 2;
  uint32_t B_smem_switch_offset = 128 * 32 * 2;

  // remap thread block to matrix_C tile to increase L2 cache hit rate
  auto wave_size_dm = wave_size_divmod.DivMod(blockIdx.x);
  auto wave_y_dm = blockIdx.x < full_wave_blocks
                       ? wave_y_divmod.DivMod(wave_size_dm.mod)
                       : last_wave_y_divmod.DivMod(wave_size_dm.mod);
  uint32_t tile_x = wave_y_dm.div;
  uint32_t tile_y = wave_size_dm.div * wave_y + wave_y_dm.mod;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_A16W8_128x128x32_SM70<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset, tile_x, tile_y);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  int k_main_loop = (params.K + 31) / 32 - 1;
  uint32_t first_k_tile = params.K - k_main_loop * 32;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile);
  ComputeTile_f16_128x128x32_SM70<QType> compute_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset, tile_x, tile_y);
  gmem_tile.sts(write_smem_buf_idx);

  __syncthreads();
  compute_tile.lds_A(read_smem_buf_idx, 0, 0);
  compute_tile.lds_B(read_smem_buf_idx, 0, 0);
  int reg_buf_idx = 1;

  for (int k_tile_idx = 0; k_tile_idx < k_main_loop; k_tile_idx++) {
    write_smem_buf_idx ^= 1;
    gmem_tile.ldg();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
      if (k_phase_idx == 3) {
        gmem_tile.sts(write_smem_buf_idx);
        read_smem_buf_idx ^= 1;
        __syncthreads();
      }

      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
    if (k_phase_idx < 3) {
      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
    }
    compute_tile.mma(reg_buf_idx ^ 1);
    reg_buf_idx ^= 1;
  }

  compute_tile.stg();
}

int log2(int x) {
  int ret = 0;
  while (x != 1) {
    x /= 2;
    ret++;
  }
  return ret;
}

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_128x128x32_mma884(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count, const float alpha,
    cudaStream_t stream) {
  // launch GEMM kernel
  {
    const int GroupCnt = (K + GroupSize - 1) / GroupSize;
    SM70_GEMM_A16W8_Params<half, QType> params{
        A, B, B_scale, B_zero, C, M, N, K, GroupCnt, log2(GroupSize)};
    uint32_t grid_x = (N + 127) / 128;
    uint32_t grid_y = (M + 127) / 128;
    uint32_t grid = grid_x * grid_y;

    // only 2 thread block per sm, limited by register and shared memory
    uint32_t blocks_per_wave = sm_count * 2;

    // wave_x * wave_x == blocks_per_wave
    uint32_t wave_y = static_cast<uint32_t>(sqrt(float(blocks_per_wave)));
    if (wave_y > grid_y) {
      wave_y = grid_y;
    }

    // if last_wave_y is 0, set it to 1 to avoid U32DivMod exception
    uint32_t last_wave_y = grid_y % wave_y == 0 ? 1 : grid_y % wave_y;

    uint32_t wave_size = wave_y * grid_x;
    uint32_t full_wave_blocks = grid - grid % wave_size;

    U32DivMod wave_size_divmod(wave_size);
    U32DivMod wave_y_divmod(wave_y);
    U32DivMod last_wave_y_divmod(last_wave_y);

    cudaFuncSetAttribute(
        volta_hgemm_A16W8_f16_f16_128x128x32_mma884_kernel<QType>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    volta_hgemm_A16W8_f16_f16_128x128x32_mma884_kernel<QType>
        <<<grid, 128, 0, stream>>>(params, full_wave_blocks, wave_y,
                                   wave_size_divmod, wave_y_divmod,
                                   last_wave_y_divmod);
  }
  // add bias and active func
  {
    const uint32_t BLOCK_SIZE = 128;
    const uint32_t UNROLL = 4;
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    add_bias<UNROLL, ActiveFunc>
        <<<grid, BLOCK_SIZE, 0, stream>>>(bias, C, M, N, alpha);
  }
}

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * QType is int8 or uint8
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_A16W8_128x128x32_SM70_SplitK {
  // element num loaded by a LDG.128 inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 8;
  const int WARP_CNT = BLOCK_SIZE / 32;  // 4

  // For matrix A, a block load/store 128(row) * 32(col) elements,
  // a warp load/store 32(row) * 32(col) elements in 4 LDG.128 inst
  const int WARP_M_SIZE = 128 / WARP_CNT;  // 32
  // For matrix B, a block load/store 32(row) * 128(col) elements,
  // a warp load/store 32(row) * 32(col) elements in 4 LDG.64 inst
  const int WARP_N_SIZE = 128 / WARP_CNT;  // 32

  __device__ GmemTile_A16W8_128x128x32_SM70_SplitK(
      const SM70_GEMM_A16W8_Params<half, int8_t>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    tile_x = params.schedule_mn == TileSchedule::N_BLK_CONTINUOUS ? blockIdx.x
                                                                  : blockIdx.y;
    tile_y = params.schedule_mn == TileSchedule::N_BLK_CONTINUOUS ? blockIdx.y
                                                                  : blockIdx.x;

    this_block_A_base_ptr =
        params.A_ptr + tile_y * 128 * params.K + blockIdx.z * params.SplitK;
    ;
    this_block_B_base_ptr =
        params.B_ptr + tile_x * 128 + blockIdx.z * params.SplitK * params.N;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int load_a_row_base_idx = warp_id * WARP_M_SIZE + lane_id / 4;
    load_a_col_idx = lane_id % 4 * LDG_ELEMENT_CNT_A;
    load_a_base_offset = load_a_row_base_idx * params.K + load_a_col_idx;

    const int load_b_col_idx =
        warp_id * WARP_N_SIZE + lane_id % 4 * LDG_ELEMENT_CNT_B;
    load_b_row_base_idx = lane_id / 4;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_base_offset =
        (warp_id * WARP_M_SIZE + lane_id / 4) * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;
    const int store_b_base_offset =
        warp_id * 32 * WARP_N_SIZE + (lane_id / 4) * WARP_N_SIZE +
        (lane_id % 2) * 2 * LDG_ELEMENT_CNT_B +
        ((lane_id / 2) % 2) * (LDG_ELEMENT_CNT_B / 2);
    store_b_base_offset_0 =
        store_b_base_offset + ((lane_id / 8) % 2) * LDG_ELEMENT_CNT_B;
    store_b_base_offset_1 =
        store_b_base_offset + ((lane_id / 8 + 1) % 2) * LDG_ELEMENT_CNT_B;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      A_ldg_guard[i] = false;
      int m_idx = tile_y * 128 + load_a_row_base_idx + i * WARP_M_SIZE / 4;
      if (m_idx < params.M) {
        A_ldg_guard[i] = true;
      }
    }

    B_ldg_guard = false;
    n_idx = tile_x * 128 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

    B_k_idx = blockIdx.z * params.SplitK;
  }

  __device__ void ldg_first_k_tile(const uint32_t& first_k_tile,
                                   const uint32_t tb_k_slice) {
    const half* this_A_ptr = this_block_A_base_ptr + tb_k_slice - first_k_tile;
// For A, a warp load 8(m) * 32(k) chunk per iter, totally load 32(m) * 32(k)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_a_offset = load_a_base_offset + i * WARP_M_SIZE / 4 * params.K;
      bool guard = A_ldg_guard[i] && load_a_col_idx < first_k_tile;
      ldg128_cg_0(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
                  this_A_ptr + load_a_offset, guard);
    }

    // load B scale and zero-point
    uint32_t k_idx = B_k_idx + tb_k_slice - first_k_tile;
    const half* this_B_scale_ptr =
        params.B_scale_ptr + (k_idx >> params.GroupSize) * params.N + n_idx;
    const half* this_B_zero_ptr =
        params.B_zero_ptr + (k_idx >> params.GroupSize) * params.N + n_idx;
    ldg128_ca_0(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
                b_regs_scale[3], this_B_scale_ptr, B_ldg_guard);
    ldg128_ca_0(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
                this_B_zero_ptr, B_ldg_guard);

    const QType* this_B_ptr =
        this_block_B_base_ptr + (tb_k_slice - first_k_tile) * params.N;
// For B, a warp load 8(k) * 32(n) chunk per iter, totally load 32(k) * 32(n)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_b_offset = load_b_base_offset + i * 32 / 4 * params.N;
      bool guard =
          B_ldg_guard && (load_b_row_base_idx + i * 32 / 4) < first_k_tile;
      ldg64_nc_0(b_regs_q[i][0], b_regs_q[i][1], this_B_ptr + load_b_offset,
                 guard);
    }
  }

  __device__ void ldg() {
// For A, a warp load 8(m) * 32(k) chunk per iter, totally load 32(m) * 32(k)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int load_a_offset = load_a_base_offset + i * WARP_M_SIZE / 4 * params.K;
      bool guard = A_ldg_guard[i];
      ldg128_cg(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
                this_block_A_base_ptr + load_a_offset, guard);
    }

    // load B scale and zero-point
    int B_scale_offset = (B_k_idx >> params.GroupSize) * params.N + n_idx;
    ldg128_ca(b_regs_scale[0], b_regs_scale[1], b_regs_scale[2],
              b_regs_scale[3], params.B_scale_ptr + B_scale_offset,
              B_ldg_guard);
    ldg128_ca(b_regs_zero[0], b_regs_zero[1], b_regs_zero[2], b_regs_zero[3],
              params.B_zero_ptr + B_scale_offset, B_ldg_guard);

// For B, a warp load 8(k) * 32(n) chunk per iter, totally load 32(k) * 32(n)
// chunk in 4 iter.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // load B
      int load_b_offset = load_b_base_offset + i * 32 / 4 * params.N;
      ldg64_nc(b_regs_q[i][0], b_regs_q[i][1],
               this_block_B_base_ptr + load_b_offset, B_ldg_guard);
    }

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N;

    B_k_idx += 32;
  }

  __device__ void sts(const int buf_idx) {
    uint32_t A_smem_addr = A_smem_base_addr + A_smem_switch_offset * buf_idx;
    uint32_t B_smem_addr = B_smem_base_addr + B_smem_switch_offset * buf_idx;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts128(a_regs[i][0], a_regs[i][1], a_regs[i][2], a_regs[i][3],
             A_smem_addr + (store_a_base_offset + i * WARP_M_SIZE / 4 * 32) *
                               2 /* ELEM_SIZE */);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b_regs[i][0].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].x), b_regs_zero[0].x),
                 b_regs_scale[0].x);
      b_regs[i][0].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].y), b_regs_zero[0].y),
                 b_regs_scale[0].y);
      b_regs[i][1].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].z), b_regs_zero[1].x),
                 b_regs_scale[1].x);
      b_regs[i][1].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][0].w), b_regs_zero[1].y),
                 b_regs_scale[1].y);
      sts64(b_regs[i][0], b_regs[i][1],
            B_smem_addr + (store_b_base_offset_0 + i * 32 / 4 * WARP_N_SIZE) *
                              2 /* ELEM_SIZE */);

      b_regs[i][2].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].x), b_regs_zero[2].x),
                 b_regs_scale[2].x);
      b_regs[i][2].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].y), b_regs_zero[2].y),
                 b_regs_scale[2].y);
      b_regs[i][3].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].z), b_regs_zero[3].x),
                 b_regs_scale[3].x);
      b_regs[i][3].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i][1].w), b_regs_zero[3].y),
                 b_regs_scale[3].y);
      sts64(b_regs[i][2], b_regs[i][3],
            B_smem_addr + (store_b_base_offset_1 + i * 32 / 4 * WARP_N_SIZE) *
                              2 /* ELEM_SIZE */);
    }
  }

  const half* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;
  int load_a_base_offset;
  int load_b_base_offset;
  int store_a_base_offset;
  int store_b_base_offset_0;
  int store_b_base_offset_1;

  int load_a_col_idx;
  int load_b_row_base_idx;

  int n_idx;
  int B_k_idx;

  bool A_ldg_guard[4]{};
  bool B_ldg_guard{};

  uint32_t a_regs[4][4];
  __half2 b_regs[4][4];
  char4 b_regs_q[4][2];
  __half2 b_regs_scale[4];
  __half2 b_regs_zero[4];

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;
  uint32_t tile_x, tile_y;

  const SM70_GEMM_A16W8_Params<half, QType>& params;
};

/*
 * warp_tile : m64n64k32
 */
template <typename QType>
struct ComputeTile_f16_128x128x32_SM70_SplitK {
  const int N_PART_SIZE = 32;

  __device__ ComputeTile_f16_128x128x32_SM70_SplitK(
      const SM70_GEMM_A16W8_Params<half, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int oct_id = get_octid(lane_id);

    int row_idx_in_oct = lane_id / 16 * 4 + lane_id % 4;
    A_kphase_col_adjust = (row_idx_in_oct / 4) ^ (row_idx_in_oct % 4);
    load_a_base_offset =
        (oct_id / 2 * 8 + row_idx_in_oct) * 32 + warp_id / 2 * 128 / 2 * 32;
    load_b_base_offset = (oct_id % 2) * N_PART_SIZE / 2 +
                         ((lane_id / 16) ^ ((lane_id % 4) / 2)) * 8 +
                         (lane_id % 4) * N_PART_SIZE +
                         (warp_id % 2) * 128 / 2 * 32;
    store_c_row_base_idx =
        oct_id / 2 * 8 + lane_id / 16 * 4 + lane_id % 4 + warp_id / 2 * 64;
    store_c_col_base_idx = oct_id % 2 * 8 + warp_id % 2 * 64;
    ;
    store_c_base_offset =
        store_c_row_base_idx * params.N + store_c_col_base_idx;

    __half2 zeros{};

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          reinterpret_cast<__half2&>(C_frag[i][j][k]) = zeros;
        }
      }
    }

    tile_x = params.schedule_mn == TileSchedule::N_BLK_CONTINUOUS ? blockIdx.x
                                                                  : blockIdx.y;
    tile_y = params.schedule_mn == TileSchedule::N_BLK_CONTINUOUS ? blockIdx.y
                                                                  : blockIdx.x;
    this_block_C_base_ptr = params.C_split_ptr +
                            blockIdx.z * params.M * params.N +
                            tile_y * 128 * params.N + tile_x * 128;
  }

  // load 64 * 8 A elements per warp per k_phase
  __device__ void lds_A(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_switch_offset * smem_buf_idx;

    int load_a_offset =
        load_a_base_offset + (k_phase_idx ^ A_kphase_col_adjust) * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      lds128(A_frag[reg_buf_idx][i][0], A_frag[reg_buf_idx][i][1],
             A_frag[reg_buf_idx][i][2], A_frag[reg_buf_idx][i][3],
             A_smem_addr + (load_a_offset + i * 16 * 32) * 2 /* ELEM SIZE */);
    }
  }

  // load 8 * 64 B elements per warp per k_phase
  __device__ void lds_B(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t B_smem_addr =
        B_smem_base_addr + B_smem_switch_offset * smem_buf_idx;

    int load_b_offset = load_b_base_offset + k_phase_idx * 8 * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds128(B_frag[reg_buf_idx][i][0], B_frag[reg_buf_idx][i][1],
             B_frag[reg_buf_idx][i][2], B_frag[reg_buf_idx][i][3],
             B_smem_addr +
                 (load_b_offset + i * 4 * N_PART_SIZE) * 2 /* ELEM SIZE */);
      lds128(B_frag[reg_buf_idx][i][4], B_frag[reg_buf_idx][i][5],
             B_frag[reg_buf_idx][i][6], B_frag[reg_buf_idx][i][7],
             B_smem_addr +
                 (load_b_offset + i * 4 * N_PART_SIZE + 32 * N_PART_SIZE) *
                     2 /* ELEM SIZE */);
    }
  }

  __device__ void mma(const int reg_buf_idx) {
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 4; ++n_idx) {
        mma_h884(A_frag[reg_buf_idx][m_idx][0], A_frag[reg_buf_idx][m_idx][1],
                 B_frag[reg_buf_idx][0][n_idx * 2],
                 B_frag[reg_buf_idx][0][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
        mma_h884(A_frag[reg_buf_idx][m_idx][2], A_frag[reg_buf_idx][m_idx][3],
                 B_frag[reg_buf_idx][1][n_idx * 2],
                 B_frag[reg_buf_idx][1][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
      }
    }
  }

  __device__ void stg() {
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 4; st_iter_m++) {
      for (int st_iter_n = 0; st_iter_n < 4; st_iter_n++) {
        half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                      st_iter_m * 16 * params.N + st_iter_n * 16;
        bool guard =
            (tile_y * 128 + store_c_row_base_idx + st_iter_m * 16) < params.M &&
            (tile_x * 128 + store_c_col_base_idx + st_iter_n * 16) < params.N;
        stg128(C_frag[st_iter_m][st_iter_n][0], C_frag[st_iter_m][st_iter_n][1],
               C_frag[st_iter_m][st_iter_n][2], C_frag[st_iter_m][st_iter_n][3],
               C_ptr, guard);
      }
    }
  }

  const SM70_GEMM_A16W8_Params<half, QType>& params;

  int load_a_base_offset;
  int load_b_base_offset;
  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_base_idx;
  half* this_block_C_base_ptr = nullptr;

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;
  uint32_t tile_x, tile_y;

  // 2 denotes double buffer, first 4 denotes M direction, second 4 denotes K
  // direction
  uint32_t A_frag[2][4][4];
  // 2 denotes double buffer, first 2 denotes K direction, second 8 denotes N
  // direction
  uint32_t B_frag[2][2][8];
  // first 4 denotes M direction, second 4 denotes N direction
  uint32_t C_frag[4][4][4];

  int A_kphase_col_adjust;
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: K x N, matrix C: M x N (row-major)
 *  K % 8 == 0 && N % 8 == 0
 *  accumulator precision: FP16
 *  output datatype: FP16
 *
 *  BLOCK_TILE: m128n128k32
 *  BLOCK_SIZE: 128
 *  WARP_TILE:  m64n64k32
 */
template <typename QType>
__global__ void __launch_bounds__(128, 2)
    volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk_kernel(
        const SM70_GEMM_A16W8_Params<half, QType> params) {
  // A smem size = 128 * 32 * 2B/elem * 2(double buffer) = 16KB
  // B smem size = 128 * 32 * 2B/elem * 2(double biffer) = 16KB
  static constexpr int SMEM_SIZE = 128 * 32 * 2 * 2 + 128 * 32 * 2 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + 128 * 32 * 2 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);
  uint32_t A_smem_switch_offset = 128 * 32 * 2;
  uint32_t B_smem_switch_offset = 128 * 32 * 2;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_A16W8_128x128x32_SM70_SplitK<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  uint32_t tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                            ? params.SplitK
                            : params.K - blockIdx.z * params.SplitK;
  int k_main_loop = (tb_k_slice + 31) / 32 - 1;
  uint32_t first_k_tile = tb_k_slice - k_main_loop * 32;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_f16_128x128x32_SM70_SplitK<QType> compute_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);
  gmem_tile.sts(write_smem_buf_idx);

  __syncthreads();
  compute_tile.lds_A(read_smem_buf_idx, 0, 0);
  compute_tile.lds_B(read_smem_buf_idx, 0, 0);
  int reg_buf_idx = 1;

  for (int k_tile_idx = 0; k_tile_idx < k_main_loop; k_tile_idx++) {
    write_smem_buf_idx ^= 1;
    gmem_tile.ldg();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
      if (k_phase_idx == 3) {
        gmem_tile.sts(write_smem_buf_idx);
        read_smem_buf_idx ^= 1;
        __syncthreads();
      }

      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
    if (k_phase_idx < 3) {
      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
    }
    compute_tile.mma(reg_buf_idx ^ 1);
    reg_buf_idx ^= 1;
  }

  compute_tile.stg();
}

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count,
    const SplitKParams splitk_params, const float alpha, cudaStream_t stream) {
  half* C_split = static_cast<half*>(workspace);

  int grid_x = M < N ? (M + 127) / 128 : (N + 127) / 128;
  int grid_y = M < N ? (N + 127) / 128 : (M + 127) / 128;
  TileSchedule schedule_mn =
      M < N ? TileSchedule::M_BLK_CONTINUOUS : TileSchedule::N_BLK_CONTINUOUS;
  int grid_z = DivCeil(K, splitk_params.SplitK);

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(128);

  SM70_GEMM_A16W8_Params<half, QType> params{
      A,          B, B_scale, B_zero,          C,       M,
      N,          K, 0,       log2(GroupSize), C_split, splitk_params.SplitK,
      schedule_mn};

  cudaFuncSetAttribute(
      volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk_kernel<QType>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk_kernel<QType>
      <<<grid, block, 0, stream>>>(params);

  // SplitK reduce
  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                           grid_z, alpha, stream);
}

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * QType is int8 or uint8
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_A16W8_32x128x32_SM70_SplitK {
  // element num loaded by a LDG inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 4;
  const int WARP_CNT = BLOCK_SIZE / 32;  // 4

  __device__ GmemTile_A16W8_32x128x32_SM70_SplitK(
      const SM70_GEMM_A16W8_Params<half, int8_t>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    this_block_A_base_ptr =
        params.A_ptr + blockIdx.x * 32 * params.K + blockIdx.z * params.SplitK;
    ;
    this_block_B_base_ptr =
        params.B_ptr + blockIdx.y * 128 + blockIdx.z * params.SplitK * params.N;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // For matrix A, a block load/store 32(row) x 32(col) elements in 1 iter
    // 8x4 warp load/store 8(row) x 32(col) elements
    const int load_a_row_idx = threadIdx.x / 4;
    load_a_col_idx = threadIdx.x % 4 * LDG_ELEMENT_CNT_A;
    load_a_offset = load_a_row_idx * params.K + load_a_col_idx;

    // For matrix B, a block load/store 32(row) * 128(col) elements in 8 iter
    // and a block load/store 4(row) * 128(col) elements per iter, a warp
    // load/store 1(row) * 128(col) per iter
    const int load_b_col_idx = lane_id * LDG_ELEMENT_CNT_B;
    load_b_row_base_idx = warp_id;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_offset =
        threadIdx.x / 4 * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;

    store_b_base_offset =
        warp_id * 128 + (lane_id ^ (warp_id * 4)) * LDG_ELEMENT_CNT_B;

    A_ldg_guard = false;
    int m_idx = blockIdx.x * 32 + load_a_row_idx;
    if (m_idx < params.M) {
      A_ldg_guard = true;
    }

    B_ldg_guard = false;
    n_idx = blockIdx.y * 128 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

    B_k_idx = blockIdx.z * params.SplitK;
  }

  __device__ void ldg_first_k_tile(const int& first_k_tile,
                                   const int tb_k_slice) {
    // load B
    const QType* this_B_ptr =
        this_block_B_base_ptr + (tb_k_slice - first_k_tile) * params.N;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * 4 * params.N;
      bool guard = B_ldg_guard && (load_b_row_base_idx + i * 4) < first_k_tile;
      ldg32_cs_0(b_regs_q[i], this_B_ptr + load_b_offset, guard);
    }

    // load B scale and zero-point
    int k_idx = B_k_idx + tb_k_slice - first_k_tile;
    const half* this_B_scale_ptr =
        params.B_scale_ptr + (k_idx >> params.GroupSize) * params.N + n_idx;
    const half* this_B_zero_ptr =
        params.B_zero_ptr + (k_idx >> params.GroupSize) * params.N + n_idx;
    ldg64_ca_0(b_regs_scale[0], b_regs_scale[1], this_B_scale_ptr, B_ldg_guard);
    ldg64_ca_0(b_regs_zero[0], b_regs_zero[1], this_B_zero_ptr, B_ldg_guard);

    // load A
    const half* this_A_ptr = this_block_A_base_ptr + tb_k_slice - first_k_tile;
    bool guard = A_ldg_guard && load_a_col_idx < first_k_tile;
    ldg128_cg_0(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                this_A_ptr + load_a_offset, guard);
  }

  __device__ void ldg() {
// load B
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * 4 * params.N;
      ldg32_cs(b_regs_q[i], this_block_B_base_ptr + load_b_offset, B_ldg_guard);
    }

    // load B scale and zero-point
    int B_scale_offset = (B_k_idx >> params.GroupSize) * params.N + n_idx;
    ldg64_ca(b_regs_scale[0], b_regs_scale[1],
             params.B_scale_ptr + B_scale_offset, B_ldg_guard);
    ldg64_ca(b_regs_zero[0], b_regs_zero[1], params.B_zero_ptr + B_scale_offset,
             B_ldg_guard);

    // load A
    ldg128_cg(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
              this_block_A_base_ptr + load_a_offset, A_ldg_guard);

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N;

    B_k_idx += 32;
  }

  __device__ void sts(const int buf_idx) {
    uint32_t A_smem_addr = A_smem_base_addr + A_smem_switch_offset * buf_idx;
    uint32_t B_smem_addr = B_smem_base_addr + B_smem_switch_offset * buf_idx;

    // store A from reg to smem
    sts128(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
           A_smem_addr + store_a_offset * 2 /* ELEM_SIZE */);

// dequant and store B from reg to smem
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      b_regs[i][0].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i].x), b_regs_zero[0].x),
                 b_regs_scale[0].x);
      b_regs[i][0].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i].y), b_regs_zero[0].y),
                 b_regs_scale[0].y);
      b_regs[i][1].x =
          __hmul(__hsub(static_cast<half>(b_regs_q[i].z), b_regs_zero[1].x),
                 b_regs_scale[1].x);
      b_regs[i][1].y =
          __hmul(__hsub(static_cast<half>(b_regs_q[i].w), b_regs_zero[1].y),
                 b_regs_scale[1].y);
      sts64(b_regs[i][0], b_regs[i][1],
            B_smem_addr +
                (store_b_base_offset + i * 4 * 128) * 2 /* ELEM_SIZE */);
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

  bool A_ldg_guard;
  bool B_ldg_guard;

  uint32_t a_regs[4];
  __half2 b_regs[8][2];
  char4 b_regs_q[8];
  __half2 b_regs_scale[2];
  __half2 b_regs_zero[2];

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;

  const SM70_GEMM_A16W8_Params<half, QType>& params;
};

/*
 * warp_tile : m32n32k32
 */
template <typename QType>
struct ComputeTile_f16_32x128x32_SM70_SplitK {
  __device__ ComputeTile_f16_32x128x32_SM70_SplitK(
      const SM70_GEMM_A16W8_Params<half, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& B_smem_addr,
      const uint32_t& A_switch_offset, const uint32_t& B_switch_offset)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        B_smem_base_addr(B_smem_addr),
        A_smem_switch_offset(A_switch_offset),
        B_smem_switch_offset(B_switch_offset) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int oct_id = get_octid(lane_id);

    int row_idx_in_oct = lane_id / 16 * 4 + lane_id % 4;
    A_kphase_col_adjust = (row_idx_in_oct / 4) ^ (row_idx_in_oct % 4);
    load_a_base_offset = (oct_id / 2 * 8 + row_idx_in_oct) * 32;

    load_b_base_offset_0 = (lane_id % 4) * 128 +
                           ((warp_id * 8 + (oct_id % 2) * 2 + (lane_id / 16)) ^
                            ((lane_id % 4) * 4)) *
                               4;
    load_b_base_offset_1 =
        (lane_id % 4) * 128 +
        ((warp_id * 8 + (oct_id % 2) * 2 + (lane_id / 16) + 4) ^
         ((lane_id % 4) * 4)) *
            4;
    store_c_row_base_idx = oct_id / 2 * 8 + lane_id / 16 * 4 + lane_id % 4;
    store_c_col_base_idx = oct_id % 2 * 8 + warp_id * 32;
    store_c_base_offset =
        store_c_row_base_idx * params.N + store_c_col_base_idx;

    this_block_C_base_ptr = params.C_split_ptr +
                            blockIdx.z * params.M * params.N +
                            blockIdx.x * 32 * params.N + blockIdx.y * 128;
  }

  // load 32 * 8 A elements per warp per k_phase
  __device__ void lds_A(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_switch_offset * smem_buf_idx;

    int load_a_offset =
        load_a_base_offset + (k_phase_idx ^ A_kphase_col_adjust) * 8;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds128(A_frag[reg_buf_idx][i][0], A_frag[reg_buf_idx][i][1],
             A_frag[reg_buf_idx][i][2], A_frag[reg_buf_idx][i][3],
             A_smem_addr + (load_a_offset + i * 16 * 32) * 2 /* ELEM SIZE */);
    }
  }

  // load 8 * 32 B elements per warp per k_phase
  __device__ void lds_B(const int smem_buf_idx, const int reg_buf_idx,
                        const int k_phase_idx) {
    uint32_t B_smem_addr =
        B_smem_base_addr + B_smem_switch_offset * smem_buf_idx;

    int k_phase_offset = k_phase_idx * 8 * 128;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds64(
          B_frag[reg_buf_idx][i][0], B_frag[reg_buf_idx][i][1],
          B_smem_addr + (load_b_base_offset_0 + k_phase_offset + i * 4 * 128) *
                            2 /* ELEM SIZE */);
      lds64(
          B_frag[reg_buf_idx][i][2], B_frag[reg_buf_idx][i][3],
          B_smem_addr + (load_b_base_offset_1 + k_phase_offset + i * 4 * 128) *
                            2 /* ELEM SIZE */);
    }
  }

  __device__ void mma(const int reg_buf_idx) {
#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 2; ++n_idx) {
        mma_h884(A_frag[reg_buf_idx][m_idx][0], A_frag[reg_buf_idx][m_idx][1],
                 B_frag[reg_buf_idx][0][n_idx * 2],
                 B_frag[reg_buf_idx][0][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
        mma_h884(A_frag[reg_buf_idx][m_idx][2], A_frag[reg_buf_idx][m_idx][3],
                 B_frag[reg_buf_idx][1][n_idx * 2],
                 B_frag[reg_buf_idx][1][n_idx * 2 + 1], C_frag[m_idx][n_idx][0],
                 C_frag[m_idx][n_idx][1], C_frag[m_idx][n_idx][2],
                 C_frag[m_idx][n_idx][3]);
      }
    }
  }

  __device__ void stg() {
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 2; st_iter_m++) {
      for (int st_iter_n = 0; st_iter_n < 2; st_iter_n++) {
        half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                      st_iter_m * 16 * params.N + st_iter_n * 16;
        bool guard = (blockIdx.x * 32 + store_c_row_base_idx + st_iter_m * 16) <
                         params.M &&
                     (blockIdx.y * 128 + store_c_col_base_idx +
                      st_iter_n * 16) < params.N;
        stg128(C_frag[st_iter_m][st_iter_n][0], C_frag[st_iter_m][st_iter_n][1],
               C_frag[st_iter_m][st_iter_n][2], C_frag[st_iter_m][st_iter_n][3],
               C_ptr, guard);
      }
    }
  }

  const SM70_GEMM_A16W8_Params<half, QType>& params;

  int load_a_base_offset;
  int load_b_base_offset_0;
  int load_b_base_offset_1;
  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_base_idx;
  half* this_block_C_base_ptr = nullptr;

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;

  // 2 denotes double buffer, first 2 denotes M direction, second 4 denotes K
  // direction
  uint32_t A_frag[2][2][4];
  // 2 denotes double buffer, first 2 denotes K direction, second 4 denotes N
  // direction
  uint32_t B_frag[2][2][4];
  // first 2 denotes M direction, second 2 denotes N direction
  uint32_t C_frag[2][2][4]{};

  int A_kphase_col_adjust;
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: K x N, matrix C: M x N (row-major)
 *  K % 8 == 0 && N % 8 == 0
 *  accumulator precision: FP16
 *  output datatype: FP16
 *
 *  BLOCK_TILE: m32n128k32
 *  BLOCK_SIZE: 128
 *  WARP_TILE:  m32n32k32
 */
template <typename QType>
__global__ void __launch_bounds__(128, 4)
    volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk_kernel(
        const SM70_GEMM_A16W8_Params<half, QType> params) {
  // A smem size = 32 * 32 * 2B/elem * 2(double buffer) = 4KB
  // B smem size = 128 * 32 * 2B/elem * 2(double biffer) = 16KB
  static constexpr int SMEM_SIZE = 32 * 32 * 2 * 2 + 128 * 32 * 2 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + 32 * 32 * 2 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);
  uint32_t A_smem_switch_offset = 32 * 32 * 2;
  uint32_t B_smem_switch_offset = 128 * 32 * 2;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_A16W8_32x128x32_SM70_SplitK<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  int tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                       ? params.SplitK
                       : params.K - blockIdx.z * params.SplitK;
  int k_main_loop = (tb_k_slice + 31) / 32 - 1;
  int first_k_tile = tb_k_slice - k_main_loop * 32;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_f16_32x128x32_SM70_SplitK<QType> compute_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);
  gmem_tile.sts(write_smem_buf_idx);

  __syncthreads();
  compute_tile.lds_A(read_smem_buf_idx, 0, 0);
  compute_tile.lds_B(read_smem_buf_idx, 0, 0);
  int reg_buf_idx = 1;

  for (int k_tile_idx = 0; k_tile_idx < k_main_loop; k_tile_idx++) {
    write_smem_buf_idx ^= 1;
    gmem_tile.ldg();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
      if (k_phase_idx == 3) {
        gmem_tile.sts(write_smem_buf_idx);
        read_smem_buf_idx ^= 1;
        __syncthreads();
      }

      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, (k_phase_idx + 1) % 4);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
    if (k_phase_idx < 3) {
      compute_tile.lds_A(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
      compute_tile.lds_B(read_smem_buf_idx, reg_buf_idx, k_phase_idx + 1);
    }
    compute_tile.mma(reg_buf_idx ^ 1);
    reg_buf_idx ^= 1;
  }

  compute_tile.stg();
}

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count,
    const SplitKParams splitk_params, const float alpha, cudaStream_t stream) {
  half* C_split = static_cast<half*>(workspace);

  int grid_x = (M + 31) / 32;
  int grid_y = (N + 127) / 128;
  int grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(128);

  SM70_GEMM_A16W8_Params<half, QType> params{
      A, B, B_scale, B_zero,          C,       M,
      N, K, 0,       log2(GroupSize), C_split, splitk_params.SplitK};

  volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk_kernel<QType>
      <<<grid, block, 0, stream>>>(params);

  // SplitK reduce
  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                           grid_z, alpha, stream);
}

template void
volta_hgemm_A16W8_f16_f16_128x128x32_mma884<int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const int, const int, const int, const int, void*, const int, const float,
    cudaStream_t);
template void
volta_hgemm_A16W8_f16_f16_128x128x32_mma884<int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const int, const int, const int, const int, void*, const int, const float,
    cudaStream_t);
template void
volta_hgemm_A16W8_f16_f16_128x128x32_mma884<int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const int, const int, const int, const int, void*, const int, const float,
    cudaStream_t);
template void
volta_hgemm_A16W8_f16_f16_128x128x32_mma884<int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const int, const int, const int, const int, void*, const int, const float,
    cudaStream_t);
template void
volta_hgemm_A16W8_f16_f16_128x128x32_mma884<int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const int, const int, const int, const int, void*, const int, const float,
    cudaStream_t);

template void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Identity>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int,
                                       const int, void*, const int,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Gelu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::GeluTanh>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int,
                                       const int, void*, const int,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Relu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Silu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);

template void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Identity>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int,
                                       const int, void*, const int,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Gelu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::GeluTanh>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int,
                                       const int, void*, const int,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Relu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Silu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, const int, void*,
                                   const int, const SplitKParams, const float,
                                   cudaStream_t);

}  // namespace cuda
}  // namespace allspark