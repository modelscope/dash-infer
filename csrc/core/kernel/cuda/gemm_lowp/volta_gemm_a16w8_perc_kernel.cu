/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    volta_gemm_a16w8_perc_kernel.cu
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
 * PerChannel
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_A16W8_PerC_32x128x32_SM70_SplitK {
  // element num loaded by a LDG inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 4;
  const int WARP_CNT = BLOCK_SIZE / 32;  // 4

  __device__ GmemTile_A16W8_PerC_32x128x32_SM70_SplitK(
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
    const half* this_B_scale_ptr = params.B_scale_ptr + n_idx;
    const half* this_B_zero_ptr = params.B_zero_ptr + n_idx;
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

    // load A
    ldg128_cg(a_regs[0], a_regs[1], a_regs[2], a_regs[3],
              this_block_A_base_ptr + load_a_offset, A_ldg_guard);

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N;
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
    volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk_kernel(
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
  GmemTile_A16W8_PerC_32x128x32_SM70_SplitK<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  // const int k_main_loop = (params.K + 31) / 32 - 1;
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
void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    void* workspace, const SplitKParams splitk_params, const float alpha,
    cudaStream_t stream) {
  half* C_split = static_cast<half*>(workspace);

  int grid_x = (M + 31) / 32;
  int grid_y = (N + 127) / 128;
  int grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(128);

  SM70_GEMM_A16W8_Params<half, QType> params{
      A, B, B_scale, B_zero, C, M, N, K, 0, -1, C_split, splitk_params.SplitK};

  volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk_kernel<QType>
      <<<grid, block, 0, stream>>>(params);

  // SplitK reduce
  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                           grid_z, alpha, stream);
}

template void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Identity>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int, void*,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Gelu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, void*,
                                   const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::GeluTanh>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const int, const int, const int, void*,
                                       const SplitKParams, const float,
                                       cudaStream_t);
template void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Relu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, void*,
                                   const SplitKParams, const float,
                                   cudaStream_t);
template void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
    int8_t, hie::activation::Silu>(const half*, const int8_t*, const half*,
                                   const half*, const half*, half*, const int,
                                   const int, const int, void*,
                                   const SplitKParams, const float,
                                   cudaStream_t);
}  // namespace cuda
}  // namespace allspark