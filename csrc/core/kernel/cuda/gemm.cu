/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm.cu
 */

#include <cusparse.h>

#include "cuda_kernel.h"
#include "gemm_utils.h"
#include "hie/cuda_activation.hpp"
#define CUSPARSE_CHECK(err) __cusparseCheck(err, __FILE__, __LINE__)
inline void __cusparseCheck(cusparseStatus_t err, const char* file,
                            const int line) {
#ifdef ERROR_CHECK
#ifdef ENABLE_CUSPARSELT
  if (err != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "cusparse call failed at %s:%i : %s\n", file, line,
            cusparseGetErrorString(err));
    exit(-1);
  }
#endif
#endif
  return;
}
namespace allspark {
namespace cuda {
/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 *
 */
template <uint32_t BLOCK_SIZE, uint32_t Bldg_pack_size>
struct GmemTile_fp16_32x128x16 {
  // element num loaded by per thread from Global Memory
  static constexpr int THREAD_ELEM_CNT_A = 32 * 16 / BLOCK_SIZE;   // 4
  static constexpr int THREAD_ELEM_CNT_B = 16 * 128 / BLOCK_SIZE;  // 16
  const int WARP_CNT = BLOCK_SIZE / 32;                            // 4

  const int ROW_OFFSET_OF_A_SMEM = 32 * 2 + 4;
  const int ROW_OFFSET_OF_B_SMEM = 128;

  __device__ GmemTile_fp16_32x128x16(const GEMM_Fp16_Params& k_params,
                                     const uint32_t& A_smem_addr,
                                     const uint32_t& B_smem_addr,
                                     const uint32_t& A_switch_offset,
                                     const uint32_t& B_switch_offset)
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
      uint32_t m_idx =
          blockIdx.x * 32 + load_a_row_base_idx + (i / 2) * 16 + i % 2;
      if (m_idx < params.M) {
        A_ldg_guard[i] = true;
      }
    }

    B_ldg_guard = false;
    uint32_t n_idx = blockIdx.y * 128 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }

// initialize a_regs all 0
#pragma unroll
    for (int i = 0; i < THREAD_ELEM_CNT_A; ++i) {
      a_regs[i] = 0;
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

    if (B_ldg_guard) {
      // load B
      const half* this_B_ptr =
          this_block_B_base_ptr + (tb_k_slice - first_k_tile) * params.N;
      const uint32_t b_rows_one_iter = (BLOCK_SIZE * Bldg_pack_size) / 128;
#pragma unroll
      for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_B / Bldg_pack_size;
           ++load_iter) {
        int load_b_offset =
            load_b_base_offset + load_iter * b_rows_one_iter * params.N;
        if ((load_b_row_base_idx + load_iter * b_rows_one_iter) <
            first_k_tile) {
          load_gdata<half, Bldg_pack_size>(b_regs + load_iter * Bldg_pack_size,
                                           this_B_ptr + load_b_offset);
        }
      }
    }
  }

  /*
   *  Load TILE_A:
   *     because the calculation is in the form of vector outer product,
   *     in order to maximize the SMEM read efficiency, BLOCK_TILE_A(m32n16)
   * needs to be continuously stored in SMEM in the direction of M but matrix A
   * is row-major, that is, the K direction is continuous and SMEM bank size is
   * 4B, datatype is FP16, so a thread load 2 elements in two adjacent rows in 2
   * successive iter
   */
  __device__ void load_from_gmem() {
#pragma unroll
    for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_A; load_iter++) {
      if (A_ldg_guard[load_iter]) {
        a_regs[load_iter] =
            *(this_block_A_base_ptr + load_a_base_offset +
              (load_iter / 2) * 16 * params.K + (load_iter % 2) * params.K);
      }
    }

    if (B_ldg_guard) {
      // load B
      const uint32_t b_rows_one_iter = (BLOCK_SIZE * Bldg_pack_size) / 128;
#pragma unroll
      for (int load_iter = 0; load_iter < THREAD_ELEM_CNT_B / Bldg_pack_size;
           ++load_iter) {
        int load_b_offset =
            load_b_base_offset + load_iter * b_rows_one_iter * params.N;
        load_gdata<half, Bldg_pack_size>(b_regs + load_iter * Bldg_pack_size,
                                         this_block_B_base_ptr + load_b_offset);
      }
    }

    // switch to next BLOCK_TILE_K
    this_block_A_base_ptr += 16;
    this_block_B_base_ptr += 16 * params.N;
  }

  /*
   *  Store A_TILE:
   *     a thread combine 2 fp16 into a float and store 4 elements to SMEM in 2
   * iter
   */
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
      store_sdata<half, Bldg_pack_size>(
          b_regs + store_iter * Bldg_pack_size,
          B_smem_addr + (store_b_base_offset +
                         store_iter * b_rows_one_iter * ROW_OFFSET_OF_B_SMEM) *
                            2 /* ELEM SIZE */);
    }
  }

  const half* this_block_A_base_ptr = nullptr;
  const half* this_block_B_base_ptr = nullptr;
  uint32_t load_a_base_offset;
  uint32_t load_b_base_offset;

  uint32_t load_a_col_idx;
  uint32_t load_b_row_base_idx;

  uint32_t store_a_base_offset;
  uint32_t store_b_base_offset;

  bool A_ldg_guard[4];
  bool B_ldg_guard;

  half a_regs[THREAD_ELEM_CNT_A];
  half b_regs[THREAD_ELEM_CNT_B];

  const uint32_t A_smem_base_addr, B_smem_base_addr;
  const uint32_t A_smem_switch_offset, B_smem_switch_offset;

  const GEMM_Fp16_Params& params;
};

template <uint32_t Bstg_pack_size>
struct ComputeTile_f16_32x128x16 {
  const int WARP_SIZE = 32;
  const int ROW_OFFSET_OF_A_SMEM = 32 * 2 + 4;
  const int ROW_OFFSET_OF_B_SMEM = 128;

  __device__ ComputeTile_f16_32x128x16(const GEMM_Fp16_Params& k_params,
                                       const uint32_t& A_smem_addr,
                                       const uint32_t& B_smem_addr,
                                       const uint32_t& A_switch_offset,
                                       const uint32_t& B_switch_offset)
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

  const GEMM_Fp16_Params& params;

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
template <uint32_t Bldg_pack_size>
__global__
__launch_bounds__(128) void gemm_fp16_32x128x16_nn_Aldg1_splitk_nonfused_kernel(
    const GEMM_Fp16_Params params) {
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
  GmemTile_fp16_32x128x16<128, Bldg_pack_size> gmem_tile(
      params, A_smem_addr, B_smem_addr, A_smem_switch_offset,
      B_smem_switch_offset);

  int write_smem_buf_idx = 0;
  int read_smem_buf_idx = 0;

  // const int k_main_loop = (params.splitK + 31) / 32 - 1;
  uint32_t tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                            ? params.SplitK
                            : params.K - blockIdx.z * params.SplitK;
  uint32_t k_main_loop = (tb_k_slice + 15) / 16 - 1;
  uint32_t first_k_tile = tb_k_slice - k_main_loop * 16;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_f16_32x128x16<Bldg_pack_size> compute_tile(
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

template <int BLOCK, int N_MATRIX, template <class> class ActiveFunc>
__global__ void gemm_fp16_splitk_reduce_kernel(const half* C_split,
                                               const half* bias, half* C,
                                               const int n, const int n_matrix,
                                               const int matrix_size) {
  int idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx >= matrix_size) {
    return;
  }

  __half sum(0);

  int n_mat = N_MATRIX > 0 ? N_MATRIX : n_matrix;
  for (int i = 0; i < n_mat; ++i) {
    sum += C_split[idx + i * matrix_size];
  }

  if (bias != nullptr) {
    sum += bias[idx % n];
  }
  C[idx] = ActiveFunc<half>::Op(sum);
}

template <template <class> class ActiveFunc>
void gemm_fp16_splitk_reduce(const __half* C_split, const half* bias, __half* C,
                             const uint32_t m, const uint32_t n,
                             const uint32_t n_matrix, cudaStream_t stream) {
  const int BLOCK = 128;
  int matrix_size = m * n;
  int grid = (matrix_size + BLOCK - 1) / BLOCK;

  void (*kernel)(const half*, const half*, half*, int, int, int) = nullptr;

  switch (n_matrix) {
    case 4:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 4, ActiveFunc>;
      break;
    case 5:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 5, ActiveFunc>;
      break;
    case 6:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 6, ActiveFunc>;
      break;
    case 7:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 7, ActiveFunc>;
      break;
    case 8:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 8, ActiveFunc>;
      break;
    case 9:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 9, ActiveFunc>;
      break;
    case 10:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 10, ActiveFunc>;
      break;
    case 11:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 11, ActiveFunc>;
      break;
    case 12:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, 12, ActiveFunc>;
      break;
    default:
      kernel = gemm_fp16_splitk_reduce_kernel<BLOCK, -1, ActiveFunc>;
      break;
  }

  kernel<<<grid, BLOCK, 0, stream>>>(C_split, bias, C, n, n_matrix,
                                     matrix_size);
}

template <template <class> class ActiveFunc>
void hgemm_32x128x16_simt_Aldg1(const half* A, const half* B, const half* bias,
                                half* C, const uint32_t M, const uint32_t N,
                                const uint32_t K, const uint32_t sm_count,
                                void* workspace, cudaStream_t stream) {
  half* C_split = static_cast<half*>(workspace);
  uint32_t grid_x = (M + 31) / 32;
  uint32_t grid_y = (N + 127) / 128;
  uint32_t grid_z;

  const float SPLIT_THRESHOLD = 5;
  uint32_t n_slice = 2;
  for (n_slice = 1; n_slice < K / 128; ++n_slice) {
    uint32_t n_block = grid_x * grid_y * n_slice;
    if (n_block >= sm_count * SPLIT_THRESHOLD &&
        (n_block % sm_count == 0 || n_block % sm_count >= sm_count / 2)) {
      break;
    }
  }

  uint32_t k_slice =
      (K / n_slice) % 16 == 0 ? K / n_slice : K / n_slice / 16 * 16 + 16;

  grid_z = (K + k_slice - 1) / k_slice;

  dim3 block(128);
  dim3 grid(grid_x, grid_y, grid_z);
  GEMM_Fp16_Params params{A, B, C, C_split, M, N, K, k_slice};

  uint32_t Bldg_pack_size = 8;
  while (N % Bldg_pack_size != 0) {
    Bldg_pack_size /= 2;
  }

  switch (Bldg_pack_size) {
    case 8:
      gemm_fp16_32x128x16_nn_Aldg1_splitk_nonfused_kernel<8>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 4:
      gemm_fp16_32x128x16_nn_Aldg1_splitk_nonfused_kernel<4>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 2:
      gemm_fp16_32x128x16_nn_Aldg1_splitk_nonfused_kernel<2>
          <<<grid, block, 0, stream>>>(params);
      break;
    case 1:
      gemm_fp16_32x128x16_nn_Aldg1_splitk_nonfused_kernel<1>
          <<<grid, block, 0, stream>>>(params);
      break;
    default:
      LOG(ERROR) << "Error pack size!";
      break;
  }
  gemm_fp16_splitk_reduce<ActiveFunc>(C_split, bias, C, M, N, grid_z, stream);
}

template <typename T>
__global__ static void broadcast_kernel(T* dst, const T* src, int N,
                                        int stride_block, int stride_bias) {
  uint32_t n_thread = gridDim.x * blockDim.x;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t idx = tid; idx < N; idx += n_thread) {
    int batch_idx = idx / stride_block;
    int bias_idx = (idx % stride_block) % stride_bias;
    dst[idx] = src[batch_idx * stride_bias + bias_idx];
  }
}

template <>
void broadcast_kernel_launcher<float>(float* dst, const float* src,
                                      int gemm_size, int hidden_size,
                                      int batch_size, cudaStream_t stream) {
  int N = batch_size * gemm_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  broadcast_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      dst, src, N, gemm_size, hidden_size);
}
#ifdef ENABLE_FP16
template <>
void broadcast_kernel_launcher<half>(half* dst, const half* src, int gemm_size,
                                     int hidden_size, int batch_size,
                                     cudaStream_t stream) {
  int N = batch_size * gemm_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  broadcast_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      dst, src, N, gemm_size, hidden_size);
}
#endif
template <>
void broadcast_kernel_launcher<hie::bfloat16>(hie::bfloat16* dst,
                                              const hie::bfloat16* src,
                                              int gemm_size, int hidden_size,
                                              int batch_size,
                                              cudaStream_t stream) {
  int N = batch_size * gemm_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  broadcast_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      dst, src, N, gemm_size, hidden_size);
}
template <>
void GemmWraper<float>(float* matrix_C, const float* matrix_A,
                       const float* matrix_B, const float* bias, int m, int n,
                       int k, bool transA, bool transB, int lda, int ldb,
                       int ldc, float alpha, float beta, const float* bin_res,
                       cublasHandle_t handle, cudaStream_t stream) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, 1, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, 1, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                            CUDA_R_32F, ldb, matrix_A, CUDA_R_32F, lda, &beta,
                            matrix_C, CUDA_R_32F, ldc, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT));
}
#ifdef ENABLE_FP16
template <>
void GemmWraper<half>(half* matrix_C, const half* matrix_A,
                      const half* matrix_B, const half* bias, int m, int n,
                      int k, bool transA, bool transB, int lda, int ldb,
                      int ldc, float alpha, float beta, const half* bin_res,
                      cublasHandle_t handle, cudaStream_t stream) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, 1, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, 1, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                            CUDA_R_16F, ldb, matrix_A, CUDA_R_16F, lda, &beta,
                            matrix_C, CUDA_R_16F, ldc, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
#endif
template <>
void GemmWraper<hie::bfloat16>(hie::bfloat16* matrix_C,
                               const hie::bfloat16* matrix_A,
                               const hie::bfloat16* matrix_B,
                               const hie::bfloat16* bias, int m, int n, int k,
                               bool transA, bool transB, int lda, int ldb,
                               int ldc, float alpha, float beta,
                               const hie::bfloat16* bin_res,
                               cublasHandle_t handle, cudaStream_t stream) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, 1, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, 1, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                            CUDA_R_16BF, ldb, matrix_A, CUDA_R_16BF, lda, &beta,
                            matrix_C, CUDA_R_16BF, ldc, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
template <>
void StridedBatchGemmWraper<float>(float* matrix_C, const float* matrix_A,
                                   const float* matrix_B, const float* bias,
                                   int m, int n, int k, bool transA,
                                   bool transB, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batch,
                                   const float* bin_res, cublasHandle_t handle,
                                   cudaStream_t stream) {
  // int strideA = m * k;
  int strideA = 0;
  int strideB = k * n;
  int strideC = m * n;
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. batch cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, batch, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, transB_, transA_, n, m, k, &alpha, matrix_B, CUDA_R_32F, ldb,
      strideB, matrix_A, CUDA_R_32F, lda, strideA, &beta, matrix_C, CUDA_R_32F,
      ldc, strideC, batch, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}
#ifdef ENABLE_FP16
template <>
void StridedBatchGemmWraper<half>(half* matrix_C, const half* matrix_A,
                                  const half* matrix_B, const half* bias, int m,
                                  int n, int k, bool transA, bool transB,
                                  int lda, int ldb, int ldc, float alpha,
                                  float beta, int batch, const half* bin_res,
                                  cublasHandle_t handle, cudaStream_t stream) {
  // int strideA = m * k;
  int strideA = 0;
  int strideB = k * n;
  int strideC = m * n;
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. batch cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, batch, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, transB_, transA_, n, m, k, &alpha, matrix_B, CUDA_R_16F, ldb,
      strideB, matrix_A, CUDA_R_16F, lda, strideA, &beta, matrix_C, CUDA_R_16F,
      ldc, strideC, batch, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
#endif
template <>
void StridedBatchGemmWraper<hie::bfloat16>(
    hie::bfloat16* matrix_C, const hie::bfloat16* matrix_A,
    const hie::bfloat16* matrix_B, const hie::bfloat16* bias, int m, int n,
    int k, bool transA, bool transB, int lda, int ldb, int ldc, float alpha,
    float beta, int batch, const hie::bfloat16* bin_res, cublasHandle_t handle,
    cudaStream_t stream) {
  // int strideA = m * k;
  int strideA = 0;
  int strideB = k * n;
  int strideC = m * n;
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  // assert beta = 0.f
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  if (bin_res) {
    // aka. batch cudaMemcpy2D
    broadcast_kernel_launcher(matrix_C, bin_res, m * n, m * n, batch, stream);
    beta = 1.f;
  }
  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, transB_, transA_, n, m, k, &alpha, matrix_B, CUDA_R_16F, ldb,
      strideB, matrix_A, CUDA_R_16F, lda, strideA, &beta, matrix_C, CUDA_R_16F,
      ldc, strideC, batch, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
void BatchGemmI8Wrapper(void** matrix_C, void** matrix_A, void** matrix_B,
                        int m, int n, int k, bool transA, bool transB,
                        int alpha, int beta, int lda, int ldb, int ldc,
                        int batch, cublasHandle_t handle) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS(cublasGemmBatchedEx(
      handle, transB_, transA_, n, m, k, &alpha, (const void**)matrix_B,
      CUDA_R_8I, ldb, (const void**)matrix_A, CUDA_R_8I, lda, &beta,
      (void**)matrix_C, CUDA_R_32I, ldc, batch, CUDA_R_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
#ifdef ENABLE_SPARSE
template <>
void CuSparseGemmCSC<float>(float* matrix_C, const float* matrix_A,
                            const int* col_offset, const int* row_indices,
                            const float* val, int nnz, const float* bias, int m,
                            int n, int k, bool transA_, bool transB_, int lda,
                            int ldb, int ldc, float alpha, float beta,
                            int batch, cudaStream_t stream) {
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);
  cusparseSetStream(handle, stream);
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void* buffer = 0;
  size_t bufferSize = 0;
  cusparseOperation_t transB =
      transA_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transA =
      transB_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, k, nnz, (void*)col_offset,
                                   (void*)row_indices, (void*)val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, k, m, k, (void*)matrix_A,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, n, m, n, matrix_C, CUDA_R_32F,
                                     CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, transA, transB, &alpha, matA, matB, &beta, matC, CUDA_R_32F,
      static_cast<cusparseSpMMAlg_t>(0), &bufferSize));
  cudaMalloc(&buffer, bufferSize);
  CUSPARSE_CHECK(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta,
                              matC, CUDA_R_32F,
                              static_cast<cusparseSpMMAlg_t>(0), buffer));
}
#ifdef ENABLE_FP16
template <>
void CuSparseGemmCSC<half>(half* matrix_C, const half* matrix_A,
                           const int* col_offset, const int* row_indices,
                           const half* val, int nnz, const half* bias, int m,
                           int n, int k, bool transA_, bool transB_, int lda,
                           int ldb, int ldc, float alpha, float beta, int batch,
                           cudaStream_t stream) {
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);
  cusparseSetStream(handle, stream);
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void* buffer = 0;
  size_t bufferSize = 0;
  cusparseOperation_t transB =
      transA_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transA =
      transB_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, k, nnz, (void*)col_offset,
                                   (void*)row_indices, (void*)val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, k, m, k, (void*)matrix_A,
                                     CUDA_R_16F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, n, m, n, matrix_C, CUDA_R_16F,
                                     CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, transA, transB, &alpha, matA, matB, &beta, matC, CUDA_R_32F,
      static_cast<cusparseSpMMAlg_t>(0), &bufferSize));
  cudaMalloc(&buffer, bufferSize);
  CUSPARSE_CHECK(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta,
                              matC, CUDA_R_32F,
                              static_cast<cusparseSpMMAlg_t>(0), buffer));
}
#endif
template <>
void CuSparseGemmCSC<hie::bfloat16>(
    hie::bfloat16* matrix_C, const hie::bfloat16* matrix_A,
    const int* col_offset, const int* row_indices, const hie::bfloat16* val,
    int nnz, const hie::bfloat16* bias, int m, int n, int k, bool transA_,
    bool transB_, int lda, int ldb, int ldc, float alpha, float beta, int batch,
    cudaStream_t stream) {
  if (bias) {
    broadcast_kernel_launcher(matrix_C, bias, m * n, n, batch, stream);
    beta = 1.f;
  }
  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);
  cusparseSetStream(handle, stream);
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void* buffer = 0;
  size_t bufferSize = 0;
  cusparseOperation_t transB =
      transA_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transA =
      transB_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, k, nnz, (void*)col_offset,
                                   (void*)row_indices, (void*)val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, k, m, k, (void*)matrix_A,
                                     CUDA_R_16F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, n, m, n, matrix_C, CUDA_R_16F,
                                     CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, transA, transB, &alpha, matA, matB, &beta, matC, CUDA_R_32F,
      static_cast<cusparseSpMMAlg_t>(0), &bufferSize));
  cudaMalloc(&buffer, bufferSize);
  CUSPARSE_CHECK(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta,
                              matC, CUDA_R_32F,
                              static_cast<cusparseSpMMAlg_t>(0), buffer));
}
#endif

void GemmInt8(int32_t* matrix_C, const int8_t* matrix_A, const int8_t* matrix_B,
              int m, int n, int k, bool transA, bool transB, int lda, int ldb,
              int ldc, int alpha, int beta, cublasHandle_t handle,
              cudaStream_t stream) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS(cublasGemmEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                            CUDA_R_8I, ldb, matrix_A, CUDA_R_8I, lda, &beta,
                            matrix_C, CUDA_R_32I, ldc, CUDA_R_32I,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template void hgemm_32x128x16_simt_Aldg1<hie::activation::Identity>(
    const half*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, cudaStream_t);

template void hgemm_32x128x16_simt_Aldg1<hie::activation::Gelu>(
    const half*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, cudaStream_t);

template void hgemm_32x128x16_simt_Aldg1<hie::activation::GeluTanh>(
    const half*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, cudaStream_t);

template void hgemm_32x128x16_simt_Aldg1<hie::activation::Relu>(
    const half*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, cudaStream_t);

template void hgemm_32x128x16_simt_Aldg1<hie::activation::Silu>(
    const half*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, void*, cudaStream_t);

}  // namespace cuda
}  // namespace allspark
