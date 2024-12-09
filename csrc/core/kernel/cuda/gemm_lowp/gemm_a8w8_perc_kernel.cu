/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a8w8_perc_kernel.cu
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gemm_a16w8_kernel.h"
#include "gemm_lowp_utils.cuh"
using namespace std;

namespace allspark {
namespace cuda {

size_t aligned_size(size_t n, size_t aligned = 16) {
  return (n + aligned - 1) / aligned * aligned;
}
// A = M x K
// require K % 8 == 0
template <int BLOCK, int UNROLL, typename FType>
__global__ void per_channel_symm_dynamic_quantization_kernel(
    const FType* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K) {
  const int row_idx = blockIdx.z * gridDim.y + blockIdx.y;
  const int col_base_idx = blockIdx.x * BLOCK * UNROLL * 8 + threadIdx.x * 8;
  const int base_offset = row_idx * K + col_base_idx;

  if (row_idx < M) {
    __shared__ float scale_smem;

    // load data
    FType fdata_reg[UNROLL][8];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      ldg128_cg_0(*reinterpret_cast<uint32_t*>(fdata_reg[i]),
                  *(reinterpret_cast<uint32_t*>(fdata_reg[i]) + 1),
                  *(reinterpret_cast<uint32_t*>(fdata_reg[i]) + 2),
                  *(reinterpret_cast<uint32_t*>(fdata_reg[i]) + 3),
                  fdata + base_offset + i * BLOCK * 8,
                  (col_base_idx + i * BLOCK * 8) < K);
    }

    float abs_max = 0.f;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        abs_max = fmaxf(abs_max, fabsf(fdata_reg[i][j]));
      }
    }

    // block reduce-max
    abs_max = ReduceBlock<MaxOp, float, BLOCK>(abs_max);

    // global reduce-max and calculate scale
    if (threadIdx.x == 0) {
      float* red_max_ptr = red_max + row_idx;
      uint32_t* red_count_ptr = red_count + row_idx;
      atomicMax(reinterpret_cast<int*>(red_max_ptr),
                reinterpret_cast<const int&>(abs_max));
      __threadfence();
      atomicInc(red_count_ptr, gridDim.x);

      uint32_t count;
      do {
        // make sure the ld.cg inside the do-wile loop
        __threadfence_block();
        asm volatile("ld.global.cg.b32 %0, [%1];"
                     : "=r"(count)
                     : "l"(red_count_ptr));
      } while (count != gridDim.x);

      asm("ld.global.cg.b32 %0, [%1];" : "=f"(abs_max) : "l"(red_max_ptr));
      scale_smem = 127 / abs_max;
    }
    __syncthreads();

    // store scale
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      scale[row_idx] = abs_max / 127;
    }

    // quantization
    int8_t qdata_reg[UNROLL][8];
    float scale_rec = scale_smem;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        qdata_reg[i][j] = static_cast<int8_t>(__float2int_rn(fmaxf(
            -128.f,
            fminf(127.f, static_cast<float>(fdata_reg[i][j]) * scale_rec))));
      }
    }

    // calculate row reduce-sum
    if (red_sum != nullptr) {
      int32_t sum = 0;
#pragma unroll
      for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum += qdata_reg[i][j];
        }
      }

      sum = ReduceBlock<SumOp, int32_t, BLOCK>(sum);
      int32_t* red_sum_ptr = red_sum + row_idx;
      if (threadIdx.x == 0) {
        atomicAdd(red_sum_ptr, sum);
      }
    }

// store qdata
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      int col_idx = col_base_idx + i * BLOCK * 8;
      if (col_idx < K) {
        stg64(*reinterpret_cast<uint32_t*>(qdata_reg[i]),
              *(reinterpret_cast<uint32_t*>(qdata_reg[i]) + 1),
              qdata + base_offset + i * BLOCK * 8);
      }
    }
  }
}

template <typename FType>
void per_channel_symm_dynamic_quantization(const FType* fdata, int8_t* qdata,
                                           float* scale, float* red_max,
                                           uint32_t* red_count,
                                           int32_t* red_sum, const int M,
                                           const int K, int sm_count,
                                           cudaStream_t stream) {
  if (K % 8 != 0) {
    std::cerr << "[ERROR]: Now kernel only supprt K % 8 = 0" << std::endl;
  }

  cudaMemsetAsync((void*)red_max, 0,
                  aligned_size(M) * sizeof(float),  // red_sum set 0
                  stream);
  cudaMemsetAsync((void*)red_count, 0,
                  aligned_size(M) * sizeof(uint32_t),  // red_count set 0
                  stream);
  cudaMemsetAsync((void*)red_sum, 0,
                  aligned_size(M) * sizeof(int32_t),  // red_sum set 0
                  stream);
  const int BLOCK_MAX = 256;
  const int UNROLL_MAX = 8;
  int block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_per_sm,
      per_channel_symm_dynamic_quantization_kernel<BLOCK_MAX, UNROLL_MAX,
                                                   FType>,
      BLOCK_MAX, 0);
  int max_reduce_len = BLOCK_MAX * UNROLL_MAX * 8 * block_per_sm * sm_count;

  const int grid_y = M;
  const int PACK = 8;
  if (K <= 2048) {
    const int BLOCK = 64;
    const int UNROLL = 4;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 3072) {
    const int BLOCK = 64;
    const int UNROLL = 6;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 4096) {
    const int BLOCK = 64;
    const int UNROLL = 8;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 6144) {
    const int BLOCK = 64;
    const int UNROLL = 12;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 8192) {
    const int BLOCK = 256;
    const int UNROLL = 4;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 12288) {
    const int BLOCK = 256;
    const int UNROLL = 6;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 13312) {
    const int BLOCK = 128;
    const int UNROLL = 13;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 16384) {
    const int BLOCK = 256;
    const int UNROLL = 8;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 20480) {
    const int BLOCK = 256;
    const int UNROLL = 5;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 24576) {
    const int BLOCK = 256;
    const int UNROLL = 6;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 32768) {
    const int BLOCK = 256;
    const int UNROLL = 8;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 36864) {
    const int BLOCK = 256;
    const int UNROLL = 9;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= 40960) {
    const int BLOCK = 512;
    const int UNROLL = 5;
    const int grid_x =
        (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK, UNROLL, FType>
        <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                     red_sum, M, K);
  } else if (K <= max_reduce_len) {
    const int grid_x = (K + BLOCK_MAX * UNROLL_MAX * PACK - 1) /
                       (BLOCK_MAX * UNROLL_MAX * PACK);
    dim3 grid(grid_x, grid_y);
    per_channel_symm_dynamic_quantization_kernel<BLOCK_MAX, UNROLL_MAX, FType>
        <<<grid, BLOCK_MAX, 0, stream>>>(fdata, qdata, scale, red_max,
                                         red_count, red_sum, M, K);
  } else {
    std::cerr << "[ERROR]: Now this kernel not support such a large K!"
              << std::endl;
  }
}

// restore from N32K16 order to K-major order
// K % 16 = 0, N_32align % 32 = 0
template <typename FType>
__global__ void __launch_bounds__(256)
    restore_n32k16_weight_to_nk_kernel(const int8_t* B_n32k16,
                                       const FType* B_scale_n32,
                                       const FType* B_zero_n32, int8_t* B_nk,
                                       FType* B_scale, FType* B_zero,
                                       const int N_32align, const int N,
                                       const int K) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  if (blockIdx.x < gridDim.x - 1) {
    // N32K16 format is arranged as (N_32align / 4)row x (K * 4)col
    // each block restore 8(row) x 512(col) N32K16-format B_n32k16
    // to 32(row) x 128(col) NK-format B_nk
    const int src_row_idx = blockIdx.y * 8 + warp_id;
    const int src_col_idx = blockIdx.x * 512 + lane_id * 16;
    const int src_offset = src_row_idx * K * 4 + src_col_idx;

    int8_t in_reg[16];
    bool ldg_guard = (src_row_idx < (N_32align / 4)) && (src_col_idx < (K * 4));
    ldg128_cg(*reinterpret_cast<uint32_t*>(in_reg),
              *(reinterpret_cast<uint32_t*>(in_reg) + 1),
              *(reinterpret_cast<uint32_t*>(in_reg) + 2),
              *(reinterpret_cast<uint32_t*>(in_reg) + 3), B_n32k16 + src_offset,
              ldg_guard);

    __shared__ char smem[32 * 128];
    uint32_t smem_addr = smem_u32addr(smem);

    const int sts_base_offset =
        warp_id * 128 + (lane_id % 4) * 2 + (lane_id / 4) * 16;
#pragma unroll
    for (int sts_iter = 0; sts_iter < 8; sts_iter++) {
      int sts_offset =
          sts_base_offset + (sts_iter / 2) * 8 * 128 + (sts_iter % 2) * 8;
      sts16(*(reinterpret_cast<uint16_t*>(in_reg) + sts_iter),
            smem_addr + sts_offset);
    }

    __syncthreads();
    const int lds_offset =
        warp_id * 4 * 128 + (lane_id % 8) * 16 + (lane_id / 8) * 128;
    int8_t out_reg[16];
    lds128(*reinterpret_cast<uint32_t*>(out_reg),
           *(reinterpret_cast<uint32_t*>(out_reg) + 1),
           *(reinterpret_cast<uint32_t*>(out_reg) + 2),
           *(reinterpret_cast<uint32_t*>(out_reg) + 3), smem_addr + lds_offset);

    const int dst_row_idx = blockIdx.y * 32 + warp_id * 4 + (lane_id / 8);
    const int dst_col_idx = blockIdx.x * 128 + (lane_id % 8) * 16;
    const int dst_offset = dst_row_idx * K + dst_col_idx;

    bool stg_guard = (dst_row_idx < N) && (dst_col_idx < K);
    if (stg_guard) {
      stg128(*reinterpret_cast<uint32_t*>(out_reg),
             *(reinterpret_cast<uint32_t*>(out_reg) + 1),
             *(reinterpret_cast<uint32_t*>(out_reg) + 2),
             *(reinterpret_cast<uint32_t*>(out_reg) + 3), B_nk + dst_offset);
    }
  } else {
    const int process_qparams_blk_num =
        (gridDim.y + (256 / 32) - 1) / (256 / 32);
    if (blockIdx.x == gridDim.x - 1 && blockIdx.y < process_qparams_blk_num) {
      int src_nidx = blockIdx.y * 256 + threadIdx.x;
      FType scale_val = 0, zero_val = 0;
      if (src_nidx < N_32align) {
        scale_val = B_scale_n32[src_nidx];
        zero_val = B_zero_n32[src_nidx];
      }
      int dst_nidx =
          blockIdx.y * 256 + warp_id * 32 + (lane_id % 4) * 8 + lane_id / 4;
      if (dst_nidx < N) {
        B_scale[dst_nidx] = scale_val;
        B_zero[dst_nidx] = zero_val;
      }
    }
  }
}

template <typename FType>
void restore_n32k16_weight_to_nk(const int8_t* B_n32k16,
                                 const FType* B_scale_n32,
                                 const FType* B_zero_n32, int8_t* B_nk,
                                 FType* B_scale, FType* B_zero,
                                 const int N_32align, const int N, const int K,
                                 cudaStream_t stream) {
  assert(K % 16 == 0);
  int grid_x = (K + 128 - 1) / 128 + 1;
  int grid_y = N_32align / 32;
  dim3 grid(grid_x, grid_y);

  const int BLOCK = 256;
  restore_n32k16_weight_to_nk_kernel<<<grid, BLOCK, 0, stream>>>(
      B_n32k16, B_scale_n32, B_zero_n32, B_nk, B_scale, B_zero, N_32align, N,
      K);
}

// A : PerChannel + Symmetric Quantization
// B : PerChannel + Asymmetric Quantization
// Dequant the int32-type immediate result obtained by A and B matrix
// multiplication FType is Half or BFloat16 N % 4 == 0
template <typename FType, int BLOCK, int UNROLL, typename ComputeType = float>
__global__ void __launch_bounds__(BLOCK)
    A_perc_symm_B_perc_asymm_dequantization_kernel(
        const int* imd_result, const float* A_scale, const int* A_redusm,
        const FType* B_scale, const FType* B_zero, const FType* bias,
        FType* result, const int M, const int N) {
  int ridx = blockIdx.y * UNROLL;
  int cidx = blockIdx.x * BLOCK * 4 + threadIdx.x * 4;
  int base_offset = ridx * N + cidx;

  int ld_reg[UNROLL][4] = {};

  float a_scale_reg[UNROLL];
  int a_redsum_reg[UNROLL];
  FType b_scale_reg[4] = {};
  FType b_zero_reg[4] = {};

  FType bias_reg[4] = {};
  FType st_reg[UNROLL][4];

  // load b_scale, b_zero, bias
  ldg64_ca(*reinterpret_cast<uint32_t*>(b_scale_reg),
           *reinterpret_cast<uint32_t*>(b_scale_reg + 2), B_scale + cidx,
           cidx < N);
  if (B_zero != nullptr) {
    ldg64_ca(*reinterpret_cast<uint32_t*>(b_zero_reg),
             *reinterpret_cast<uint32_t*>(b_zero_reg + 2), B_zero + cidx,
             cidx < N);
  }
  if (bias != nullptr) {
    ldg64_ca(*reinterpret_cast<uint32_t*>(bias_reg),
             *reinterpret_cast<uint32_t*>(bias_reg + 2), bias + cidx, cidx < N);
  }

// load a_redsum, a_scale
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (B_zero != nullptr) {
      ldg32_ca(a_redsum_reg[i], A_redusm + ridx + i, (ridx + i) < M);
    }
    ldg32_ca(a_scale_reg[i], A_scale + ridx + i, (ridx + i) < M);
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int offset = base_offset + i * N;
    ldg128_cg(ld_reg[i][0], ld_reg[i][1], ld_reg[i][2], ld_reg[i][3],
              imd_result + offset, cidx < N && (ridx + i) < M);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ComputeType fval = static_cast<ComputeType>(ld_reg[i][j]) -
                         static_cast<ComputeType>(b_zero_reg[j]) *
                             static_cast<ComputeType>(a_redsum_reg[i]);
      fval *= a_scale_reg[i] * static_cast<ComputeType>(b_scale_reg[j]);
      if (bias) {
        fval += static_cast<ComputeType>(bias_reg[j]);
      }
      st_reg[i][j] = static_cast<FType>(fval);
    }
    if ((ridx + i) < M && cidx < N) {
      *reinterpret_cast<int2*>(result + offset) =
          *reinterpret_cast<int2*>(st_reg[i]);
    }
  }
}

template <typename FType>
void A_perc_symm_B_perc_asymm_dequantization(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType* B_scale, const FType* B_zero, const FType* bias, FType* result,
    const int M, const int N, cudaStream_t stream) {
  const int BLOCK = 256;
  const int UNROLL = 4;
  const int PACK = 4;
  // const int size = M * N;

  const int grid_x = (N + BLOCK * PACK - 1) / (BLOCK * PACK);
  const int grid_y = (M + UNROLL - 1) / UNROLL;
  dim3 grid(grid_x, grid_y);

  if (N % PACK != 0) {
    std::cerr << "Now this kernel only support N % " << PACK << " == 0!"
              << std::endl;
  }

  A_perc_symm_B_perc_asymm_dequantization_kernel<FType, BLOCK, UNROLL>
      <<<grid, BLOCK, 0, stream>>>(imd_result, A_scale, A_redsum, B_scale,
                                   B_zero, bias, result, M, N);
}

// A : PerChannel + Symmetric Quantization
// B : PerChannel + Asymmetric Quantization
// Dequant the int32-type immediate result obtained by A and B matrix
// multiplication FType is Half or BFloat16 N % 4 == 0
template <typename FType, int BLOCK, int UNROLL, typename ComputeType = float>
__global__ void __launch_bounds__(BLOCK)
    A_perc_symm_B_perc_array_asymm_dequantization_kernel(
        const int* imd_result, const float* A_scale, const int* A_redsum,
        const FType** B_scale_arr, const FType** B_zero_arr, const FType* bias,
        FType* result, const int M, const int N) {
  int blk_idx = blockIdx.z;  // use blockIdx.z to handle max_block
  const FType* B_scale = B_scale_arr[blk_idx];
  const FType* B_zero = B_zero_arr[blk_idx];

  int ridx = blockIdx.y * UNROLL;
  int cidx = blockIdx.x * BLOCK * 4 + threadIdx.x * 4;
  int base_offset = blk_idx * M * N + ridx * N + cidx;
  int base_offset_m = blk_idx * M;
  int base_offset_n = blk_idx * N;

  float a_scale_reg[UNROLL];
  int a_redsum_reg[UNROLL];
  FType b_scale_reg[4] = {};
  FType b_zero_reg[4] = {};
  FType bias_reg[4] = {};
  FType st_reg[UNROLL][4];
  int ld_reg[UNROLL][4] = {};

  // Load b_scale, b_zero, bias
  ldg64_ca(*reinterpret_cast<uint32_t*>(b_scale_reg),
           *reinterpret_cast<uint32_t*>(b_scale_reg + 2), B_scale + cidx,
           cidx < N);

  if (B_zero != nullptr) {
    ldg64_ca(*reinterpret_cast<uint32_t*>(b_zero_reg),
             *reinterpret_cast<uint32_t*>(b_zero_reg + 2), B_zero + cidx,
             cidx < N);
  }

  if (bias != nullptr) {
    ldg64_ca(*reinterpret_cast<uint32_t*>(bias_reg),
             *reinterpret_cast<uint32_t*>(bias_reg + 2),
             bias + base_offset_n + cidx, cidx < N);
  }

// Load a_redsum, a_scale
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (B_zero != nullptr) {
      ldg32_ca(a_redsum_reg[i], A_redsum + base_offset_m + ridx + i,
               (ridx + i) < M);
    }
    ldg32_ca(a_scale_reg[i], A_scale + base_offset_m + ridx + i,
             (ridx + i) < M);
  }
// Perform dequantization
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int offset = base_offset + i * N;
    ldg128_cg(ld_reg[i][0], ld_reg[i][1], ld_reg[i][2], ld_reg[i][3],
              imd_result + offset, cidx < N && (ridx + i) < M);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ComputeType fval = static_cast<ComputeType>(ld_reg[i][j]) -
                         static_cast<ComputeType>(b_zero_reg[j]) *
                             static_cast<ComputeType>(a_redsum_reg[i]);
      fval *= a_scale_reg[i] * static_cast<ComputeType>(b_scale_reg[j]);
      if (bias) {
        fval += static_cast<ComputeType>(bias_reg[j]);
      }
      st_reg[i][j] = static_cast<FType>(fval);
    }

    if ((ridx + i) < M && cidx < N) {
      *reinterpret_cast<int2*>(result + offset) =
          *reinterpret_cast<int2*>(st_reg[i]);
    }
  }
}

template <typename FType>
void A_perc_symm_B_perc_array_asymm_dequantization(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType** B_scale, const FType** B_zero, const FType* bias,
    FType* result, const int M, const int N, const int max_block,
    cudaStream_t stream) {
  const int BLOCK = 256;
  const int UNROLL = 4;
  const int PACK = 4;
  const int grid_x = (N + BLOCK * PACK - 1) / (BLOCK * PACK);
  const int grid_y = (M + UNROLL - 1) / UNROLL;
  dim3 grid(grid_x, grid_y, max_block);  // Adding max_block dimension

  if (N % PACK != 0) {
    std::cerr << "Now this kernel only support N % " << PACK << " == 0!"
              << std::endl;
    return;
  }

  A_perc_symm_B_perc_array_asymm_dequantization_kernel<FType, BLOCK, UNROLL>
      <<<grid, BLOCK, 0, stream>>>(imd_result, A_scale, A_redsum, B_scale,
                                   B_zero, bias, result, M, N);
}

//-------------------
//-------------------
template void restore_n32k16_weight_to_nk<half>(
    const int8_t* B_n32k16, const half* B_scale_n32, const half* B_zero_n32,
    int8_t* B_nk, half* B_scale, half* B_zero, const int N_32align, const int N,
    const int K, cudaStream_t stream);
template void restore_n32k16_weight_to_nk<hie::bfloat16>(
    const int8_t* B_n32k16, const hie::bfloat16* B_scale_n32,
    const hie::bfloat16* B_zero_n32, int8_t* B_nk, hie::bfloat16* B_scale,
    hie::bfloat16* B_zero, const int N_32align, const int N, const int K,
    cudaStream_t stream);
template void A_perc_symm_B_perc_asymm_dequantization<half>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const half* B_scale, const half* B_zero, const half* bias, half* result,
    const int M, const int N, cudaStream_t stream);
template void A_perc_symm_B_perc_asymm_dequantization<hie::bfloat16>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const hie::bfloat16* B_scale, const hie::bfloat16* B_zero,
    const hie::bfloat16* bias, hie::bfloat16* result, const int M, const int N,
    cudaStream_t stream);
template void A_perc_symm_B_perc_array_asymm_dequantization<half>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const half** B_scale, const half** B_zero, const half* bias, half* result,
    const int M, const int N, const int max_block, cudaStream_t stream);
template void A_perc_symm_B_perc_array_asymm_dequantization<hie::bfloat16>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const hie::bfloat16** B_scale, const hie::bfloat16** B_zero,
    const hie::bfloat16* bias, hie::bfloat16* result, const int M, const int N,
    const int max_block, cudaStream_t stream);
template void per_channel_symm_dynamic_quantization<half>(
    const half* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream);
template void per_channel_symm_dynamic_quantization<hie::bfloat16>(
    const hie::bfloat16* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark