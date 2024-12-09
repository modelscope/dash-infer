/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_sparse_a8w8_perc_kernel.cu
 */
#ifdef ENABLE_CUSPARSELT
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

// A = M x K
// require K % 8 == 0
template <int BLOCK, int UNROLL, typename FType>
__global__ void per_channel_symm_dynamic_quantization_kernel_vllm(
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
void per_channel_symm_dynamic_quantization_vllm(
    const FType* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream, bool is_prompt) {
  if (K % 8 != 0) {
    std::cerr << "[ERROR]: Now kernel only supprt K % 8 = 0" << std::endl;
  }

  cudaMemsetAsync((void*)red_max, 0,
                  aligned_size(M) * sizeof(float) +         // redmax set 0
                      aligned_size(M) * sizeof(uint32_t) +  // red_count set 0
                      aligned_size(M) * sizeof(int32_t),    // red_sum set 0
                  stream);

  const int BLOCK_MAX = 256;
  const int UNROLL_MAX = 8;
  int block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_per_sm,
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK_MAX, UNROLL_MAX,
                                                        FType>,
      BLOCK_MAX, 0);
  int max_reduce_len = BLOCK_MAX * UNROLL_MAX * 8 * block_per_sm * sm_count;

  // support for large M
  const int grid_y = M <= 65535 ? M : 65535;
  const int grid_z = M <= 65535 ? 1 : (M + 65535 - 1) / 65535;
  if (grid_z > 65535) {
    std::cerr << "Too large M is not supported!" << std::endl;
  }
  const int PACK = 8;
  if (K <= 2048) {
    const int BLOCK = 64;
    if (is_prompt) {
      const int UNROLL = 4;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 1;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 3072) {
    const int BLOCK = 64;
    if (is_prompt) {
      const int UNROLL = 6;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 4096) {
    const int BLOCK = 64;
    if (is_prompt) {
      const int UNROLL = 8;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 6144) {
    const int BLOCK = 64;
    if (is_prompt) {
      const int UNROLL = 12;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 3;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 8192) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 4;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 1;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 12288) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 6;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 13312) {
    const int BLOCK = 128;
    if (is_prompt) {
      const int UNROLL = 13;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 4;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 16384) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 8;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 20480) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 5;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 24576) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 6;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 3;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 32768) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 8;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 4;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 36864) {
    const int BLOCK = 256;
    if (is_prompt) {
      const int UNROLL = 9;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 5;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= 40960) {
    const int BLOCK = 512;
    if (is_prompt) {
      const int UNROLL = 5;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    } else {
      const int UNROLL = 2;
      const int grid_x =
          (K + BLOCK * UNROLL * PACK - 1) / (BLOCK * UNROLL * PACK);
      dim3 grid(grid_x, grid_y, grid_z);
      per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK, UNROLL, FType>
          <<<grid, BLOCK, 0, stream>>>(fdata, qdata, scale, red_max, red_count,
                                       red_sum, M, K);
    }
  } else if (K <= max_reduce_len) {
    const int grid_x =
        (K + BLOCK_MAX * UNROLL_MAX * 8 - 1) / (BLOCK_MAX * UNROLL_MAX * 8);
    dim3 grid(grid_x, grid_y, grid_z);
    per_channel_symm_dynamic_quantization_kernel_vllm<BLOCK_MAX, UNROLL_MAX,
                                                      FType>
        <<<grid, BLOCK_MAX, 0, stream>>>(fdata, qdata, scale, red_max,
                                         red_count, red_sum, M, K);
  } else {
    std::cerr << "[ERROR]: Now this kernel not support such a large K!"
              << std::endl;
  }
}

void I8GemmWraper(int32_t* matrix_C, const int8_t* matrix_A,
                  const int8_t* matrix_B, int m, int n, int k, bool transA,
                  bool transB, int lda, int ldb, int ldc, int32_t alpha,
                  int32_t beta, cublasHandle_t handle, cudaStream_t stream) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  /*
  if (bias) {
      broadcast_kernel_launcher(matrix_C, bias, m * n, n, 1, stream);
      beta = 1;
  }
  */
  cublasGemmEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B, CUDA_R_8I,
               ldb, matrix_A, CUDA_R_8I, lda, &beta, matrix_C, CUDA_R_32I, ldc,
               CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
void A_perc_symm_B_perc_asymm_dequantization_vllm(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType* B_scale, const FType* B_zero, const FType* bias, FType* result,
    const int M, const int N, cudaStream_t stream, bool is_prompt) {
  const int BLOCK = 256;
  const int PACK = 4;
  const int grid_x = (N + BLOCK * PACK - 1) / (BLOCK * PACK);
  if (N % PACK != 0) {
    std::cerr << "Now this kernel only support N % " << PACK << " == 0!"
              << std::endl;
  }

  if (is_prompt) {
    const int UNROLL = 4;
    const int grid_y = (M + UNROLL - 1) / UNROLL;
    dim3 grid(grid_x, grid_y);

    A_perc_symm_B_perc_asymm_dequantization_kernel<FType, BLOCK, UNROLL>
        <<<grid, BLOCK, 0, stream>>>(imd_result, A_scale, A_redsum, B_scale,
                                     B_zero, bias, result, M, N);
  } else {
    const int UNROLL = 1;
    const int grid_y = (M + UNROLL - 1) / UNROLL;
    dim3 grid(grid_x, grid_y);

    A_perc_symm_B_perc_asymm_dequantization_kernel<FType, BLOCK, UNROLL>
        <<<grid, BLOCK, 0, stream>>>(imd_result, A_scale, A_redsum, B_scale,
                                     B_zero, bias, result, M, N);
  }
}

/*
 * A8W8 is used for both prefill and decoder phase
 * param A_scale: model parameter, not used now
 * ********** For A symm perchannel + B asymm perchannel version
*********************
 * param workspace: consists of 6 part.
 *    1) Aq, shape = (M, K), int8_t
 *    2) A scale, shape = (M,), float.
 *    3) A red_max, shape = (M,), float. used to store reduce-max value for per
channel
 *    4) A red_count, shape = (M,), uint32_t. used to count the number of blocks
that complte reduce-max
 *    5) A red_sum, shape = (M,), int32_t. used for dequantize kernel
 *    6) imd_result, shape = (M, N), int32_t
//  * */
template <typename FType, typename QType>
void a8w8_gemm_cublas_qdq(const FType* Af, const FType* A_scale,
                          const QType* B_nk, const FType* B_scale,
                          const FType* B_zero, const FType* bias, FType* C,
                          void* workspace, const int M, const int N,
                          const int K, const int sm_count, bool is_prompt,
                          cudaStream_t stream, cublasHandle_t handle) {
  int8_t* Aq = reinterpret_cast<int8_t*>(workspace);
  float* A_scale_runtime =
      reinterpret_cast<float*>(Aq + aligned_size(M * K) * sizeof(int8_t));
  float* A_red_max = reinterpret_cast<float*>((char*)A_scale_runtime +
                                              aligned_size(M) * sizeof(float));
  uint32_t* A_red_count = reinterpret_cast<uint32_t*>(
      (char*)A_red_max + aligned_size(M) * sizeof(float));
  int32_t* A_red_sum = reinterpret_cast<int32_t*>(
      (char*)A_red_count + aligned_size(M) * sizeof(uint32_t));
  int32_t* imd_result = reinterpret_cast<int32_t*>(
      (char*)A_red_sum + aligned_size(M) * sizeof(int32_t));

  // step 1: quantize A tensor from FType to int8_t
  per_channel_symm_dynamic_quantization_vllm(Af, Aq, A_scale_runtime, A_red_max,
                                             A_red_count, A_red_sum, M, K,
                                             sm_count, stream, is_prompt);

  // step 2: call cuBLAS INT8 GEMM
  int lda = K;
  int ldb = K;
  int ldc = N;
  I8GemmWraper(imd_result, Aq, B_nk, M, N, K, false, true, lda, ldb, ldc, 1, 0,
               handle, stream);

  // step 3: dequant int32 imd_result to get FType C
  A_perc_symm_B_perc_asymm_dequantization_vllm(imd_result, A_scale_runtime,
                                               A_red_sum, B_scale, B_zero, bias,
                                               C, M, N, stream, is_prompt);
}

///////////////////////////// A8W8(Hopper) /////////////////////////////
template <typename FType>
void allspark_perc_qgemm_a8w8_gpu(const void* Af, const void* A_scale,
                                  const int8_t* B, const void* B_scale,
                                  const void* B_zero, const void* bias, void* C,
                                  void* workspace, const int M, const int N,
                                  const int K, const int sm_count,
                                  bool is_prompt, cudaStream_t stream,
                                  cublasHandle_t handle) {
  a8w8_gemm_cublas_qdq<FType, int8_t>(
      reinterpret_cast<const FType*>(Af),
      reinterpret_cast<const FType*>(A_scale), B,
      reinterpret_cast<const FType*>(B_scale),
      reinterpret_cast<const FType*>(B_zero),
      reinterpret_cast<const FType*>(bias), reinterpret_cast<FType*>(C),
      workspace, M, N, K, sm_count, is_prompt, stream, handle);
}

template <typename FType>
void allspark_perc_qgemm_sparse_a8w8_gpu(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle, cusparseLtHandle_t cslt_handle,
    cusparseLtMatmulPlan_t& plan) {
  int8_t* a_qdata = reinterpret_cast<int8_t*>(workspace);
  float* a_scale = reinterpret_cast<float*>(a_qdata + M * K * sizeof(int8_t));
  float* a_red_max =
      reinterpret_cast<float*>((char*)a_scale + M * sizeof(float));
  uint32_t* a_red_count =
      reinterpret_cast<uint32_t*>((char*)a_red_max + M * sizeof(float));
  int32_t* a_red_sum =
      reinterpret_cast<int32_t*>((char*)a_red_count + M * sizeof(uint32_t));
  int32_t* imd_result =
      reinterpret_cast<int32_t*>((char*)a_red_sum + M * sizeof(int32_t));
  void* cslt_workspace =
      reinterpret_cast<void*>((char*)imd_result + M * N * sizeof(int32_t));

  per_channel_symm_dynamic_quantization_vllm<FType>(
      (FType*)Af, a_qdata, a_scale, a_red_max, a_red_count, a_red_sum, M, K,
      sm_count, stream, is_prompt);

  float alpha = 1.0f, beta = 0.0f;
  cusparseLtMatmul(&cslt_handle, &plan, &alpha, B, a_qdata, &beta, imd_result,
                   imd_result, cslt_workspace, &stream, 1);

  A_perc_symm_B_perc_asymm_dequantization_vllm<FType>(
      imd_result, a_scale, a_red_sum, (FType*)B_scale, (FType*)B_zero,
      (FType*)bias, (FType*)C, M, N, stream, is_prompt);
}

template void allspark_perc_qgemm_sparse_a8w8_gpu<half>(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle, cusparseLtHandle_t cslt_handle,
    cusparseLtMatmulPlan_t& plan);
template void allspark_perc_qgemm_sparse_a8w8_gpu<hie::bfloat16>(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle, cusparseLtHandle_t cslt_handle,
    cusparseLtMatmulPlan_t& plan);
template void allspark_perc_qgemm_a8w8_gpu<half>(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle);
template void allspark_perc_qgemm_a8w8_gpu<hie::bfloat16>(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle);
template void per_channel_symm_dynamic_quantization_vllm<half>(
    const half* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream, bool is_prompt);
template void per_channel_symm_dynamic_quantization_vllm<hie::bfloat16>(
    const hie::bfloat16* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream, bool is_prompt);
template void A_perc_symm_B_perc_asymm_dequantization_vllm<half>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const half* B_scale, const half* B_zero, const half* bias, half* result,
    const int M, const int N, cudaStream_t stream, bool is_prompt);
template void A_perc_symm_B_perc_asymm_dequantization_vllm<hie::bfloat16>(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const hie::bfloat16* B_scale, const hie::bfloat16* B_zero,
    const hie::bfloat16* bias, hie::bfloat16* result, const int M, const int N,
    cudaStream_t stream, bool is_prompt);

}  // namespace cuda
}  // namespace allspark
#endif