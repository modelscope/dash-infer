/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_kernel.cpp
 */

#include "gemm_a16w8_kernel.h"

namespace allspark {
namespace cuda {
namespace GemmA16W8Launcher {

// get split-k params for A16W8 Fused GEMV kernel
int GetSplitKParams(const uint32_t M, const uint32_t N, const uint32_t K,
                    const uint32_t M_tile, const uint32_t N_tile,
                    const uint32_t K_tile, const uint32_t K_low_bound,
                    const float SPLIT_THRESHOLD, const int sm_count,
                    const int blocks_per_sm, SplitKParams& splitk_params) {
  int grid_x = (M + M_tile - 1) / M_tile;
  int grid_y = (N + N_tile - 1) / N_tile;
  int grid_z;

  const int blocks_per_wave = sm_count * blocks_per_sm;
  int n_slice;
  for (n_slice = 1; n_slice < K / K_low_bound; ++n_slice) {
    int n_block = grid_x * grid_y * n_slice;
    if (n_block >= blocks_per_wave * SPLIT_THRESHOLD &&
        (n_block % blocks_per_wave == 0 ||
         n_block % blocks_per_wave >= blocks_per_wave * 0.8)) {
      break;
    }
  }
  int k_slice = (K / n_slice) % K_tile == 0
                    ? K / n_slice
                    : K / n_slice / K_tile * K_tile + K_tile;
  grid_z = (K + k_slice - 1) / k_slice;

  splitk_params.EnableSplitK = true;
  splitk_params.SplitK = k_slice;
  return grid_z;
}

uint64_t GetWorkSpaceSize(const KernelType k_type, const uint32_t M,
                          const uint32_t N, const uint32_t K,
                          const int GroupSize, const int sm_count,
                          const int sm_version, SplitKParams& splitk_params) {
  switch (k_type) {
    case KernelType::A16W8_GEMV_SUBC_M1: {
      /* Go Through*/
    }
    case KernelType::A16W8_GEMV: {
      const uint32_t SplitK = 512;
      const uint64_t ws_size =
          (K + SplitK - 1) / SplitK * M * N * sizeof(uint16_t);
      return ws_size;
    }
    case KernelType::A16W8_GEMV_SUBC: {
      uint32_t M_tile = 32;
      uint32_t N_tile = 128;
      uint32_t K_tile = 16;

      const int blocks_per_sm = 7;
      const float SPLIT_THRESHOLD = 0.8;
      const uint32_t K_low_bound = 128;

      int grid_z = GetSplitKParams(M, N, K, M_tile, N_tile, K_tile, K_low_bound,
                                   SPLIT_THRESHOLD, sm_count, blocks_per_sm,
                                   splitk_params);
      return M * N * grid_z * sizeof(uint16_t);
    }
    case KernelType::Volta_A16W8_GEMM_SUBC_128x128x32: {
      /* Go Through*/
    }
    case KernelType::Volta_A16W8_GEMM_SUBC_128x128x32_SplitK: {
      // uint32_t grid_z = (K + splitk_params.SplitK - 1) /
      // splitk_params.SplitK;
      return M * N * VOLTA_GEMM_MAX_GRIDZ * sizeof(uint16_t);
    }
    case KernelType::Volta_A16W8_GEMM_PERC_32x128x32: {
      /* Same as next */
    }
    case KernelType::Volta_A16W8_GEMM_SUBC_32x128x32: {
      // 4 thread block per sm, limited by register and shared memory
      const int blocks_per_sm = 4;
      const float SPLIT_THRESHOLD = 0.8;
      const uint32_t K_low_bound = 256;

      int grid_z =
          GetSplitKParams(M, N, K, 32, 128, 32, K_low_bound, SPLIT_THRESHOLD,
                          sm_count, blocks_per_sm, splitk_params);
      return M * N * grid_z * sizeof(uint16_t);
    }
    case KernelType::Ampere_A16W8_GEMM_PERC_64x128x32: {
      // 4 thread block per sm, limited by register and shared memory
      const int blocks_per_sm = 4;
      const float SPLIT_THRESHOLD = 0.9;
      const uint32_t K_low_bound = 256;

      int grid_z =
          GetSplitKParams(M, N, K, 64, 128, 32, K_low_bound, SPLIT_THRESHOLD,
                          sm_count, blocks_per_sm, splitk_params);
      return M * N * grid_z * sizeof(uint16_t);
    }
    case KernelType::A16W8_GEMM_SUBC_16816: {
      constexpr float SPLIT_THRESHOLD = 4;
      constexpr uint32_t BLOCK_TILE_M = 32;
      constexpr uint32_t BLOCK_TILE_N = 128;
      constexpr uint32_t BLOCK_TILE_K = 32;
      uint32_t grid_x = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
      uint32_t grid_y = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
      uint32_t grid_z;

      uint32_t num_slice;
      for (num_slice = 1; num_slice < K / 256; ++num_slice) {
        uint32_t num_block = grid_x * grid_y * num_slice;
        if (num_block > sm_count * SPLIT_THRESHOLD &&
            (num_block % sm_count == 0 ||
             num_block % sm_count >= sm_count / 2)) {
          break;
        }
      }
      const uint32_t SplitK =
          (K / num_slice) % GroupSize == 0
              ? (K / num_slice)
              : (K / num_slice / GroupSize * GroupSize + GroupSize);
      grid_z = (K + SplitK - 1) / SplitK;
      return M * N * grid_z * sizeof(uint16_t);
    }
    case KernelType::A16W8_GEMM: {
      /* Go Through*/
    }
    case KernelType::A16W8_GEMM_SUBC: {
      // As have bug. When ws is zero will cause error.
      // So here ws is setted M * N.
      return M * N * sizeof(uint16_t);
    }
    default: {
      // TODO:
      break;
    }
  }
  return 0;
}

}  // namespace GemmA16W8Launcher
}  // namespace cuda
}  // namespace allspark