/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_kernel.cpp
 */

#include "gemm_a16w4_kernel.h"

namespace allspark {
namespace cuda {
namespace GemmA16W4Launcher {

// get split-k params for A16W4 Fused GEMV kernel
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
    case KernelType::UNDEFINE: {
      uint64_t ws_size = K * N * sizeof(uint16_t);
      return ws_size;
    }
    case KernelType::Volta_A16W4_GEMV_PERC_32x256x32: {
      uint32_t M_tile = 32;
      uint32_t N_tile = 256;
      uint32_t K_tile = 32;

      const int blocks_per_sm = 4;
      const float SPLIT_THRESHOLD = 0.9;
      const uint32_t K_low_bound = 256;

      int grid_z = GetSplitKParams(M, N, K, M_tile, N_tile, K_tile, K_low_bound,
                                   SPLIT_THRESHOLD, sm_count, blocks_per_sm,
                                   splitk_params);
      return aligned_size(M * N * grid_z) * sizeof(uint16_t)  // for C_split
             + aligned_size(M * grid_z) * sizeof(float)       // for A_redsum
             + aligned_size(1) * sizeof(int);  // for reduce count
    }
    case KernelType::Volta_A16W4_GEMV_SUBC_32x256x32: {
      uint32_t M_tile = 32;
      uint32_t N_tile = 256;
      uint32_t K_tile = 32;

      const int blocks_per_sm = 4;
      const float SPLIT_THRESHOLD = 0.9;
      const uint32_t K_low_bound = 256;

      int grid_z = GetSplitKParams(M, N, K, M_tile, N_tile, K_tile, K_low_bound,
                                   SPLIT_THRESHOLD, sm_count, blocks_per_sm,
                                   splitk_params);

      return M * N * grid_z * sizeof(uint16_t);  // for C_split
    }
    case KernelType::A16W4_GEMV_PERC_16816: {
      const float SPLIT_THRESHOLD = 3;
      constexpr uint32_t BLOCK_TILE_M = 32;
      constexpr uint32_t BLOCK_TILE_N = 256;
      constexpr uint32_t BLOCK_TILE_K = 32;

      uint32_t grid_x = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
      uint32_t grid_y = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
      uint32_t grid_z;

      uint32_t num_slice;
      for (num_slice = 1; num_slice < K / 64; ++num_slice) {
        uint32_t num_block = grid_x * grid_y * num_slice;
        if (num_block > sm_count * SPLIT_THRESHOLD &&
            (num_block % sm_count == 0 ||
             num_block % sm_count >= sm_count / 2)) {
          break;
        }
      }
      const uint32_t SplitK =
          (K / num_slice) % BLOCK_TILE_K == 0
              ? (K / num_slice)
              : (K / num_slice / BLOCK_TILE_K * BLOCK_TILE_K + BLOCK_TILE_K);
      grid_z = (K + SplitK - 1) / SplitK;
      return M * N * grid_z * sizeof(uint16_t);
    }
    case KernelType::A16W4_GEMV_SUBC_16816: {
      constexpr uint32_t M_tile = 32;
      constexpr uint32_t N_tile = 256;
      constexpr uint32_t K_tile = 32;

      const int blocks_per_sm = 3;
      const float SPLIT_THRESHOLD = 0.8;
      const uint32_t K_low_bound = 128;

      int grid_z = GetSplitKParams(M, N, K, M_tile, N_tile, K_tile, K_low_bound,
                                   SPLIT_THRESHOLD, sm_count, blocks_per_sm,
                                   splitk_params);
      return M * N * grid_z * sizeof(uint16_t);  // for C_split
    }
    default: {
      // TODO:
      break;
    }
  }
  return uint64_t(0);
}

}  // namespace GemmA16W4Launcher
}  // namespace cuda
}  // namespace allspark