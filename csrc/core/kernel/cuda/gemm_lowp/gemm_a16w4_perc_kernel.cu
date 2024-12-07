/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_perc_kernel.cu
 */

#include <utility/check_cuda.h>

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

template <typename FT, typename QT>
__global__ void dequantize_rhs_a16w4_kernel(const QT* data_in, const FT* scales,
                                            const FT* zeros, FT* data_out,
                                            const int K, const int N,
                                            const int NPack) {
  const int k_idx = blockIdx.y;
  const int n_pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_idx = n_pack_idx * 2;
  if (n_pack_idx < NPack && k_idx < K) {
    FT s_reg[2]{};
    FT z_reg[2]{};
    FT fdata[2]{};
// Load
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if ((n_idx + i) < N) {
        s_reg[i] = scales[n_pack_idx * 2 + i];
        z_reg[i] = zeros[n_pack_idx * 2 + i];
      }
    }
    const QT idata = data_in[k_idx * NPack + n_pack_idx];
    // CVT + DQ
    cvt_4bx2_to_16bx2<FT, QT>::cvt(idata, fdata);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      fdata[i] = (fdata[i] - z_reg[i]) * s_reg[i];
    }
// Store
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if ((n_idx + i) < N) {
        data_out[k_idx * N + n_idx + i] = fdata[i];
      }
    }
  }
}

// Now only support UNROLL = 8
// QT is uint8, contains 2 4-bit unsigned
// N % 2 == 0
template <typename FT, typename QT, int BLOCK_SIZE, int UNROLL>
__global__ void dequantize_rhs_a16w4_kernel_opt(
    const QT* qdata_pack, const FT* scales, const FT* zeros, FT* fdata,
    const uint32_t N, PackedEltwiseConfig packConfig) {
  size_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < packConfig.nPack) {
    uint32_t qval_reg;
    typename HalfType<FT>::T2 s_reg[UNROLL / 2];
    typename HalfType<FT>::T2 z_reg[UNROLL / 2];
    typename HalfType<FT>::T2 fval_reg[UNROLL / 2];

    qval_reg = *(reinterpret_cast<const uint32_t*>(qdata_pack) + idx);
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      const uint32_t offset = (idx * UNROLL + i * 2) % N / 2;
      s_reg[i] =
          reinterpret_cast<const typename HalfType<FT>::T2*>(scales)[offset];
      z_reg[i] =
          reinterpret_cast<const typename HalfType<FT>::T2*>(zeros)[offset];
    }
// UnPack
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      QT val = reinterpret_cast<QT*>(&qval_reg)[i];
      cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(val, fval_reg[i]);
    }
// DQ
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      fval_reg[i] = dequantize_func<typename HalfType<FT>::T2>(
          fval_reg[i], s_reg[i], z_reg[i]);
    }
    // Store
    stg128(*reinterpret_cast<uint32_t*>(fval_reg),
           *reinterpret_cast<uint32_t*>(fval_reg + 1),
           *reinterpret_cast<uint32_t*>(fval_reg + 2),
           *reinterpret_cast<uint32_t*>(fval_reg + 3),
           reinterpret_cast<int4*>(fdata) + idx);
  } else if (idx <
             packConfig.nPack + (packConfig.nThread - packConfig.nPack) / 2) {
    // One thread process 2x4bit
    idx = packConfig.nPack * UNROLL + (idx - packConfig.nPack) * 2;

    const int param_offset = idx % N / 2;
    QT qval_reg = qdata_pack[idx / 2];
    typename HalfType<FT>::T2 fval_reg;
    typename HalfType<FT>::T2 s_reg =
        reinterpret_cast<const typename HalfType<FT>::T2*>(
            scales)[param_offset];
    typename HalfType<FT>::T2 z_reg =
        reinterpret_cast<const typename HalfType<FT>::T2*>(zeros)[param_offset];

    cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(qval_reg, fval_reg);
    fval_reg =
        dequantize_func<typename HalfType<FT>::T2>(fval_reg, s_reg, z_reg);

    fdata[idx] = reinterpret_cast<FT*>(&fval_reg)[0];
    fdata[idx + 1] = reinterpret_cast<FT*>(&fval_reg)[1];
  }
}

// Now only support UNROLL = 8
// QT is uint8, contains 2 4-bit unsigned
// N % 2 == 0
template <typename FT, typename QT, int BLOCK, int UNROLL>
__global__ void dequantize_rhs_a16w4_subc_kernel_opt(
    const QT* qdata, const FT* scales, const FT* zeros, FT* fdata,
    PackedEltwiseConfig packConfig, const uint32_t N, const int GroupSize) {
  static_assert(std::is_same<QT, uint8_t>::value, "Now only support uint8");
  int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
  if (idx < packConfig.nPack) {
    uint32_t qval_reg;
    typename HalfType<FT>::T2 s_reg[UNROLL / 2];
    typename HalfType<FT>::T2 z_reg[UNROLL / 2];
    typename HalfType<FT>::T2 fval_reg[UNROLL / 2];

    qval_reg = *(reinterpret_cast<const uint32_t*>(qdata) + idx);
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      const int n_idx = (idx * UNROLL + i * 2) % N;
      const int qparam_k_idx = ((idx * UNROLL + i * 2) / N) / GroupSize;
      const int offset = qparam_k_idx * N + n_idx;
      s_reg[i] = reinterpret_cast<const typename HalfType<FT>::T2*>(
          scales)[offset / 2];
      z_reg[i] =
          reinterpret_cast<const typename HalfType<FT>::T2*>(zeros)[offset / 2];
    }
// UnPack
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      QT val = reinterpret_cast<QT*>(&qval_reg)[i];
      cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(val, fval_reg[i]);
    }
// DQ
#pragma unroll
    for (int i = 0; i < UNROLL / 2; ++i) {
      fval_reg[i] = dequantize_func<typename HalfType<FT>::T2>(
          fval_reg[i], s_reg[i], z_reg[i]);
    }
    // Store
    stg128(*reinterpret_cast<uint32_t*>(fval_reg),
           *reinterpret_cast<uint32_t*>(fval_reg + 1),
           *reinterpret_cast<uint32_t*>(fval_reg + 2),
           *reinterpret_cast<uint32_t*>(fval_reg + 3),
           reinterpret_cast<int4*>(fdata) + idx);
  } else if (idx <
             packConfig.nPack + (packConfig.nThread - packConfig.nPack) / 2) {
    // One thread process 2x4bit
    const int offset = packConfig.nPack * UNROLL + (idx - packConfig.nPack) * 2;
    QT qval_reg = qdata[offset / 2];

    const int qparam_k_idx = (offset / N) / GroupSize;
    const int n_idx = offset % N;
    int param_offset = qparam_k_idx * N + n_idx;
    typename HalfType<FT>::T2 fval_reg;
    typename HalfType<FT>::T2 s_reg =
        reinterpret_cast<const typename HalfType<FT>::T2*>(
            scales)[param_offset / 2];
    typename HalfType<FT>::T2 z_reg =
        reinterpret_cast<const typename HalfType<FT>::T2*>(
            zeros)[param_offset / 2];

    cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(qval_reg, fval_reg);
    fval_reg =
        dequantize_func<typename HalfType<FT>::T2>(fval_reg, s_reg, z_reg);

    fdata[offset] = reinterpret_cast<FT*>(&fval_reg)[0];
    fdata[offset + 1] = reinterpret_cast<FT*>(&fval_reg)[1];
  }
}

template <typename FT, typename QT>
__global__ void dequantize_rhs_a16w4_subc_kernel(
    const QT* data_in, const FT* scales, const FT* zeros, FT* data_out,
    const int K, const int N, const int NPack, const int GroupSize) {
  const int k_idx = blockIdx.y;
  const int n_pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_idx = n_pack_idx * 2;
  if (n_pack_idx < NPack && k_idx < K) {
    FT s_reg[2]{};
    FT z_reg[2]{};
    FT fdata[2]{};
    // Load
    const int qparam_kidx = k_idx / GroupSize;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if ((n_idx + i) < N) {
        s_reg[i] = scales[qparam_kidx * N + n_idx + i];
        z_reg[i] = zeros[qparam_kidx * N + n_idx + i];
      }
    }
    const QT idata = data_in[k_idx * NPack + n_pack_idx];
    // CVT + DQ
    cvt_4bx2_to_16bx2<FT, QT>::cvt(idata, fdata);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      fdata[i] = (fdata[i] - z_reg[i]) * s_reg[i];
    }
// Store
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if ((n_idx + i) < N) {
        data_out[k_idx * N + n_idx + i] = fdata[i];
      }
    }
  }
}

template <typename FT, typename QT>
void dequantize_rhs_a16w4(const QT* qdata, const FT* scales, const FT* zeros,
                          FT* fdata, const uint32_t N, const uint32_t NPack,
                          const uint32_t K, const int GroupSize,
                          cudaStream_t stream) {
  if (GroupSize == -1) {
    if (NPack * 2 != N) {
      const uint32_t THREADS = 256;
      const dim3 BLOCK_DIMS = dim3(DivCeil(NPack, THREADS), K, 1);
      dequantize_rhs_a16w4_kernel<FT, QT><<<BLOCK_DIMS, THREADS, 0, stream>>>(
          qdata, scales, zeros, fdata, K, N, NPack);
    } else {
      int packSize = std::min(GetPackSize(qdata), GetPackSize(fdata));
      const int64_t BLOCK_SIZE = 128;
      PackedEltwiseConfig packConfig(N * K, packSize, BLOCK_SIZE);
      switch (packSize) {
        case 8:
          dequantize_rhs_a16w4_kernel_opt<FT, QT, BLOCK_SIZE, 8>
              <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
                  qdata, scales, zeros, fdata, N, packConfig);
          break;
        default:
          LOG(ERROR) << "Now only support out ptr is 16-byte aligned";
          break;
      }
    }
  } else {
    if (NPack * 2 != N) {
      const uint32_t THREADS = 256;
      const dim3 BLOCK_DIMS = dim3(DivCeil(NPack, THREADS), K, 1);
      dequantize_rhs_a16w4_subc_kernel<FT, QT>
          <<<BLOCK_DIMS, THREADS, 0, stream>>>(qdata, scales, zeros, fdata, K,
                                               N, NPack, GroupSize);
    } else {
      int packSize = std::min(GetPackSize(qdata), GetPackSize(fdata));
      const int64_t BLOCK_SIZE = 128;
      PackedEltwiseConfig packConfig(N * K, packSize, BLOCK_SIZE);

      switch (packSize) {
        case 8:
          dequantize_rhs_a16w4_subc_kernel_opt<FT, QT, BLOCK_SIZE, 8>
              <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
                  qdata, scales, zeros, fdata, packConfig, N, GroupSize);
          break;
        default:
          LOG(ERROR) << "Now only support out ptr is 16-byte aligned";
          break;
      }
    }
  }
}

/**
 * @brief
 *
 * 32x256x32_32x32_16816_NN_SPLITK_A16W4_PERC
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
__global__ void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk_kernel(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev, FT* Cdev,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t SplitK) {
  constexpr uint32_t BLOCK_TILE_M = 32;
  constexpr uint32_t BLOCK_TILE_N = 256;
  constexpr uint32_t BLOCK_TILE_K = 32;
  // constexpr uint32_t WARP_TILE_M = 32;
  // constexpr uint32_t WARP_TILE_N = 32;
  // constexpr uint32_t WARP_TILE_K = 16;

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

  // Load Quantize params
  typename HalfType<FT>::T2 BS[4];
  typename HalfType<FT>::T2 BZ[4];
  ldg128_cg<typename HalfType<FT>::T2>(BS[0], BS[1], BS[2], BS[3], BSdev_ptr,
                                       B_ldg_guard);
  ldg128_cg<typename HalfType<FT>::T2>(BZ[0], BZ[1], BZ[2], BZ[3], BZdev_ptr,
                                       B_ldg_guard);

  int4 ldg_a_reg;
  uint32_t ldg_b_reg[4];
  typename HalfType<FT>::T2 sts_b_reg[4][4];

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

    // STS A from reg to smem
    if (warp_id < 4) {
      sts128<int32_t>(ldg_a_reg.x, ldg_a_reg.y, ldg_a_reg.z, ldg_a_reg.w,
                      A_sts_addr);
    }

// CVT + DQ
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const QT i4x2_val = reinterpret_cast<QT*>(&ldg_b_reg[i])[j];
        // Convert 2x4bit->2xFT
        cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(i4x2_val,
                                                              sts_b_reg[i][j]);
        // DQ
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
  for (; k_tile_num > 0; --k_tile_num) {
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
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const QT i4x2_val = reinterpret_cast<QT*>(&ldg_b_reg[i])[j];
        // Convert 2x4bit->2xFT
        cvt_4bx2_to_16bx2<typename HalfType<FT>::T2, QT>::cvt(i4x2_val,
                                                              sts_b_reg[i][j]);
        // DQ
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
void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const int sm_count,
    cudaStream_t stream) {
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
        (num_block % sm_count == 0 || num_block % sm_count >= sm_count / 2)) {
      break;
    }
  }
  const uint32_t SplitK =
      (K / num_slice) % BLOCK_TILE_K == 0
          ? (K / num_slice)
          : (K / num_slice / BLOCK_TILE_K * BLOCK_TILE_K + BLOCK_TILE_K);
  grid_z = (K + SplitK - 1) / SplitK;
  {
    dim3 block(256);
    dim3 grid(grid_x, grid_y, grid_z);
    hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk_kernel<FT, QT>
        <<<grid, block, 0, stream>>>(Adev, Bdev, BSdev, BZdev,
                                     static_cast<FT*>(workspace), M, N, K,
                                     SplitK);
  }
  {
    const uint32_t THREADS_PER_BLOCK = 128;
    const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), M, 1);
    reduce_sum<FT, ActiveFunc><<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<const FT*>(workspace), bias, Cdev, M, N, K,
        DivCeil(K, SplitK));
  }
}

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * QType is uint8, contains 2 4-bit unsigned
 *
 */
template <int BLOCK_SIZE, typename QType>
struct GmemTile_Perc_A16W4_32x256x32_SM70_SplitK {
  static_assert(std::is_same<QType, uint8_t>::value,
                "A16W4 GmemTile only support uint8");
  // element num loaded by a LDG inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 8;
  const int WARP_CNT = BLOCK_SIZE / WARP_SIZE;  // 4

  __device__ GmemTile_Perc_A16W4_32x256x32_SM70_SplitK(
      const SM70_GEMM_A16W4_Params<half, uint8_t>& k_params,
      const uint32_t& A_smem_base_addr, const uint32_t& B_smem_base_addr,
      const int& tile_xidx, const int& tile_yidx, const int& tile_zidx)
      : params(k_params),
        A_smem_addr(A_smem_base_addr),
        B_smem_addr(B_smem_base_addr),
        tile_x(tile_xidx),
        tile_y(tile_yidx),
        tile_z(tile_zidx) {
    this_block_A_base_ptr =
        params.A_ptr + tile_x * 32 * params.K + tile_z * params.SplitK;
    ;
    this_block_B_base_ptr =
        params.B_ptr + (tile_y * 256 + tile_z * params.SplitK * params.N) / 2;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // For matrix A, a block load/store 32(row) x 32(col) elements in 1 iter
    // 8x4 warp load/store 8(row) x 32(col) elements
    const int load_a_row_idx = threadIdx.x / 4;
    load_a_col_idx = threadIdx.x % 4 * LDG_ELEMENT_CNT_A;
    load_a_offset = load_a_row_idx * params.K + load_a_col_idx;

    // For matrix B, a block load/store 32(row) * 256(col) elements in 8 iter
    // and a block load/store 4(row) * 256(col) elements per iter, a warp
    // load/store 1(row) * 256(col) per iter
    const int load_b_col_idx = lane_id * LDG_ELEMENT_CNT_B;
    load_b_row_base_idx = warp_id;
    ;
    load_b_base_offset = load_b_row_base_idx * params.N + load_b_col_idx;

    store_a_offset =
        threadIdx.x / 4 * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;

    store_b_base_offset =
        warp_id * 256 + (lane_id ^ (warp_id * 2)) * LDG_ELEMENT_CNT_B;

    A_ldg_guard = false;
    int m_idx = tile_x * 32 + load_a_row_idx;
    if (m_idx < params.M) {
      A_ldg_guard = true;
    }

    B_ldg_guard = false;
    n_idx = tile_y * 256 + load_b_col_idx;
    if (n_idx < params.N) {
      B_ldg_guard = true;
    }
  }

  __device__ void ldg_first_two_k_tile(const int& first_k_tile,
                                       const int tb_k_slice) {
// load first k_tile
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * 4 * params.N;
      bool guard = B_ldg_guard && (load_b_row_base_idx + i * 4) < first_k_tile;
      ldg32_cs_0(b_regs_q_now[i], this_block_B_base_ptr + (load_b_offset / 2),
                 guard);
    }

    bool guard = A_ldg_guard && load_a_col_idx < first_k_tile;
    ldg128_cg_0(a_regs_now[0], a_regs_now[1], a_regs_now[2], a_regs_now[3],
                this_block_A_base_ptr + load_a_offset, guard);

    // switch to next 32
    this_block_A_base_ptr += first_k_tile;
    this_block_B_base_ptr += first_k_tile * params.N / 2;

// load second k_tile
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * 4 * params.N;
      bool guard = B_ldg_guard &&
                   (load_b_row_base_idx + first_k_tile + i * 4) < tb_k_slice;
      ldg32_cs_0(b_regs_q_next[i], this_block_B_base_ptr + (load_b_offset / 2),
                 guard);
    }
    guard = A_ldg_guard && (load_a_col_idx + first_k_tile) < tb_k_slice;
    ldg128_cg_0(a_regs_next[0], a_regs_next[1], a_regs_next[2], a_regs_next[3],
                this_block_A_base_ptr + load_a_offset, guard);

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N / 2;
  }

  __device__ void ldg() {
// load B
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int load_b_offset = load_b_base_offset + i * 4 * params.N;
      ldg32_cs(b_regs_q_next[i], this_block_B_base_ptr + (load_b_offset / 2),
               B_ldg_guard);
    }

    // load A
    ldg128_cg(a_regs_next[0], a_regs_next[1], a_regs_next[2], a_regs_next[3],
              this_block_A_base_ptr + load_a_offset, A_ldg_guard);

    // switch to next 32
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += 32 * params.N / 2;
  }

  __device__ void switch_now_next() {
    for (int i = 0; i < 8; ++i) {
      b_regs_q_now[i] = b_regs_q_next[i];
    }
    for (int i = 0; i < 4; ++i) {
      a_regs_now[i] = a_regs_next[i];
    }
  }

  __device__ void dq() {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      b_regs[i][0].x = static_cast<half>(b_regs_q_now[i] & 0xf);
      b_regs[i][0].y = static_cast<half>((b_regs_q_now[i] >> 4) & 0xf);
      b_regs[i][1].x = static_cast<half>((b_regs_q_now[i] >> 8) & 0xf);
      b_regs[i][1].y = static_cast<half>((b_regs_q_now[i] >> 12) & 0xf);
      b_regs_q_now[i] = b_regs_q_now[i] >> 16;
      b_regs[i][2].x = static_cast<half>(b_regs_q_now[i] & 0xf);
      b_regs[i][2].y = static_cast<half>((b_regs_q_now[i] >> 4) & 0xf);
      b_regs[i][3].x = static_cast<half>((b_regs_q_now[i] >> 8) & 0xf);
      b_regs[i][3].y = static_cast<half>((b_regs_q_now[i] >> 12) & 0xf);
    }
  }

  __device__ void sts() {
    // store A from reg to smem
    sts128(a_regs_now[0], a_regs_now[1], a_regs_now[2], a_regs_now[3],
           A_smem_addr + store_a_offset * 2 /* ELEM_SIZE */);

// dequant and store B from reg to smem
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sts128(b_regs[i][0], b_regs[i][1], b_regs[i][2], b_regs[i][3],
             B_smem_addr +
                 (store_b_base_offset + i * 4 * 256) * 2 /* ELEM_SIZE */);
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

  uint32_t a_regs_now[4];
  uint32_t a_regs_next[4];

  __half2 b_regs[8][4];
  uint32_t b_regs_q_now[8];
  uint32_t b_regs_q_next[8];

  const uint32_t A_smem_addr, B_smem_addr;
  const int tile_x, tile_y, tile_z;
  const SM70_GEMM_A16W4_Params<half, QType>& params;
};

/*
 * warp_tile : m32n64k32
 */
template <typename QType, typename RedsumType>
struct ComputeTile_Perc_f16_32x256x32_SM70_SplitK {
  __device__ ComputeTile_Perc_f16_32x256x32_SM70_SplitK(
      const SM70_GEMM_A16W4_Params<half, QType>& k_params,
      const uint32_t& A_smem_base_addr, const uint32_t& B_smem_base_addr,
      const int& tile_xidx, const int& tile_yidx, const int& tile_zidx)
      : params(k_params),
        A_smem_addr(A_smem_base_addr),
        B_smem_addr(B_smem_base_addr),
        tile_x(tile_xidx),
        tile_y(tile_yidx),
        tile_z(tile_zidx) {
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

    this_block_C_base_ptr = params.C_split_ptr + tile_z * params.M * params.N +
                            tile_x * 32 * params.N + tile_y * 256;
  }

  // load 32 * 8 A elements per warp per k_phase
  // the threads in the same warp have double the data read repeatedly
  __device__ void lds_A(const int k_phase_idx) {
    int load_a_offset =
        load_a_base_offset + (k_phase_idx ^ A_kphase_col_adjust) * 8;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lds128(A_frag[i][0], A_frag[i][1], A_frag[i][2], A_frag[i][3],
             A_smem_addr + (load_a_offset + i * 16 * 32) * 2 /* ELEM SIZE */);
    }
  }

  // load 8 * 64 B elements per warp per k_phase in two steps
  // the threads in the same warp have double the data read repeatedly
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

  __device__ void ldg_qparams(RedsumType* A_redsum) {
#pragma unroll
    for (int st_iter_n = 0; st_iter_n < 4; ++st_iter_n) {
      int n_idx = tile_y * 256 + store_c_col_base_idx + st_iter_n * 16;
      bool guard = n_idx < params.N;
      ldg128_ca_0(b_regs_zero[st_iter_n][0], b_regs_zero[st_iter_n][1],
                  b_regs_zero[st_iter_n][2], b_regs_zero[st_iter_n][3],
                  params.B_zero_ptr + n_idx, guard);
    }
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 2; st_iter_m++) {
      int m_idx = tile_x * 32 + store_c_row_base_idx + st_iter_m * 16;
      bool guard =
          (tile_x * 32 + store_c_row_base_idx + st_iter_m * 16) < params.M;
      if (guard) {
        a_redsum[st_iter_m] = *(A_redsum + tile_z * params.M + m_idx);
      }
    }
  }

  __device__ void stg() {
#pragma unroll
    for (int st_iter_m = 0; st_iter_m < 2; st_iter_m++) {
      for (int st_iter_n = 0; st_iter_n < 4; st_iter_n++) {
        for (int i = 0; i < 4; ++i) {
          __half2 tmp =
              reinterpret_cast<__half2&>(C_frag[st_iter_m][st_iter_n][i]);
          tmp = __hfma2(b_regs_zero[st_iter_n][i],
                        __half2half2(static_cast<half>(
                            -1.f * (float)a_redsum[st_iter_m])),
                        tmp);
          C_frag[st_iter_m][st_iter_n][i] = reinterpret_cast<uint32_t&>(tmp);
        }
        half* C_ptr = this_block_C_base_ptr + store_c_base_offset +
                      st_iter_m * 16 * params.N + st_iter_n * 16;
        bool guard =
            (tile_x * 32 + store_c_row_base_idx + st_iter_m * 16) < params.M &&
            (tile_y * 256 + store_c_col_base_idx + st_iter_n * 16) < params.N;
        stg128(C_frag[st_iter_m][st_iter_n][0], C_frag[st_iter_m][st_iter_n][1],
               C_frag[st_iter_m][st_iter_n][2], C_frag[st_iter_m][st_iter_n][3],
               C_ptr, guard);
      }
    }
  }

  const SM70_GEMM_A16W4_Params<half, QType>& params;
  const int tile_x, tile_y, tile_z;

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

  __half2 b_regs_zero[4][4];
  RedsumType a_redsum[2];

  int A_kphase_col_adjust;
};

template <typename T>
__device__ T ReduceWarp(T val) {
#pragma unroll
  for (int i = WARP_SIZE; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = tmp + val;
  }
  return val;
}

// A: m x k
// gird_x: ceilDiv(m, 32), grid_y: ceilDiv(k, SplitK)
// A_redsum: grid_y x m, m continuous
template <typename QType, typename RedsumType = float>
__device__ void Amatrix_row_reduce_sum(
    const SM70_GEMM_A16W4_Params<half, QType>& params, const int& tb_k_slice,
    const int& red_x, const int& red_y, RedsumType* A_redsum, int* count_ptr) {
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int src_base_offset = red_x * 32 * params.K + red_y * params.SplitK +
                              lane_id + warp_id * params.K;
  const int dst_base_offset = red_y * params.M + red_x * 32 + warp_id;

  const int iter = (tb_k_slice + WARP_SIZE - 1) / WARP_SIZE;

  for (int outer = 0; outer < 32 / 4; ++outer) {
    RedsumType sum = 0.f;
    if ((red_x * 32 + warp_id + outer * 4) < params.M) {
      for (int inner = 0; inner < iter; ++inner) {
        int src_offset =
            src_base_offset + outer * 4 * params.K + inner * WARP_SIZE;
        half val = 0.f;
        if ((inner * WARP_SIZE + lane_id) < tb_k_slice) {
          val = *(params.A_ptr + src_offset);
        }
        sum += RedsumType(val);
      }
      sum = ReduceWarp(sum);
      if (lane_id == 0) {
        int dst_offset = dst_base_offset + outer * 4;
        A_redsum[dst_offset] = sum;
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(count_ptr, 1);
  }
}

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
 */
template <typename QType, typename RedsumType>
__global__ void __launch_bounds__(128, 4)
    volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk_kernel(
        const SM70_GEMM_A16W4_Params<half, QType> params, const int grid_x,
        const int grid_y, const int grid_z, U32DivMod yz_divmod,
        U32DivMod z_divmod, RedsumType* A_redsum, int* count_ptr) {
  // the first grid_x * grid_z blocks compute the split-k row redsum for matrix
  // A the result is used for subsequent block' dequantization
  if (blockIdx.x < grid_x * grid_z) {
    auto z_dm = z_divmod.DivMod(blockIdx.x);
    const int red_x = z_dm.div;
    const int red_y = z_dm.mod;
    int tb_k_slice = red_y * params.SplitK + params.SplitK <= params.K
                         ? params.SplitK
                         : params.K - red_y * params.SplitK;
    Amatrix_row_reduce_sum<QType, RedsumType>(params, tb_k_slice, red_x, red_y,
                                              A_redsum, count_ptr);
    return;
  }

  // A smem size = 32 * 32 * 2B/elem = 4KB
  // B smem size = 256 * 32 * 2B/elem = 16KB
  static constexpr int SMEM_SIZE = 32 * 32 * 2 + 256 * 32 * 2;
  __shared__ char smem[SMEM_SIZE];
  char* A_smem = smem;
  char* B_smem = smem + 32 * 32 * 2;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t B_smem_addr = smem_u32addr(B_smem);

  auto yz_dm = yz_divmod.DivMod(blockIdx.x - grid_x * grid_z);
  const int tile_x = yz_dm.div;
  const int tile_y = z_divmod.Div(yz_dm.mod);
  const int tile_z = z_divmod.Mod(blockIdx.x - grid_x * grid_z);

  // initialize the data move process from GM to SMEM for this block
  GmemTile_Perc_A16W4_32x256x32_SM70_SplitK<128, QType> gmem_tile(
      params, A_smem_addr, B_smem_addr, tile_x, tile_y, tile_z);

  int tb_k_slice = tile_z * params.SplitK + params.SplitK <= params.K
                       ? params.SplitK
                       : params.K - tile_z * params.SplitK;

  int k_main_loop = (tb_k_slice + 31) / 32 - 1;
  int first_k_tile = tb_k_slice - k_main_loop * 32;

  // load 1'st tile to shared memory
  gmem_tile.ldg_first_two_k_tile(first_k_tile, tb_k_slice);
  ComputeTile_Perc_f16_32x256x32_SM70_SplitK<QType, RedsumType> compute_tile(
      params, A_smem_addr, B_smem_addr, tile_x, tile_y, tile_z);
  gmem_tile.dq();
  gmem_tile.sts();

  for (int k_tile_idx = 0; k_tile_idx < k_main_loop - 1; k_tile_idx++) {
    __syncthreads();
    gmem_tile.switch_now_next();
    gmem_tile.ldg();
    gmem_tile.dq();

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
      compute_tile.lds_A(k_phase_idx);
      compute_tile.lds_B(k_phase_idx, 0);
      compute_tile.mma(0);

      compute_tile.lds_B(k_phase_idx, 1);
      compute_tile.mma(1);
    }

    __syncthreads();
    gmem_tile.sts();
  }

  int count;
  do {
    // make sure the ld.cg inside the do-wile loop
    __threadfence_block();
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(count) : "l"(count_ptr));
  } while (count != grid_x * grid_z);

  compute_tile.ldg_qparams(A_redsum);

  __syncthreads();
  gmem_tile.switch_now_next();
  gmem_tile.dq();

#pragma unroll
  for (int k_phase_idx = 0; k_phase_idx < 4; k_phase_idx++) {
    compute_tile.lds_A(k_phase_idx);
    compute_tile.lds_B(k_phase_idx, 0);
    compute_tile.mma(0);

    compute_tile.lds_B(k_phase_idx, 1);
    compute_tile.mma(1);
  }

  __syncthreads();
  gmem_tile.sts();
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
void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const SplitKParams splitk_params,
    cudaStream_t stream) {
  uint32_t grid_x = (M + 31) / 32;
  uint32_t grid_y = (N + 255) / 256;
  uint32_t grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  half* C_split = static_cast<half*>(workspace);
  float* A_redsum = (float*)((char*)C_split +
                             aligned_size(M * N * grid_z) * sizeof(uint16_t));
  int* count =
      (int*)((char*)A_redsum + aligned_size(M * grid_z) * sizeof(float));

  dim3 grid(grid_x * (grid_y + 1) * grid_z);
  dim3 block(128);

  SM70_GEMM_A16W4_Params<half, QType> params{
      A,      B,      B_scale, B_zero,  C,
      (int)M, (int)N, (int)K,  C_split, splitk_params.SplitK};

  U32DivMod yz_divmod(grid_y * grid_z);
  U32DivMod z_divmod(grid_z);
  AS_CHECK_CUDA(cudaMemsetAsync(count, 0, sizeof(int), stream));
  volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk_kernel<QType,
                                                                         float>
      <<<grid, block, 0, stream>>>(params, grid_x, grid_y, grid_z, yz_divmod,
                                   z_divmod, A_redsum, count);

  // SplitK reduce
  gemm_f16_splitk_reduce<half, ActiveFunc>(C_split, B_scale, bias, C, M, N,
                                           grid_z, 1.0f, stream);
}

//-------------------
//-------------------
template void dequantize_rhs_a16w4<half, uint8_t>(
    const uint8_t*, const half*, const half*, half*, const uint32_t,
    const uint32_t, const uint32_t, const int, cudaStream_t);
template void dequantize_rhs_a16w4<hie::bfloat16, uint8_t>(
    const uint8_t*, const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*,
    const uint32_t, const uint32_t, const uint32_t, const int, cudaStream_t);

//
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Identity>(const half*, const uint8_t*,
                                              const half*, const half*,
                                              const half*, half*,
                                              const uint32_t, const uint32_t,
                                              const uint32_t, void*, const int,
                                              cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Gelu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, void*, const int,
                                          cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::GeluTanh>(const half*, const uint8_t*,
                                              const half*, const half*,
                                              const half*, half*,
                                              const uint32_t, const uint32_t,
                                              const uint32_t, void*, const int,
                                              cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Relu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, void*, const int,
                                          cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    half, uint8_t, hie::activation::Silu>(const half*, const uint8_t*,
                                          const half*, const half*, const half*,
                                          half*, const uint32_t, const uint32_t,
                                          const uint32_t, void*, const int,
                                          cudaStream_t);

template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Identity>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Gelu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::GeluTanh>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Relu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, cudaStream_t);
template void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<
    hie::bfloat16, uint8_t, hie::activation::Silu>(
    const hie::bfloat16*, const uint8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, cudaStream_t);

template void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Identity>(const half*, const uint8_t*,
                                        const half*, const half*, const half*,
                                        half*, const uint32_t, const uint32_t,
                                        const uint32_t, void*,
                                        const SplitKParams, cudaStream_t);
template void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Gelu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, void*, const SplitKParams,
                                    cudaStream_t);
template void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::GeluTanh>(const half*, const uint8_t*,
                                        const half*, const half*, const half*,
                                        half*, const uint32_t, const uint32_t,
                                        const uint32_t, void*,
                                        const SplitKParams, cudaStream_t);
template void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Relu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, void*, const SplitKParams,
                                    cudaStream_t);
template void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
    uint8_t, hie::activation::Silu>(const half*, const uint8_t*, const half*,
                                    const half*, const half*, half*,
                                    const uint32_t, const uint32_t,
                                    const uint32_t, void*, const SplitKParams,
                                    cudaStream_t);

}  // namespace cuda
}  // namespace allspark