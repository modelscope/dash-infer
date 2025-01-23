/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_kernel.h
 */

#pragma once
#ifdef ENABLE_CUSPARSELT
#include <cusparseLt.h>
#endif

#include <cstdint>
#include <cstdlib>

#include "../hie/cuda_activation.hpp"
#include "cuda/cuda_common.h"
#include "gemm_lowp_common.h"

namespace allspark {
namespace cuda {
#ifdef ENABLE_CUSPARSELT
template <typename FType>
void A_perc_symm_B_perc_asymm_dequantization_vllm(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType* B_scale, const FType* B_zero, const FType* bias, FType* result,
    const int M, const int N, cudaStream_t stream, bool is_prompt);

template <typename FType>
void per_channel_symm_dynamic_quantization_vllm(
    const FType* fdata, int8_t* qdata, float* scale, float* red_max,
    uint32_t* red_count, int32_t* red_sum, const int M, const int K,
    int sm_count, cudaStream_t stream, bool is_prompt);

template <typename FType>
void allspark_perc_qgemm_a8w8_gpu(const void* Af, const void* A_scale,
                                  const int8_t* B, const void* B_scale,
                                  const void* B_zero, const void* bias, void* C,
                                  void* workspace, const int M, const int N,
                                  const int K, const int sm_count,
                                  bool is_prompt, cudaStream_t stream,
                                  cublasHandle_t handle);
template <typename FType>
void allspark_perc_qgemm_sparse_a8w8_gpu(
    const void* Af, const void* A_scale, const int8_t* B, const void* B_scale,
    const void* B_zero, const void* bias, void* C, void* workspace, const int M,
    const int N, const int K, const int sm_count, bool is_prompt,
    cudaStream_t stream, cublasHandle_t handle, cusparseLtHandle_t cslt_handle,
    cusparseLtMatmulPlan_t& plan);
#endif

template <typename T>
__device__ __host__ __forceinline__ T DivCeil(const T a, const T b) {
  return (a + b - 1) / b;
}

template <typename FT>
void get_input_padded_k_align(const FT* in, FT* in_padded, const uint32_t m,
                              const uint32_t k, const uint32_t k_padded,
                              cudaStream_t stream);

template <typename FT>
void remove_padded_n_align(const FT* out_padded, FT* out, const uint32_t m,
                           const uint32_t n, const uint32_t padded_n,
                           cudaStream_t stream);

template <typename FT, typename QT>
void dequantize_rhs_a16w8(const QT* qdata, const FT* scales, const FT* zeros,
                          FT* fdata, const uint32_t N, const uint32_t K,
                          const int GroupSize, cudaStream_t stream);

template <typename FT, typename QT>
void restore_N32_K16_dequantize_rhs_a16w8(const QT* qdata, const FT* scales,
                                          const FT* zeros, FT* fdata,
                                          const int N_32align, const int N,
                                          const int K, const int GroupSize,
                                          cudaStream_t stream);

template <typename QT, template <class> class ActiveFunc>
void gemv_f16w8_perc_splitk(const half* lhs, const QT* rhs, const half* scales,
                            const half* zeros, const half* bias, half* data_out,
                            const uint32_t M, const uint32_t N,
                            const uint32_t K, void* workspace,
                            const float alpha, cudaStream_t stream);

template <typename FT, typename QT, template <class> class ActiveFunc>
void gemv_a16w8_subc_splitk(const FT* lhs, const QT* rhs, const FT* scales,
                            const FT* zeros, const FT* bias, FT* data_out,
                            const uint32_t M, const uint32_t N,
                            const uint32_t K, const uint32_t GroupSize,
                            void* workspace, const float alpha,
                            cudaStream_t stream);

template <typename QT, template <class> class ActiveFunc>
void hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8(
    const __half* A, const QT* B, const __half* B_scale, const __half* B_zero,
    const half* bias, __half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const float alpha, cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void hgemm_A16W8_subc_32x128x16_simt_Aldg1(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const int sm_count, const SplitKParams splitk_params, const float alpha,
    cudaStream_t stream);

template <typename QT, template <class> class ActiveFunc>
void hgemm_A16W8_subc_128x128x32_hmma1688_ldg8(
    const half* A, const QT* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const float alpha, cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_128x128x32_mma884(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count, const float alpha,
    cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count,
    const SplitKParams, const float alpha, cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    const int GroupSize, void* workspace, const int sm_count,
    const SplitKParams, const float alpha, cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const int M, const int N, const int K,
    void* workspace, const SplitKParams, const float alpha,
    cudaStream_t stream);

template <typename FT, typename QT, template <class> class ActiveFunc>
void hgemm_a16w8_subc_32x128x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, const uint32_t GroupSize, void* workspace,
    const int sm_count, const float alpha, cudaStream_t stream);

template <typename FType, typename QType, template <class> class ActiveFunc>
void ampere_hgemm_A16W8_perc_f16_f16_MtilexNtilex32_mma16816_multistage_AN_BTN32K16_CN_splitk(
    const FType* A, const QType* B, const FType* B_scale, const FType* B_zero,
    const FType* bias, FType* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const int sm_version,
    const SplitKParams fused_gemm_params, const float alpha,
    cudaStream_t stream);


/**
 * @brief
 *
 */
namespace GemmA16W8Launcher {

enum KernelType {
  UNDEFINE,                                 // DQ + Cublas
  A16W8_GEMV,                               // PerC :
  A16W8_GEMM,                               // PerC :
  A16W8_GEMV_SUBC,                          // SubC : M<=32, CudaCore
  A16W8_GEMV_SUBC_M1,                       // SubC : M==1, CudaCore
  Volta_A16W8_GEMM_SUBC_128x128x32,         // SubC : TensorCore 128x128x32_884
                                            // Non-SplitK
  Volta_A16W8_GEMM_SUBC_128x128x32_SplitK,  // SubC : TensorCore 128x128x32_884
                                            // SplitK
  Volta_A16W8_GEMM_SUBC_32x128x32,  // SubC : Tensor Core 32x128x32_884 SplitK
  Volta_A16W8_GEMM_PERC_32x128x32,  // PerC : Tensor Core 32x128x32_884 SplitK
  A16W8_GEMM_SUBC,                  // SubC : TensorCore 128x128x32_1688
  A16W8_GEMM_SUBC_16816,            // SubC : TensorCore 32x128x32_16816
  Ampere_A16W8_GEMM_PERC_MtilexNtilex32,      // PerC : TensorCore
                                              // MtilexNtilex32_16816
};

/*
 *   Process OOM bug caused by different kernel selection in the warm up phase
 * and the running phase If VOLTA_GEMM_MAX_GRIDZ is set too large, it will waste
 * too much GPU memory when running Subc INT8-Quant model in Volta GPU If
 * VOLTA_GEMM_MAX_GRIDZ is set too small, it may hurt performance when running
 * Subc INT8-Quant model in Volta GPU
 */
constexpr int VOLTA_GEMM_MAX_GRIDZ = 15;

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int CTA_PER_SM>
bool IsEnableSplitK(const int M, const int N, const int K, const int sm_count,
                    SplitKParams& splitk_params) {
  int grid_xy = DivCeil(M, BLOCK_TILE_M) * DivCeil(N, BLOCK_TILE_N);
  int grid_z;

  // split-K
  int blocks_per_wave = sm_count * CTA_PER_SM;
  int total_waves = DivCeil(grid_xy, sm_count);

  // too many waves, not enable SplitK
  if (total_waves > 15) {
    splitk_params.EnableSplitK = false;
    return false;
  }
  if (total_waves >= 4 && (grid_xy % sm_count) >= 0.8 * sm_count) {
    splitk_params.EnableSplitK = false;
    return false;
  }
  const float SPLIT_THRESHOLD = 0.8;
  int n_slice;
  for (n_slice = 1; n_slice < K / 512; ++n_slice) {
    int n_block = grid_xy * n_slice;
    if (n_block >= blocks_per_wave * SPLIT_THRESHOLD &&
        (n_block % sm_count == 0 ||
         n_block % sm_count >= sm_count * SPLIT_THRESHOLD)) {
      break;
    }
  }
  int k_slice =
      (K / n_slice) % 32 == 0 ? K / n_slice : K / n_slice / 32 * 32 + 32;
  grid_z = (K + k_slice - 1) / k_slice;
  bool result = grid_z > 1 && grid_z < VOLTA_GEMM_MAX_GRIDZ ? true : false;
  splitk_params.EnableSplitK = result;
  if (result) {
    splitk_params.SplitK = k_slice;
  }
  return result;
}

/**
 * @brief Select Kernel
 */
template <typename FT, typename QT>
struct SelectKernel {
  void operator()(const uint32_t M, const uint32_t N, const uint32_t K,
                  const int GroupSize, const int sm_count, const int sm_version,
                  SplitKParams& splitk_params);
};

template <typename QT>
struct SelectKernel<half, QT> {
  KernelType operator()(const uint32_t M, const uint32_t N, const uint32_t K,
                        const int GroupSize, const int sm_count,
                        const int sm_version, SplitKParams& splitk_params) {
    // PerChannel
    if (GroupSize == -1) {
      if (M <= 1024) {
        if (sm_version >= 0x0800) {
          if (K % 16 == 0 && N % 8 == 0) {
            return KernelType::Ampere_A16W8_GEMM_PERC_MtilexNtilex32;
          }
          return KernelType::UNDEFINE;
        }
        if (M <= 32 && sm_version == 0x0700 && K % 8 == 0 && N % 8 == 0) {
          return KernelType::Volta_A16W8_GEMM_PERC_32x128x32;
        }
        if (M <= 8 && N % 2 == 0) {
          return KernelType::A16W8_GEMV;
        }
      } else {
        if (N % 8 == 0 && K % 8 == 0 && sm_version >= 0x0705) {
#if (CUDA_VERSION >= 11000)
// return KernelType::A16W8_GEMM;
#endif
        }
      }
    }
    // SubChannel
    else {
      if (M <= 32) {
        if (sm_version == 0x0700 && K % 8 == 0 && N % 8 == 0) {
          return KernelType::Volta_A16W8_GEMM_SUBC_32x128x32;
        }
        if (sm_version >= 0x0800 && K % 2 == 0 && N % 4 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W8_GEMM_SUBC_16816;
#endif
        }
        if (sm_version >= 0x0705 && M == 1 && N % 8 == 0) {
          return KernelType::A16W8_GEMV_SUBC_M1;
        }
        return KernelType::A16W8_GEMV_SUBC;
      } else {
        if (sm_version == 0x0700 && N % 8 == 0 && K % 8 == 0) {
          // only 2 thread block per sm, limited by register and shared memory
          bool is_splitk =
              IsEnableSplitK<128, 128, 2>(M, N, K, sm_count, splitk_params);
          if (is_splitk && splitk_params.SplitK <= 2048) {
            return Volta_A16W8_GEMM_SUBC_128x128x32_SplitK;
          }
          if (!is_splitk && K <= 2048) {
            return Volta_A16W8_GEMM_SUBC_128x128x32;
          }
        }
        if (sm_version >= 0x0705 && M <= 512 && N % 8 == 0 && K % 8 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W8_GEMM_SUBC;
#endif
        }
      }
    }
    return KernelType::UNDEFINE;
  }
};
template <typename QT>
struct SelectKernel<hie::bfloat16, QT> {
  KernelType operator()(const uint32_t M, const uint32_t N, const uint32_t K,
                        const int GroupSize, const int sm_count,
                        const int sm_version, SplitKParams& splitk_params) {
    if (GroupSize == -1) {
      // TODO: PerChannel
      if (M <= 1024) {
        if (sm_version >= 0x0800) {
          if (K % 16 == 0 && N % 8 == 0) {
            return KernelType::Ampere_A16W8_GEMM_PERC_MtilexNtilex32;
          }
        }
      }
    } else {
      // TODO: SubChannel
      if (M == 1 && N % 8 == 0) {
        return KernelType::A16W8_GEMV_SUBC_M1;
      }
      if (M <= 32 && K % 2 == 0 && N % 4 == 0 && sm_version >= 0x0800) {
#if (CUDA_VERSION >= 11000)
        return KernelType::A16W8_GEMM_SUBC_16816;
#endif
      }
    }
    return KernelType::UNDEFINE;
  }
};

/**
 * @brief
 *
 */
uint64_t GetWorkSpaceSize(const KernelType k_type, const uint32_t M,
                          const uint32_t N, const uint32_t K,
                          const int GroupSize, const int sm_count,
                          const int sm_version, SplitKParams& splitk_params);

/**
 * @brief Kernel Lauch
 *
 */
template <typename FT, typename QT, template <class> class ActiveFunc>
struct KernelLaunch {
  void operator()(const KernelType k_type, const FT* lhs, const QT* rhs,
                  const FT* scales, const FT* zeros, const FT* bias,
                  FT* data_out, const uint32_t M, const uint32_t N,
                  const uint32_t K, const int GroupSize, const int sm_count,
                  const int sm_version, void* workspace,
                  const SplitKParams splitk_params, const float alpha,
                  cudaStream_t stream);
};

// FP16
template <typename QT, template <class> class ActiveFunc>
struct KernelLaunch<half, QT, ActiveFunc> {
  using FT = half;
  void operator()(const KernelType k_type, const FT* lhs, const QT* rhs,
                  const FT* scales, const FT* zeros, const FT* bias,
                  FT* data_out, const uint32_t M, const uint32_t N,
                  const uint32_t K, const int GroupSize, const int sm_count,
                  const int sm_version, void* workspace,
                  const SplitKParams splitk_params, const float alpha,
                  cudaStream_t stream) {
    switch (k_type) {
      case KernelType::A16W8_GEMV_SUBC_M1: {
        gemv_a16w8_subc_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, alpha, stream);
        break;
      }
      case KernelType::A16W8_GEMV_SUBC: {
        hgemm_A16W8_subc_32x128x16_simt_Aldg1<QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, splitk_params, alpha, stream);
        break;
      }
      case KernelType::A16W8_GEMM_SUBC: {
        hgemm_A16W8_subc_128x128x32_hmma1688_ldg8<QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, alpha, stream);
        break;
      }
      case KernelType::Volta_A16W8_GEMM_SUBC_128x128x32: {
        volta_hgemm_A16W8_f16_f16_128x128x32_mma884<QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, alpha, stream);
        break;
      }
      case KernelType::Volta_A16W8_GEMM_SUBC_128x128x32_SplitK: {
        volta_hgemm_A16W8_f16_f16_128x128x32_mma884_nonfused_splitk<QT,
                                                                    ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, splitk_params, alpha, stream);
        break;
      }
      case KernelType::Volta_A16W8_GEMM_SUBC_32x128x32: {
        volta_hgemm_A16W8_f16_f16_32x128x32_mma884_nonfused_splitk<QT,
                                                                   ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, splitk_params, alpha, stream);
        break;
      }
      case KernelType::Volta_A16W8_GEMM_PERC_32x128x32: {
        volta_hgemm_A16W8_perc_f16_f16_32x128x32_mma884_nonfused_splitk<
            QT, ActiveFunc>(lhs, rhs, scales, zeros, bias, data_out, M, N, K,
                            workspace, splitk_params, alpha, stream);
        break;
      }
      case KernelType::A16W8_GEMV: {
        gemv_f16w8_perc_splitk<QT, ActiveFunc>(lhs, rhs, scales, zeros, bias,
                                               data_out, M, N, K, workspace,
                                               alpha, stream);
        break;
      }
      case KernelType::A16W8_GEMM: {
        hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, workspace, alpha,
            stream);
        break;
      }
      case KernelType::A16W8_GEMM_SUBC_16816: {
        hgemm_a16w8_subc_32x128x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, alpha, stream);
        break;
      }
      case KernelType::Ampere_A16W8_GEMM_PERC_MtilexNtilex32: {
        ampere_hgemm_A16W8_perc_f16_f16_MtilexNtilex32_mma16816_multistage_AN_BTN32K16_CN_splitk<
            FT, QT, ActiveFunc>(lhs, rhs, scales, zeros, bias, data_out, M, N,
                                K, workspace, sm_version, splitk_params, alpha,
                                stream);
        break;
      }
      default:
        LOG(ERROR) << "GemmFP16W8 No Kernel Launch\n";
        break;
    }
  }
};

// BF16
template <typename QT, template <class> class ActiveFunc>
struct KernelLaunch<hie::bfloat16, QT, ActiveFunc> {
  using FT = hie::bfloat16;
  void operator()(const KernelType k_type, const FT* lhs, const QT* rhs,
                  const FT* scales, const FT* zeros, const FT* bias,
                  FT* data_out, const uint32_t M, const uint32_t N,
                  const uint32_t K, const int GroupSize, const int sm_count,
                  const int sm_version, void* workspace,
                  const SplitKParams splitk_params, const float alpha,
                  cudaStream_t stream) {
    switch (k_type) {
      case KernelType::A16W8_GEMV_SUBC_M1: {
        gemv_a16w8_subc_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, alpha, stream);
        break;
      }
      case KernelType::A16W8_GEMM_SUBC_16816: {
        hgemm_a16w8_subc_32x128x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, sm_count, alpha, stream);
        break;
      }
      case KernelType::Ampere_A16W8_GEMM_PERC_MtilexNtilex32: {
        ampere_hgemm_A16W8_perc_f16_f16_MtilexNtilex32_mma16816_multistage_AN_BTN32K16_CN_splitk<
            FT, QT, ActiveFunc>(lhs, rhs, scales, zeros, bias, data_out, M, N,
                                K, workspace, sm_version, splitk_params, alpha,
                                stream);
        break;
      }
      default:
        LOG(ERROR) << "GemmBF16W8 No Kernel Launch\n";
        break;
    }
  }
};

template <typename FT, typename QT>
void Run(const KernelType k_type, const FT* lhs, const QT* rhs,
         const FT* scales, const FT* zeros, const FT* bias, FT* data_out,
         const uint32_t M, const uint32_t N, const uint32_t K,
         const int GroupSize, void* workspace, const int active_func_type,
         const int sm_count, const int sm_version,
         const SplitKParams splitk_params, const float alpha,
         const cudaStream_t stream) {
  switch (active_func_type) {
    case 0: {
      KernelLaunch<FT, QT, hie::activation::Identity>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, alpha, stream);
      break;
    }
    // case 1:// Tanh
    //     break;
    case 2: {  // Gelu
      KernelLaunch<FT, QT, hie::activation::Gelu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, alpha, stream);
      break;
    }
    case 3: {  // GeluTanh
      KernelLaunch<FT, QT, hie::activation::GeluTanh>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, alpha, stream);
      break;
    }
    case 4: {  // Relu
      KernelLaunch<FT, QT, hie::activation::Relu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, alpha, stream);
      break;
    }
    case 5: {  // Silu
      KernelLaunch<FT, QT, hie::activation::Silu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, alpha, stream);
      break;
    }
    default:
      LOG(ERROR) << "No Kernel for acvtivtion : " << active_func_type
                 << std::endl;
      break;
  }
}
}  // namespace GemmA16W8Launcher
template <typename FType, typename DType>
void per_tensor_symm_quantization(const FType* fdata, float* scale,
                                  DType* qdata, const int M, const int K,
                                  cudaStream_t stream);
template <typename FType>
void per_channel_symm_dynamic_quantization(const FType* fdata, int8_t* qdata,
                                           float* scale, float* red_max,
                                           uint32_t* red_count,
                                           int32_t* red_sum, const int M,
                                           const int K, int sm_count,
                                           cudaStream_t stream);
template <typename FType>
void A_pert_symm_B_perc_symm_dequantization(const int* imd_result,
                                            const float* A_scale,
                                            const FType* B_scale,
                                            const FType* bias, FType* result,
                                            const int M, const int N,
                                            cudaStream_t stream);
template <typename FType>
void rearrange_kn_weight_as_n32k16_order_ldg16(
    const int8_t* B, const FType* B_scale, const FType* B_zero,
    int8_t* B_result, FType* B_scale_result, FType* B_zero_result, const int K,
    const int N, const int N_32align, cudaStream_t stream);
template <typename FType>
void A_perc_symm_B_perc_asymm_dequantization(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType* B_scale, const FType* B_zero, const FType* bias, FType* result,
    const int M, const int N, cudaStream_t stream);
template <typename FType>
void A_perc_symm_B_perc_array_asymm_dequantization(
    const int* imd_result, const float* A_scale, const int* A_redsum,
    const FType** B_scale, const FType** B_zero, const FType* bias,
    FType* result, const int M, const int N, const int max_block,
    cudaStream_t stream);
template <typename FType>
void restore_n32k16_weight_to_nk(const int8_t* B_n32k16,
                                 const FType* B_scale_n32,
                                 const FType* B_zero_n32, int8_t* B_nk,
                                 FType* B_scale, FType* B_zero,
                                 const int N_32align, const int N, const int K,
                                 cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
