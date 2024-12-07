/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_kernel.h
 */

#pragma once
#include <cstdint>
#include <cstdlib>

#include "../hie/cuda_activation.hpp"
#include "cuda/cuda_common.h"
#include "gemm_lowp_common.h"

namespace allspark {
namespace cuda {

template <typename FT, typename QT>
void dequantize_rhs_a16w4(const QT* qdata, const FT* scales, const FT* zeros,
                          FT* fdata, const uint32_t N, const uint32_t NPack,
                          const uint32_t K, const int GroupSize,
                          cudaStream_t stream);

template <typename FT, typename QT, template <class> class ActiveFunc>
void hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const int sm_count, cudaStream_t stream);

template <typename FT, typename QT, template <class> class ActiveFunc>
void hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk(
    const FT* Adev, const QT* Bdev, const FT* BSdev, const FT* BZdev,
    const FT* bias, FT* Cdev, const uint32_t M, const uint32_t N,
    const uint32_t K, const int GroupSize, void* workspace,
    const SplitKParams splitk_params, cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const SplitKParams splitk_params,
    cudaStream_t stream);

template <typename QType, template <class> class ActiveFunc>
void volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk(
    const half* A, const QType* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, const int GroupSize, void* workspace,
    const SplitKParams splitk_params, cudaStream_t stream);
/**
 * @brief
 *
 */
namespace GemmA16W4Launcher {

enum KernelType {
  UNDEFINE,                         // DQ + Cublas
  A16W4_GEMV_PERC_16816,            // PerC : TensorCore 32x256x32_16816
  A16W4_GEMV_SUBC_16816,            // SubC : TensorCore 32x256x32_16816
  Volta_A16W4_GEMV_PERC_32x256x32,  // PerC : TensorCore 32x256x32_884 SplitK
  Volta_A16W4_GEMV_SUBC_32x256x32   // SUBC : TensorCore 32x256x32_884 SplitK
};

/**
 * @brief Select Kernel
 */
template <typename FT, typename QT>
struct SelectKernel {
  KernelType operator()(const uint32_t M, const uint32_t N, const uint32_t K,
                        const int GroupSize, const int sm_version);
};
template <typename QT>
struct SelectKernel<half, QT> {
  KernelType operator()(const uint32_t M, const uint32_t N, const uint32_t K,
                        const int GroupSize, const int sm_version) {
    // PerChannel
    if (GroupSize == -1) {
      if (M <= 32) {
        if (sm_version == 0x0700 && K % 8 == 0 && N % 8 == 0) {
          return KernelType::Volta_A16W4_GEMV_PERC_32x256x32;
        }
        if (sm_version >= 0x0800 && K % 8 == 0 && N % 8 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W4_GEMV_PERC_16816;
#endif
        }
      }
    }
    // SubChannel
    else {
      if (M <= 32) {
        if (sm_version == 0x0700 && K % 8 == 0 && N % 8 == 0) {
          return KernelType::Volta_A16W4_GEMV_SUBC_32x256x32;
        }
        if (sm_version >= 0x0800 && K % 8 == 0 && N % 8 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W4_GEMV_SUBC_16816;
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
                        const int GroupSize, const int sm_version) {
    // PerChannel
    if (GroupSize == -1) {
      if (M <= 32) {
        if (sm_version >= 0x0800 && K % 8 == 0 && N % 8 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W4_GEMV_PERC_16816;
#endif
        }
      }
    }
    // SubChannel
    else {
      if (M <= 32) {
        if (sm_version >= 0x0800 && K % 8 == 0 && N % 8 == 0) {
#if (CUDA_VERSION >= 11000)
          return KernelType::A16W4_GEMV_SUBC_16816;
#endif
        }
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
                  const SplitKParams splitk_params, cudaStream_t stream);
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
                  const SplitKParams splitk_params, cudaStream_t stream) {
    switch (k_type) {
      case KernelType::Volta_A16W4_GEMV_PERC_32x256x32: {
        volta_hgemm_perc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
            QT, ActiveFunc>(lhs, rhs, scales, zeros, bias, data_out, M, N, K,
                            workspace, splitk_params, stream);
        break;
      }
      case KernelType::Volta_A16W4_GEMV_SUBC_32x256x32: {
        volta_hgemm_subc_A16W4_f16_f16_32x256x32_mma884_nonfused_splitk<
            QT, ActiveFunc>(lhs, rhs, scales, zeros, bias, data_out, M, N, K,
                            GroupSize, workspace, splitk_params, stream);
        break;
      }
      case KernelType::A16W4_GEMV_PERC_16816: {
        hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, workspace,
            sm_count, stream);
        break;
      }
      case KernelType::A16W4_GEMV_SUBC_16816: {
        hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, splitk_params, stream);
        break;
      }
      default:
        LOG(ERROR) << "GemmFP16W4 No Kernel Launch\n";
        break;
    };
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
                  const SplitKParams splitk_params, cudaStream_t stream) {
    switch (k_type) {
      case KernelType::A16W4_GEMV_PERC_16816: {
        hgemm_a16w4_perc_32x256x32_32x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, workspace,
            sm_count, stream);
        break;
      }
      case KernelType::A16W4_GEMV_SUBC_16816: {
        hgemm_a16w4_subc_32x256x32_32x32_16816_nn_splitk<FT, QT, ActiveFunc>(
            lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
            workspace, splitk_params, stream);
        break;
      }
      default:
        LOG(ERROR) << "GemmFP16W4 No Kernel Launch\n";
        break;
    };
  }
};

template <typename FT, typename QT>
void Run(const KernelType k_type, const FT* lhs, const QT* rhs,
         const FT* scales, const FT* zeros, const FT* bias, FT* data_out,
         const uint32_t M, const uint32_t N, const uint32_t K,
         const int GroupSize, void* workspace, const int active_func_type,
         const int sm_count, const int sm_version,
         const SplitKParams splitk_params, const cudaStream_t stream) {
  switch (active_func_type) {
    case 0: {
      KernelLaunch<FT, QT, hie::activation::Identity>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, stream);
      break;
    }
      // case 1:// Tanh
      //     break;
    case 2: {  // Gelu
      KernelLaunch<FT, QT, hie::activation::Gelu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, stream);
      break;
    }
    case 3: {  // GeluTanh
      KernelLaunch<FT, QT, hie::activation::GeluTanh>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, stream);
      break;
    }
    case 4: {  // Relu
      KernelLaunch<FT, QT, hie::activation::Relu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, stream);
      break;
    }
    case 5: {  // Silu
      KernelLaunch<FT, QT, hie::activation::Silu>()(
          k_type, lhs, rhs, scales, zeros, bias, data_out, M, N, K, GroupSize,
          sm_count, sm_version, workspace, splitk_params, stream);
      break;
    }
    default:
      LOG(ERROR) << "No Kernel for acvtivtion : " << active_func_type
                 << std::endl;
      break;
  }
}

}  // namespace GemmA16W4Launcher
}  // namespace cuda
}  // namespace allspark