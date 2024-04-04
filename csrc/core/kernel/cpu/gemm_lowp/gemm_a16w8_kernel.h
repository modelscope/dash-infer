/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_kernel.h
 */

#pragma once
#include <math.h>

#include <cstdint>
#include <cstdlib>

#ifdef ENABLE_ARM_V84_V9
#include "arm/gemm_kernel.h"
#endif

namespace allspark {
namespace cpu {
/**
 *
 */
struct GemmA16W8Launcher {
  enum KernelType {
    UNDEFINE,
    A16W8_GEMV,
    A16W16_GEMM
    /* A16W8_GEMM, A16W8_GEMV_SUBC, A16W8_GEMM_SUBC */
  };

  template <typename FT, typename QT>
  static void Run(const FT* lhs, const QT* rhs, const hie::bfloat16* rhs_bf16,
                  const void* scales, const void* scalexzp, const float* bias,
                  float* data_out, const uint32_t M, const uint32_t N,
                  const uint32_t K, const uint32_t lda,
                  const uint32_t GroupSize, void* workspace,
                  const int actType) {
    // switch (actType) {
    //     case UnaryType::UNARYTYPE_UNDEFINED: {
    KernelLaunch<FT, QT>(lhs, rhs, rhs_bf16, scales, scalexzp, bias, data_out,
                         M, N, K, lda, GroupSize, actType, workspace);
    //         break;
    //     }
    //     case UnaryType::TANH: {
    //         break;
    //     }
    //     case UnaryType::GELU_ERF: {
    //         break;
    //     }
    //     case UnaryType::GELU_TANH: {
    //         break;
    //     }
    //     case UnaryType::RELU: {
    //         break;
    //     }
    //     case UnaryType::SILU: {
    //         break;
    //     }
    //     default: {
    //         LOG(ERROR) << "No Kernel for acvtivtion : " << actType
    //                    << std::endl;
    //         break;
    //     }
    // }
  }
  static int64_t GetWorkSpaceSize(const uint32_t M, const uint32_t N,
                                  const uint32_t K, const uint32_t GroupSize) {
    int64_t ws_size = 0;
    KernelType k_type = SelectKernel(M, N, K, GroupSize);
    int bf16_elem_size = 2;
    switch (k_type) {
      case KernelType::A16W8_GEMV: {
        int K_pack = std::ceil(K / 8.0) * 8;
        int a_bf16_size = (M * K_pack + M % 2 * K_pack) * bf16_elem_size;
        ws_size = a_bf16_size;
        break;
      }
      case KernelType::A16W16_GEMM: {
        int K_pack = std::ceil(K / 8.0) * 8;
        int a_bf16_size = (M * K_pack + M % 2 * K_pack) * bf16_elem_size;
        ws_size = a_bf16_size;
        break;
      }
      default: {
        // TODO(wumi): impl kernels of other types
        break;
      }
    }
    return ws_size;
  }

  static KernelType SelectKernel(const uint32_t M, const uint32_t N,
                                 const uint32_t K, const uint32_t GroupSize) {
    if (M <= 16)
      return KernelType::A16W8_GEMV;
    else
      return KernelType::A16W16_GEMM;
  }

 private:
  template <typename FT, typename QT>
  static void KernelLaunch(const FT* lhs, const QT* rhs,
                           const hie::bfloat16* rhs_bf16, const void* scales,
                           const void* scalexzp, const FT* bias, FT* data_out,
                           const uint32_t M, const uint32_t N, const uint32_t K,
                           const uint32_t lda, const uint32_t GroupSize,
                           const int actType, void* workspace) {
    const KernelType k_type = SelectKernel(M, N, K, GroupSize);
    switch (k_type) {
#ifdef ENABLE_ARM_V84_V9
      case KernelType::A16W8_GEMV: {
        gemv_kernel_arm(M, N, K, lda, (float*)(lhs), (uint8_t*)(rhs),
                        (float*)(data_out), (float*)(bias), (void*)(scales),
                        (void*)(scalexzp), GroupSize, actType, workspace);
        break;
      }
      case KernelType::A16W16_GEMM: {
        gemm_kernel_arm(M, N, K, lda, (float*)(lhs), (hie::bfloat16*)(rhs_bf16),
                        (float*)(data_out), (float*)(bias), actType, workspace);
        break;
      }
#endif
      default:
        LOG(ERROR) << "GemmF16W8 No Kernel Launch\n";
        break;
    }
  }
};
}  // namespace cpu
}  // namespace allspark
