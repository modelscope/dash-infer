/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_1x8x256.hpp
 */

#ifndef __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x8x256_HPP__
#define __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x8x256_HPP__

#include <cstdint>
#include <cstdlib>

#include "cuda/cuda_common.h"

namespace hie {

namespace dynamic_quant_matmul_fused {

constexpr int32_t ksna_align = 8;

template <typename TYPE, template <class> class ACT, int32_t NPROC,
          int32_t ALIGN, int32_t BLOCK>
__global__ void gemm_nn_1x8x256_activation_fused_kernel(
    const int8_t* gptr_a, const int8_t* zero_a, const int32_t* reduce_a,
    const float* scale_a, const int8_t* gptr_b, const int8_t* zero_b,
    const int32_t* reduce_b, const float* scale_b, const TYPE* tbias,
    TYPE* gptr_c, TYPE alpha, TYPE beta, int32_t gemm_m, int32_t gemm_n,
    int32_t gemm_k, int32_t loop_k, int32_t grid_n);

// const
constexpr int32_t gemm_nn_1x8x256_block = 256;
constexpr int32_t gemm_nn_1x8x256_align = 8;
constexpr int32_t gemm_nn_1x8x256_nproc = 64;
constexpr int32_t gemm_nn_1x8x256_k_num =
    gemm_nn_1x8x256_block / (gemm_nn_1x8x256_nproc / gemm_nn_1x8x256_align);

}  // namespace dynamic_quant_matmul_fused

int64_t dynamicQuantMatMulActivationFused1x8x256WorkSpace(hie::DataType dtype,
                                                          std::string act,
                                                          int sm_ver,
                                                          int sm_cnt, int m,
                                                          int n, int k);

void dynamicQuantMatMulActivationFused1x8x256Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c);

}  // namespace hie

#endif  // __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x8x256_HPP__
