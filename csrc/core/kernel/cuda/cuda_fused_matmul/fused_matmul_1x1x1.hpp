/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_1x1x1.hpp
 */

#ifndef __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x1x1_HPP__
#define __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x1x1_HPP__

#include <cstdint>
#include <cstdlib>

#include "cuda/cuda_common.h"
namespace hie {

namespace dynamic_quant_matmul_fused {

// const
constexpr int32_t gemm_nn_1x1x1_block = 64;
constexpr int32_t gemm_nn_1x1x1_align = 1;
constexpr int32_t gemm_nn_1x1x1_nproc = 64;

}  // namespace dynamic_quant_matmul_fused

int64_t dynamicQuantMatMulActivationFused1x1x1WorkSpace(hie::DataType dtype,
                                                        std::string act,
                                                        int sm_ver, int sm_cnt,
                                                        int m, int n, int k);

void dynamicQuantMatMulActivationFused1x1x1Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c);

}  // namespace hie
#endif  // __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_1x1x1_HPP__
