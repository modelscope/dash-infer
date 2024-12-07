/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_interface.hpp
 */

#ifndef __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_INTERFACE_HPP__
#define __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_INTERFACE_HPP__

#include <vector>

#include "cuda/cuda_common.h"
namespace hie {

typedef enum {
  DQMM_FUSED_NONE = 0,
  DQMM_FUSED_1x1x1 = 1,
  DQMM_FUSED_1x8x256 = 2,
  DQMM_FUSED_256x128x64 = 3,
} dqmm_fused_kernel;

dqmm_fused_kernel dynamicQuantizeMatMulFusedFindKernel(hie::DataType dtype,
                                                       std::string act,
                                                       int sm_ver, int sm_cnt,
                                                       int m, int n, int k);

// if invalid return -1.
int64_t dynamicQuantizeMatMulWorkSpace(dqmm_fused_kernel kernel,
                                       hie::DataType dtype, std::string act,
                                       int sm_ver, int sm_cnt, int m, int n,
                                       int k);

void dynamicQuantizeMatMulLaunch(cudaStream_t stream, dqmm_fused_kernel kernel,
                                 hie::DataType dtype, std::string act,
                                 int sm_ver, int sm_cnt, int m, int n, int k,
                                 float alpha, float beta, const int8_t* aquant,
                                 const int8_t* azero, const int32_t* areduce,
                                 const float* ascale, const int8_t* bquant,
                                 const int8_t* bzero, const int32_t* breduce,
                                 const float* bscale, const void* bias,
                                 void* c);

}  // namespace hie

#endif  // __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_INTERFACE_HPP__
