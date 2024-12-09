/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel.h
 */

#pragma once

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

#include <common/hie_bfloat16.hpp>

namespace allspark {
namespace cpu {

#define TEST_TYPE_CVT_FP16 1  // temporary macro for test

void gemv_kernel_arm(int M, int N, int K, int lda, float* a_fp32, uint8_t* b_u8,
                     float* c_fp32, float* bias_fp32, void* wei_scale,
                     void* wei_scaleXzp, int GroupSize, int actType,
                     void* workspace);

void gemm_kernel_arm(int M, int N, int K, int lda, float* a_fp32,
                     hie::bfloat16* b_bf16, float* c_fp32, float* bias_fp32,
                     int actType, void* workspace);

void gemm_pack_weight_U8toU8_arm(int N, int K, int K_pack,
                                 const uint8_t* b_u8_unpack, uint8_t* b_u8);

void gemm_pack_weight_U8toBF16_arm(int N, int K, int K_pack,
                                   const uint8_t* b_u8_unpack,
                                   hie::bfloat16* b_bf16, const float* scale,
                                   const float* zero, int group_size);

void gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack,
                                     const float* b_fp32,
                                     hie::bfloat16* b_bf16);

}  // namespace cpu
}  // namespace allspark
#endif
