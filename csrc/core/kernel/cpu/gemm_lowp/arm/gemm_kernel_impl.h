/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel_impl.h
 */

#pragma once

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

#include <common/hie_bfloat16.hpp>

#include "allspark.pb.h"

namespace allspark {
namespace cpu {

template <typename Ta, typename Tb, typename Tp>
class GemmParam {
 public:
  GemmParam(int M, int N, int K_pack, Ta* a_ptr, Tb* b_ptr, float* c_ptr,
            float* bias_ptr, Tp* wei_scale, Tp* wei_scaleXzp, int GroupSize,
            int with_bias, int actType)
      : M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(wei_scale),
        wei_scaleXzp(wei_scaleXzp),
        GroupSize(GroupSize),
        with_bias(with_bias),
        actType(actType) {}

  GemmParam(int M, int N, int K_pack, Ta* a_ptr, Tb* b_ptr, float* c_ptr,
            float* bias_ptr, int with_bias, int actType)
      : M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(nullptr),
        wei_scaleXzp(nullptr),
        GroupSize(K_pack),
        with_bias(with_bias),
        actType(actType) {}

  GemmParam() {}
  ~GemmParam() {}

  int M, N, K_pack;
  Ta* a_ptr;
  Tb* b_ptr;
  float* c_ptr;
  float* bias_ptr;
  Tp *wei_scale, *wei_scaleXzp;
  int GroupSize;
  int with_bias;
  int actType = 0;
  int do_act = 1;
};

void thread_block_u8_m2_fp32(GemmParam<hie::bfloat16, uint8_t, float>& p, int m,
                             int n, int k, int k_tile, int is_res);

void thread_block_u8_m4_fp32(GemmParam<hie::bfloat16, uint8_t, float>& p, int m,
                             int n, int k, int k_tile);
void thread_block_u8_m4_fp32_res(GemmParam<hie::bfloat16, uint8_t, float>& p,
                                 int m, int n, int k, int k_tile);

void thread_block_u8_m8_fp32(GemmParam<hie::bfloat16, uint8_t, float>& p, int m,
                             int n, int k, int k_tile);
void thread_block_u8_m8_fp32_res(GemmParam<hie::bfloat16, uint8_t, float>& p,
                                 int m, int n, int k, int k_tile);

void thread_block_u8_m2_fp16(GemmParam<hie::bfloat16, uint8_t, float16_t>& p,
                             int m, int n, int k, int k_tile, int is_res);

void thread_block_u8_m4_fp16(GemmParam<hie::bfloat16, uint8_t, float16_t>& p,
                             int m, int n, int k, int k_tile);
void thread_block_u8_m4_fp16_res(
    GemmParam<hie::bfloat16, uint8_t, float16_t>& p, int m, int n, int k,
    int k_tile);

void thread_block_u8_m8_fp16(GemmParam<hie::bfloat16, uint8_t, float16_t>& p,
                             int m, int n, int k, int k_tile);
void thread_block_u8_m8_fp16_res(
    GemmParam<hie::bfloat16, uint8_t, float16_t>& p, int m, int n, int k,
    int k_tile);

void thread_block_bf16_m8(GemmParam<hie::bfloat16, hie::bfloat16, float>& p,
                          int m, int n, int k, int k_tile);
void thread_block_bf16_m8_mres(
    GemmParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k,
    int k_tile);
void thread_block_bf16_m8_nres(
    GemmParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k,
    int k_tile);
void thread_block_bf16_m8_res(GemmParam<hie::bfloat16, hie::bfloat16, float>& p,
                              int m, int n, int k, int k_tile);

void pack_input_impl_simd(int M, int N, int K, int lda, int K_pack,
                          float* a_fp32, hie::bfloat16* a_bf16);

void pack_input_impl_parallel_simd(int M, int N, int K, int lda, int K_pack,
                                   float* a_fp32, hie::bfloat16* a_bf16);

}  // namespace cpu
}  // namespace allspark
#endif
