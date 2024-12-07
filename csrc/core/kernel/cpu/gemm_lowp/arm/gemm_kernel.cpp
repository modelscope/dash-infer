/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include "gemm_kernel.h"

#include <arm_sve.h>
#include <assert.h>
#include <math.h>

#include <algorithm>
#include <common/hie_bfloat16.hpp>
#include <iostream>

#include "../../cpu_common.h"
#include "gemm_kernel_impl.h"

namespace allspark {
namespace cpu {
static void pack_input_arm(int M, int N, int K, int lda, int K_pack,
                           float* a_fp32, hie::bfloat16* a_bf16) {
  pack_input_impl_parallel_simd(M, N, K, lda, K_pack, a_fp32, a_bf16);
  // pack_input_impl_simd(M, N, K, lda, K_pack, a_fp32, a_bf16);
  return;
}

static void gemv_get_tile_n(int M, int N, int K_pack, int* tile_size,
                            int* num_threads) {
#define BLOCK_DIV_STRATEGY 1
  int l1size = 64 * 1024;    // 64KB
  int l2size = 1024 * 1024;  // 1MB

  int wei_inner_w = K_pack * 2;
  int wei_inner_h = 2;
  int wei_inner_size = wei_inner_w * wei_inner_h * 1.0;  // sizeof(u8) = 1.0
  int src_h = M > 8 ? 8 : M;
  int src_size = (src_h * K_pack + src_h % 2 * K_pack) * 2;  // sizeof(bf16) = 2

#if BLOCK_DIV_STRATEGY
  float l2ratio = 0.8;
  int n_tile = 0;
  int n_tile_max = std::floor(l2size * l2ratio / wei_inner_size);
  for (int i = 0; i < n_tile_max; i++) {
    if ((i * wei_inner_size + src_size) < (l2size * l2ratio)) n_tile++;
  }
  n_tile = std::floor(n_tile / 4.0) * 4;
#else  // BLOCK_DIV_STRATEGY
  int n_tile = std::floor(l2size * 1.0 / wei_inner_size) * 0.25;
  n_tile = std::floor(n_tile / 4.0) * 4;
#endif
  n_tile = n_tile < 4 ? 4 : n_tile;
  int n_thread = std::ceil(N * 1.0 / n_tile);

  *tile_size = n_tile;
  *num_threads = n_thread;
  return;
}

#if TEST_TYPE_CVT_FP16
static void gemv_thread_strategy(
    GemmParam<hie::bfloat16, uint8_t, float16_t>& p) {
  int M = p.M;
  int N = p.N;
  int K_pack = p.K_pack;

  int n_tile = 4, n_thread = (N + 3) / 4;
  gemv_get_tile_n(M, N, K_pack, &n_tile, &n_thread);

  int m_tile = 2;
  if (M > 2) m_tile = 4;
  if (M > 4) m_tile = 8;

  parallel_for(n_thread, [&](int n) {
    int nn_max = std::min(N, (n + 1) * n_tile);
    for (int nn = n * n_tile; nn < nn_max; nn += 4) {
      for (int m = 0; m < M; m += m_tile) {
        int m_tile_real = m_tile;
        if (m + m_tile > M) m_tile_real = M - m;

        if (m_tile_real > 4) {
          if ((nn + 4 > nn_max) || (m + m_tile > M)) {
            thread_block_u8_m8_fp16_res(p, m, nn, 0, K_pack);
          } else {
            thread_block_u8_m8_fp16(p, m, nn, 0, K_pack);
          }
        } else if (m_tile_real > 2) {
          if ((nn + 4 > nn_max) || (m + m_tile > M)) {
            thread_block_u8_m4_fp16_res(p, m, nn, 0, K_pack);
          } else {
            thread_block_u8_m4_fp16(p, m, nn, 0, K_pack);
          }
        } else {
          thread_block_u8_m2_fp16(p, m, nn, 0, K_pack, m + 1 >= M);
        }
      }
    }
  });
}
#else
static void gemv_thread_strategy(GemmParam<hie::bfloat16, uint8_t, float>& p) {
  int M = p.M;
  int N = p.N;
  int K_pack = p.K_pack;

  int n_tile = 4, n_thread = (N + 3) / 4;
  gemv_get_tile_n(M, N, K_pack, &n_tile, &n_thread);

  int m_tile = 2;
  if (M > 2) m_tile = 4;
  if (M > 4) m_tile = 8;

  parallel_for(n_thread, [&](int n) {
    int nn_max = std::min(N, (n + 1) * n_tile);
    for (int nn = n * n_tile; nn < nn_max; nn += 4) {
      for (int m = 0; m < M; m += m_tile) {
        int m_tile_real = m_tile;
        if (m + m_tile > M) m_tile_real = M - m;

        if (m_tile_real > 4) {
          if ((nn + 4 > nn_max) || (m + m_tile > M)) {
            thread_block_u8_m8_fp32_res(p, m, nn, 0, K_pack);
          } else {
            thread_block_u8_m8_fp32(p, m, nn, 0, K_pack);
          }
        } else if (m_tile_real > 2) {
          if ((nn + 4 > nn_max) || (m + m_tile > M)) {
            thread_block_u8_m4_fp32_res(p, m, nn, 0, K_pack);
          } else {
            thread_block_u8_m4_fp32(p, m, nn, 0, K_pack);
          }
        } else {
          thread_block_u8_m2_fp32(p, m, nn, 0, K_pack, m + 1 >= M);
        }
      }
    }
  });
}
#endif

void gemv_kernel_arm(int M, int N, int K, int lda, float* a_fp32, uint8_t* b_u8,
                     float* c_fp32, float* bias_fp32, void* wei_scale,
                     void* wei_scaleXzp, int GroupSize, int actType,
                     void* workspace) {
  int K_pack = std::ceil(K / 8.0) * 8;
  int with_bias = bias_fp32 == nullptr ? 0 : 1;

  hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
  int a_bf16_size = (M * K_pack + M % 2 * K_pack) * 2;
  memset(a_bf16, 0, a_bf16_size);

  pack_input_arm(M, N, K, lda, K_pack, a_fp32, a_bf16);

#if TEST_TYPE_CVT_FP16
  GemmParam<hie::bfloat16, uint8_t, float16_t> p(
      M, N, K_pack, a_bf16, b_u8, c_fp32, bias_fp32, (float16_t*)wei_scale,
      (float16_t*)wei_scaleXzp, GroupSize, with_bias, actType);
#else
  GemmParam<hie::bfloat16, uint8_t, float> p(
      M, N, K_pack, a_bf16, b_u8, c_fp32, bias_fp32, (float*)wei_scale,
      (float*)wei_scaleXzp, GroupSize, with_bias, actType);
#endif

  gemv_thread_strategy(p);

  return;
}

/*********************************************/

static void gemm_thread_block_bf16(
    GemmParam<hie::bfloat16, hie::bfloat16, float> p, int m, int n, int m_tile,
    int n_tile, int k_tile) {
  int nn_max = std::min(p.N, (n + 1) * n_tile);
  int mm_max = std::min(p.M, (m + 1) * m_tile);
  int last_block = 0;
  if (mm_max == p.M && p.M % 8 != 0) last_block |= 0x1;

  if (nn_max == p.N && p.N % 8 != 0) last_block |= 0x2;

  for (int k = 0; k < p.K_pack; k += k_tile) {
    p.do_act = 0;
    if ((k + k_tile) >= p.K_pack) p.do_act = 1;
    for (int nn = n * n_tile; nn < nn_max; nn += 8) {
      for (int mm = m * m_tile; mm < mm_max; mm += 8) {
        if (LIKELY(last_block == 0x0)) {
          thread_block_bf16_m8(p, mm, nn, k, k_tile);
        } else if (last_block == 0x1) {
          thread_block_bf16_m8_mres(p, mm, nn, k, k_tile);
        } else if (last_block == 0x2) {
          thread_block_bf16_m8_nres(p, mm, nn, k, k_tile);
        } else {
          thread_block_bf16_m8_res(p, mm, nn, k, k_tile);
        }
      }
    }
  }
}

static void gemm_thread_strategy(
    GemmParam<hie::bfloat16, hie::bfloat16, float>& p) {
  int m_tile = 32;
  int n_tile = 64;
  int k_tile = 2560;
  if (p.K_pack == 5120) k_tile = 5120;

  int m_max = (p.M + m_tile - 1) / m_tile;
  int n_max = (p.N + n_tile - 1) / n_tile;
  parallel_for(m_max, n_max, [&](int m, int n) {
    gemm_thread_block_bf16(p, m, n, m_tile, n_tile, k_tile);
  });
  return;
}

void gemm_kernel_arm(int M, int N, int K, int lda, float* a_fp32,
                     hie::bfloat16* b_bf16, float* c_fp32, float* bias_fp32,
                     int actType, void* workspace) {
  int K_pack = std::ceil(K / 8.0) * 8;
  int with_bias = bias_fp32 == nullptr ? 0 : 1;

  hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
  int a_bf16_size = (M * K_pack + M % 2 * K_pack) * 2;
  memset(a_bf16, 0, a_bf16_size);

  pack_input_arm(M, N, K, lda, K_pack, a_fp32, a_bf16);

  GemmParam<hie::bfloat16, hie::bfloat16, float> p(
      M, N, K_pack, a_bf16, b_bf16, c_fp32, bias_fp32, with_bias, actType);

  gemm_thread_strategy(p);

  return;
}

/*********************************************/

void gemm_pack_weight_U8toU8_arm(int N, int K, int K_pack,
                                 const uint8_t* b_u8_unpack, uint8_t* b_u8) {
  int k_tile = 1024;  // empirical var: 1024, 5120
  int k_thread = std::ceil(K * 1.0 / k_tile);

  parallel_for(k_thread, [&](int k) {
    for (int n = 0; n < N; n += 2) {
      uint8_t* b_u8_unpack_ptr1 =
          (uint8_t*)(b_u8_unpack + n + (k * k_tile) * N);
      uint8_t* b_u8_unpack_ptr2 =
          (uint8_t*)(b_u8_unpack + n + 1 + (k * k_tile) * N);
      uint8_t* b_u8_ptr = b_u8 + (n / 2) * K_pack * 2 + k * k_tile * 2;
      int kk_max = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;
      for (int kk = k * k_tile; kk < kk_max; kk += 4) {
        for (int i = 0; i < 4 && (kk + i) < K; i++) {
          b_u8_ptr[i + 0] = b_u8_unpack_ptr1[i * N];
          if (n + 1 < N) {
            b_u8_ptr[i + 4] = b_u8_unpack_ptr2[i * N];
          } else {
            b_u8_ptr[i + 4] = 0;
          }
        }
        b_u8_ptr += 8;
        b_u8_unpack_ptr1 += 4 * N;
        b_u8_unpack_ptr2 += 4 * N;
      }
    }
  });

  return;
}

void gemm_pack_weight_U8toBF16_arm(int N, int K, int K_pack,
                                   const uint8_t* b_u8_unpack,
                                   hie::bfloat16* b_bf16, const float* scale,
                                   const float* zero, int group_size) {
  int N_pack = (N + 1) / 2;
  parallel_for(N_pack, [&](int n) {
    for (int k = 0, subch_idx = 0; k < K; k += group_size, subch_idx++) {
      uint8_t* b_u8_unpack_ptr1 = (uint8_t*)b_u8_unpack + (2 * n + 0) + k * N;
      uint8_t* b_u8_unpack_ptr2 = (uint8_t*)b_u8_unpack + (2 * n + 1) + k * N;
      hie::bfloat16* b_bf16_ptr =
          (hie::bfloat16*)b_bf16 + (2 * n + 0) * K_pack + k * 2;
      int kk_max = k + group_size < K ? k + group_size : K;
      for (int kk = k; kk < kk_max; kk += 4) {
        for (int i = 0; i < 4 && (kk + i) < K; i++) {
          int param_idx = subch_idx * N + (2 * n + 0);
          b_bf16_ptr[i + 0] =
              (b_u8_unpack_ptr1[i * N] - zero[param_idx]) * scale[param_idx];
          if ((2 * n + 1) < N) {
            int param_idx = subch_idx * N + (2 * n + 1);
            b_bf16_ptr[i + 4] =
                (b_u8_unpack_ptr2[i * N] - zero[param_idx]) * scale[param_idx];
          } else {
            b_bf16_ptr[i + 4] = 0;
          }
        }
        b_bf16_ptr += 8;
        b_u8_unpack_ptr1 += 4 * N;
        b_u8_unpack_ptr2 += 4 * N;
      }
    }
  });

  return;
}

void gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack,
                                     const float* b_fp32,
                                     hie::bfloat16* b_bf16) {
  int k_tile = 1024;  // empirical var: 1024, 5120
  int k_thread = std::ceil(K_pack * 1.0 / k_tile);

  parallel_for(k_thread, [&](int k) {
    for (int n = 0; n < N; n += 2) {
      float* b_fp32_ptr1 = (float*)b_fp32 + k * k_tile * N + n + 0;
      float* b_fp32_ptr2 = (float*)b_fp32 + k * k_tile * N + n + 1;
      hie::bfloat16* b_bf16_ptr = b_bf16 + n * K_pack + k * k_tile * 2;
      int kk_max = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;
      for (int kk = k * k_tile; kk < kk_max; kk += 4) {
        for (int i = 0; i < 4 && (kk + i < kk_max); i++) {
          b_bf16_ptr[i] = b_fp32_ptr1[i * N];
          if (n != (N - 1)) {
            b_bf16_ptr[i + 4] = b_fp32_ptr2[i * N];
          }
        }
        b_bf16_ptr += 8;
        b_fp32_ptr1 += 4 * N;
        b_fp32_ptr2 += 4 * N;
      }
    }
  });

  return;
}
}  // namespace cpu
}  // namespace allspark
#endif  // ENABLE_ARM_V84_V9
