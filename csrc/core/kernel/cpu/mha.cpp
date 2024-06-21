/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mha.cpp
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "cpu_common.h"
#include "cpu_kernel.h"
#include "omp.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined ALLSPARK_USE_NEON_
#include "cpu_neon.h"
#endif
#include <memory.h>

#include <iostream>
#include <limits>
namespace allspark {
namespace cpu {

#if defined(__x86_64__) || defined(_M_X64)
static const uint32_t mask_f32[14] = {
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0,          0,          0,
    0,          0,          0,          0};
void vSoftmax(int n, float* vector, float temperature = 1.0) {
  __m256i mask = _mm256_loadu_si256((__m256i*)mask_f32);
  __m256 max, sum, scale;
  int i;
  float fmax, fsum;

  max = _mm256_set1_ps(-std::numeric_limits<float>::max());
  for (i = 0; i < n - 7; i += 8) {
    max = _mm256_max_ps(max, _mm256_loadu_ps(vector + i) / temperature);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_maskload_ps(vector + i, mask) / temperature;
    src = _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::max()),
                           src, _mm256_castsi256_ps(mask));
    max = _mm256_max_ps(max, src);
  }
  max = _mm256_max_ps(max, _mm256_permute2f128_ps(max, max, 0x1));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0x4E));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0xB1));
  fmax = _mm_cvtss_f32(_mm256_extractf128_ps(max, 1));
  max = _mm256_set1_ps(fmax);

  sum = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i) / temperature;
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    _mm256_storeu_ps(vector + i, poly);
    // sum
    sum = _mm256_add_ps(sum, poly);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask) / temperature;
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    _mm256_maskstore_ps(vector + i, mask, poly);
    // sum
    poly =
        _mm256_blendv_ps(_mm256_setzero_ps(), poly, _mm256_castsi256_ps(mask));
    sum = _mm256_add_ps(sum, poly);
  }
  sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x1));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0x4E));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0xB1));
  fsum = _mm_cvtss_f32(_mm256_extractf128_ps(sum, 1));
  // scale
  scale = _mm256_set1_ps(1.f / fsum);
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i);
    src = _mm256_mul_ps(src, scale);
    _mm256_storeu_ps(vector + i, src);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_mul_ps(src, scale);
    _mm256_maskstore_ps(vector + i, mask, src);
  }
}
void vLogSoftmax(int n, const float* vector, float* output) {
  __m256i mask = _mm256_loadu_si256((__m256i*)mask_f32);
  __m256 max, sum, scale;
  int i;
  float fmax, fsum;

  max = _mm256_set1_ps(-std::numeric_limits<float>::max());
  for (i = 0; i < n - 7; i += 8) {
    max = _mm256_max_ps(max, _mm256_loadu_ps(vector + i));
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::max()),
                           src, _mm256_castsi256_ps(mask));
    max = _mm256_max_ps(max, src);
  }
  max = _mm256_max_ps(max, _mm256_permute2f128_ps(max, max, 0x1));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0x4E));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0xB1));
  fmax = _mm_cvtss_f32(_mm256_extractf128_ps(max, 1));
  max = _mm256_set1_ps(fmax);

  sum = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i);
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    // sum
    sum = _mm256_add_ps(sum, poly);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    // sum
    poly =
        _mm256_blendv_ps(_mm256_setzero_ps(), poly, _mm256_castsi256_ps(mask));
    sum = _mm256_add_ps(sum, poly);
  }
  sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x1));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0x4E));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0xB1));
  fsum = _mm_cvtss_f32(_mm256_extractf128_ps(sum, 1));
  // scale
  scale = _mm256_set1_ps(log(fsum));
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i);
    src = _mm256_sub_ps(src, max);
    src = _mm256_sub_ps(src, scale);
    _mm256_storeu_ps(output + i, src);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_sub_ps(src, max);
    src = _mm256_sub_ps(src, scale);
    _mm256_maskstore_ps(output + i, mask, src);
  }
}
void vSoftmaxMask(int n, float* vector, const float* mask_input) {
  __m256i mask = _mm256_loadu_si256((__m256i*)mask_f32);
  __m256 max, sum, scale;
  int i;
  float fmax, fsum;

  max = _mm256_set1_ps(-std::numeric_limits<float>::max());
  for (i = 0; i < n - 7; i += 8) {
    max = _mm256_max_ps(max, _mm256_loadu_ps(vector + i));
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::max()),
                           src, _mm256_castsi256_ps(mask));
    max = _mm256_max_ps(max, src);
  }
  max = _mm256_max_ps(max, _mm256_permute2f128_ps(max, max, 0x1));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0x4E));
  max = _mm256_max_ps(max, _mm256_shuffle_ps(max, max, 0xB1));
  fmax = _mm_cvtss_f32(_mm256_extractf128_ps(max, 1));
  max = _mm256_set1_ps(fmax);
  sum = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i);
    __m256 mask_in = _mm256_loadu_ps(mask_input + i);
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    poly = _mm256_mul_ps(poly, mask_in);
    _mm256_storeu_ps(vector + i, poly);
    // sum
    sum = _mm256_add_ps(sum, poly);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    __m256 mask_in = _mm256_maskload_ps(mask_input + i, mask);
    src = _mm256_sub_ps(src, max);
    // exp
    // fx = x * log2ef + 0.5
    src =
        _mm256_max_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50)));
    __m256 fx =
        _mm256_mul_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)));
    fx = _mm256_add_ps(fx, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000)));
    // tmp = floor(fx)
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 reduced = _mm256_fnmadd_ps(
        tmp, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)), src);
    // 2^n
    __m256i exp = _mm256_cvtps_epi32(tmp);
    exp = _mm256_add_epi32(exp, _mm256_set1_epi32(0x7f));
    exp = _mm256_slli_epi32(exp, 23);
    // poly
    __m256 poly = _mm256_fmadd_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce)), reduced,
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb)));
    poly = _mm256_fmadd_ps(poly, reduced,
                           _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));
    poly = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp));
    poly = _mm256_mul_ps(poly, mask_in);
    _mm256_maskstore_ps(vector + i, mask, poly);
    // sum
    poly =
        _mm256_blendv_ps(_mm256_setzero_ps(), poly, _mm256_castsi256_ps(mask));
    sum = _mm256_add_ps(sum, poly);
  }
  sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x1));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0x4E));
  sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0xB1));
  fsum = _mm_cvtss_f32(_mm256_extractf128_ps(sum, 1));
  // scale
  scale = _mm256_set1_ps(1.f / fsum);
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(vector + i);
    src = _mm256_mul_ps(src, scale);
    _mm256_storeu_ps(vector + i, src);
  }
  if (i < n) {
    __m256 src = _mm256_maskload_ps(vector + i, mask);
    src = _mm256_mul_ps(src, scale);
    _mm256_maskstore_ps(vector + i, mask, src);
  }
}
#elif defined ALLSPARK_USE_NEON_
float vMax(int n, const float* a) {
  float max = a[0];
  float32x4x4_t max_v;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    max_v.val[i] = vdupq_n_f32(max);
  }
  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(a + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      max_v.val[i] = vmaxq_f32(max_v.val[i], regs.val[i]);
    }
  }
  for (; d < n; ++d) {
    max = std::max(max, a[d]);
  }
  max_v.val[0] = vmaxq_f32(max_v.val[0], max_v.val[1]);
  max_v.val[2] = vmaxq_f32(max_v.val[2], max_v.val[3]);
  max_v.val[0] = vmaxq_f32(max_v.val[0], max_v.val[2]);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    max = std::max(max, max_v.val[0][i]);
  }
  return max;
}
void vSoftmax(int n, float* vector, float temperature = 1.0) {
  int d = 0;
  if (temperature != 1.f) {
    // vector[i] /= temperature;
    const float32x4_t temp_v = vdupq_n_f32(1 / temperature);
    for (; d <= n - 16; d += 16) {
      float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        regs.val[i] = vmulq_f32(regs.val[i], temp_v);
      }
      vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
      vector[d] /= temperature;
    }
  }
  // Find Max
  const float max_val = vMax(n, vector);
  const float32x4_t max_v = vdupq_n_f32(max_val);
  float reduce_sum = 0.0f;
  float32x4_t reduce_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum_v[i] = vdupq_n_f32(0.0f);
  }

  // Sub Max and Exp and ReduceSum
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vexpq_f32(vsubq_f32(regs.val[i], max_v));
      reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
    }
    vst1q_f32_x4(vector + d, regs);
  }
  for (; d < n; ++d) {
    float val = vector[d];
    val = std::exp(val - max_val);
    reduce_sum += val;
    vector[d] = val;
  }
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[1]);
  reduce_sum_v[2] = vaddq_f32(reduce_sum_v[2], reduce_sum_v[3]);
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum += reduce_sum_v[0][i];
  }

  // Div ReduceSum
  const float reduce_sum_mul = 1.0f / reduce_sum;
  const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vmulq_f32(regs.val[i], reduce_sum_mul_v);
    }
    vst1q_f32_x4(vector + d, regs);
  }
  for (; d < n; ++d) {
    vector[d] = vector[d] * reduce_sum_mul;
  }
}
void vLogSoftmax(int n, const float* vector, float* output) {
  // Find Max
  const float max_val = vMax(n, vector);
  const float32x4_t max_v = vdupq_n_f32(max_val);
  float reduce_sum = 0.0f;
  float32x4_t reduce_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum_v[i] = vdupq_n_f32(0.0f);
  }

  // Sub Max and Exp and ReduceSum
  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vexpq_f32(vsubq_f32(regs.val[i], max_v));
      reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
    }
    vst1q_f32_x4(output + d, regs);
  }
  for (; d < n; ++d) {
    float val = vector[d];
    val = std::exp(val - max_val);
    reduce_sum += val;
    output[d] = val;
  }
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[1]);
  reduce_sum_v[2] = vaddq_f32(reduce_sum_v[2], reduce_sum_v[3]);
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum += reduce_sum_v[0][i];
  }

  // Div ReduceSum
  const float reduce_sum_mul = 1.0f / reduce_sum;
  const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(output + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vmulq_f32(regs.val[i], reduce_sum_mul_v);
      regs.val[i] = vlogq_f32(regs.val[i]);
    }
    vst1q_f32_x4(output + d, regs);
  }
  for (; d < n; ++d) {
    output[d] = std::log(output[d] * reduce_sum_mul);
  }
}
void vSoftmaxMask(int n, float* vector, const float* mask_input) {
  // set vector based on mask
  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4_t mask_val = vdupq_n_f32(-100000.0f);
    float32x4_t one = vdupq_n_f32(1.f);
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
    float32x4x4_t mask_v = vld1q_f32_x4(mask_input + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      mask_v.val[i] = vsubq_f32(one, mask_v.val[i]);
      mask_v.val[i] = vmulq_f32(mask_val, mask_v.val[i]);
      regs.val[i] = vaddq_f32(mask_v.val[i], regs.val[i]);
    }
    vst1q_f32_x4(vector + d, regs);
  }
  for (; d < n; ++d) {
    vector[d] = vector[d] + (1.0f - mask_input[d]) * (-100000.f);
  }
  // Find Max
  const float max_val = vMax(n, vector);
  const float32x4_t max_v = vdupq_n_f32(max_val);
  float reduce_sum = 0.0f;
  float32x4_t reduce_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum_v[i] = vdupq_n_f32(0.0f);
  }

  // Sub Max and Exp and ReduceSum
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vexpq_f32(vsubq_f32(regs.val[i], max_v));
      reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
    }
    vst1q_f32_x4(vector + d, regs);
  }
  for (; d < n; ++d) {
    float val = vector[d];
    val = std::exp(val - max_val);
    reduce_sum += val;
    vector[d] = val;
  }
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[1]);
  reduce_sum_v[2] = vaddq_f32(reduce_sum_v[2], reduce_sum_v[3]);
  reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reduce_sum += reduce_sum_v[0][i];
  }

  // Div ReduceSum
  const float reduce_sum_mul = 1.0f / (reduce_sum + 1e-12f);
  const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vmulq_f32(regs.val[i], reduce_sum_mul_v);
    }
    vst1q_f32_x4(vector + d, regs);
  }
  for (; d < n; ++d) {
    vector[d] = vector[d] * reduce_sum_mul;
  }
}
#endif

template <>
void MHAKernel(float* out, const float* q, const float* k, const float* v,
               const float* mask, float* score, int batch_size, int num_heads,
               int seq_length, int step, int hidden_size, int size_per_head,
               float alpha) {
  int gemm_batch_size = batch_size * num_heads;
  int qkv_stride = hidden_size * 3;
  float beta = 0.0;
  if (mask) {
    parallel_for(gemm_batch_size, [&](int i) {
      int m = i / num_heads;  // batch_size
      int n = i % num_heads;  // num_heads
      int out_offset = m * seq_length * hidden_size + n * size_per_head;
      int qkv_offset = m * seq_length * qkv_stride + n * size_per_head;
      float* score_buf =
          score + m * num_heads * seq_length * step + n * seq_length * step;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_length, step,
                  size_per_head, alpha, q + qkv_offset, qkv_stride,
                  k + qkv_offset, qkv_stride, beta, score_buf, step);
      for (int j = 0; j < seq_length; ++j) {
        size_t k = i * seq_length + j;
        vSoftmaxMask(step, score + k * step,
                     mask + (m * seq_length + j) * step);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_length,
                  size_per_head, step, 1, score_buf, step, v + qkv_offset,
                  qkv_stride, 0, out + out_offset, hidden_size);
    });
  } else {
    parallel_for(gemm_batch_size, [&](int i) {
      int m = i / num_heads;  // batch_size
      int n = i % num_heads;  // num_heads
      int out_offset = m * seq_length * hidden_size + n * size_per_head;
      int qkv_offset = m * seq_length * qkv_stride + n * size_per_head;
      float* score_buf =
          score + m * num_heads * seq_length * step + n * seq_length * step;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_length, step,
                  size_per_head, alpha, q + qkv_offset, qkv_stride,
                  k + qkv_offset, qkv_stride, beta, score_buf, step);
      for (int j = 0; j < seq_length; ++j) {
        size_t k = i * seq_length + j;
        vSoftmax(step, score + k * step);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_length,
                  size_per_head, step, 1, score_buf, step, v + qkv_offset,
                  qkv_stride, 0, out + out_offset, hidden_size);
    });
  }
}
template <>
void MHAKernel(float* out, const float* q, const float* k, const float* v,
               const float* mask, float* score, int batch_size, int beam_size,
               int num_heads, int seq_length, int step, int hidden_size,
               int size_per_head, int q_stride, int kv_stride, int max_seq_len,
               float alpha) {
  int gemm_batch_size = batch_size * num_heads;
  float beta = 0.0;
  if (mask) {
    parallel_for(gemm_batch_size, [&](int i) {
      int m = i / num_heads;  // batch_size
      int n = i % num_heads;  // num_heads
      int out_offset = m * seq_length * hidden_size + n * size_per_head;
      int q_offset = m * seq_length * q_stride + n * size_per_head;
      int kv_offset = m * max_seq_len * kv_stride + n * size_per_head;
      float* score_buf = score + m * num_heads * seq_length * step + n * step;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_length, step,
                  size_per_head, alpha, q + q_offset, q_stride, k + kv_offset,
                  kv_stride, beta, score_buf, step * num_heads);
      for (int j = 0; j < seq_length; ++j) {
        size_t k = m * seq_length * num_heads + j * num_heads + n;
        vSoftmaxMask(step, score + k * step,
                     mask + (m / beam_size * step + j) * step);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_length,
                  size_per_head, step, 1, score_buf, step * num_heads,
                  v + kv_offset, kv_stride, 0, out + out_offset, hidden_size);
    });
  } else {
    parallel_for(gemm_batch_size, [&](int i) {
      int m = i / num_heads;  // batch_size
      int n = i % num_heads;  // num_heads
      int out_offset = m * seq_length * hidden_size + n * size_per_head;
      int q_offset = m * seq_length * q_stride + n * size_per_head;
      int kv_offset = m * max_seq_len * kv_stride + n * size_per_head;
      float* score_buf = score + m * num_heads * seq_length * step + n * step;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_length, step,
                  size_per_head, alpha, q + q_offset, q_stride, k + kv_offset,
                  kv_stride, beta, score_buf, step * num_heads);
      for (int j = 0; j < seq_length; ++j) {
        size_t k = m * seq_length * num_heads + j * num_heads + n;
        ;
        vSoftmax(step, score + k * step);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_length,
                  size_per_head, step, 1, score_buf, step * num_heads,
                  v + kv_offset, kv_stride, 0, out + out_offset, hidden_size);
    });
  }
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
static inline __m512 to_lowest_mask(__mmask16 mask, __m512 vmask) {
  const __m512 vone = _mm512_set1_ps(1.0);
  const __m512 vmax = _mm512_set1_ps(std::numeric_limits<float>::max());
  __m512 vmask_rev = vmask - vone;
  __m512 vmask_lowest = vmask_rev * vmax;
  return vmask_lowest;
}

static inline __m512 __m512_vexp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // exp(x) = exp(m*ln(2) + r)
  // m = floor(x/ln(2) + 0.5)
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

  // r = x - m*ln(2).
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

static void vSoftmaxTile(float* AB, float* ABout, float* sum, float* max,
                         float* preSum, float* preMax, float scale,
                         const float* attnMask, int m, int k,
                         int attnMskStride) {
  float maxVal = std::numeric_limits<float>::lowest();
  __m512 vscale = _mm512_set1_ps(scale);
  for (int i = 0; i < m; ++i) {
    float* buf = AB + i * k;
    float* obuf = ABout + i * k;
    const float* attnMsk = attnMask + i * attnMskStride;
    // max val for avoiding inf and nan
    __m512 vmax = _mm512_set1_ps(maxVal);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
      __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
      __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMsk + off);
      __m512 vmask_lowest = to_lowest_mask(mask, vmask);
      vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx * vscale + vmask_lowest);
    }
    float _max = _mm512_reduce_max_ps(vmax);

    _max = _max > max[i] ? _max : max[i];
    __m512 merr = _mm512_set1_ps(max[i] - _max);
    merr = __m512_vexp(merr);
    max[i] = _max;

    // exp and get sum
    __m512 vsum = _mm512_set1_ps(0);
    vmax = _mm512_set1_ps(_max);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
      __m512 vmask = _mm512_maskz_loadu_ps(mask, attnMsk + off);
      __m512 vmask_lowest = to_lowest_mask(mask, vmask);
      vx = __m512_vexp(vx * vscale + vmask_lowest - vmax);
      _mm512_mask_storeu_ps(obuf + off, mask, vx);

      vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
    }
    float _sum = _mm512_reduce_add_ps(vsum);
    float fac = _mm512_cvtss_f32(merr);
    sum[i] = sum[i] * fac + _sum;
    _sum = sum[i];

    // Compute exp/sum(exp) and store
    __m512 vrsum = _mm512_set1_ps(1.0f / _sum);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      __m512 vx = _mm512_maskz_loadu_ps(mask, obuf + off);
      vx = vx * vrsum;

      _mm512_mask_storeu_ps(obuf + off, mask, vx);
    }
  }
}

static void vUpdateOutTile(float* output, const float* expABC, float* preSum,
                           float* sum, float* preMax, float* max, int m, int n,
                           int stride) {
  for (int i = 0; i < m; ++i) {
    const float* buf = expABC + i * n;
    float* outbuf = output + i * stride;
    __m512 merr = _mm512_set1_ps(preMax[i] - max[i]);
    merr = __m512_vexp(merr);
    __m512 vfac = _mm512_set1_ps(preSum[i] / sum[i]);
    for (int off = 0; off < n; off += 16) {
      int remain = n - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
      __m512 vout = _mm512_maskz_loadu_ps(mask, outbuf + off);
      __m512 vabc = _mm512_maskz_loadu_ps(mask, buf + off);
      __m512 vupt = vout * merr * vfac + vabc;
      _mm512_mask_storeu_ps(outbuf + off, mask, vupt);
    }
    preSum[i] = sum[i];
    preMax[i] = max[i];
  }
}

// output = softmax(AB/scale)*C
static void vIncrementalTileAttention(
    const float* A, const float* B, const float* C, const float* mask, int m,
    int n, int k, int mask_stride, float* pre_sum, float* sum, float* pre_max,
    float* max, float scale, float* AB, float* expABC, float* output,
    int q_stride, int k_stride, int v_stride, int stride) {
  // AB = S_ij = Q_i K^T_j
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, 1.0, A,
              q_stride, B, k_stride, 0, AB, k);

  // AB = P_ij = softmax(S_ij / scale)
  vSoftmaxTile(AB, AB, sum, max, pre_sum, pre_max, scale, mask, m, k,
               mask_stride);

  // expABC = P_ij V_j
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, AB, k, C,
              v_stride, 0.0, expABC, n);
  // output = O_i = preSum/sum * O_i + expABC / sum
  vUpdateOutTile(output, expABC, pre_sum, sum, pre_max, max, m, n, stride);
}

template <>
void SelfScaledDpAttention(float* output, const float* query, const float* key,
                           const float* value, int q_num_heads,
                           int kv_num_heads, int size_per_head, int o_stride,
                           int q_stride, int kv_stride, int batch_size,
                           const int* input_seq_lens, const int* past_seq_lens,
                           void* workspace, int src_blk, int tgt_blk,
                           const float* mask, float scale, int num_thread) {
  // output = softmax(query * trans(key)) * value

  // get the max seq_len
  int max_src_len = 0, max_tgt_len = 0;
  for (int i = 0; i < batch_size; ++i) {
    max_src_len = std::max(max_src_len, input_seq_lens[i]);
    max_tgt_len = std::max(max_tgt_len, input_seq_lens[i] + past_seq_lens[i]);
  }
  // compute the seq_start_loc
  int seq_start_loc[batch_size + 1];
  seq_start_loc[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    seq_start_loc[i + 1] = seq_start_loc[i] + input_seq_lens[i];
  }

  int num_group = q_num_heads / kv_num_heads;

  constexpr int NUM_ARR = 7;
  // 4: pre_sum, sum, pre_max, max; tgt_blk: exp_qkv; 2: Q_i, PV_i
  int arr_stride = (4 + tgt_blk + 2 * size_per_head) * src_blk;
  int64_t thr_buf_size = sizeof(float) * num_thread * arr_stride;
  int64_t thr_ptr_buf_size = sizeof(float*) * num_thread * NUM_ARR;

  float* thr_buf = (float*)workspace;
  float** thr_ptr_buf = (float**)((uint8_t*)workspace + thr_buf_size);

  float** pre_sum = thr_ptr_buf;
  float** sum = thr_ptr_buf + num_thread;
  float** pre_max = thr_ptr_buf + num_thread * 2;
  float** max = thr_ptr_buf + num_thread * 3;
  float** qk_arr = thr_ptr_buf + num_thread * 4;
  float** exp_qkv_arr = thr_ptr_buf + num_thread * 5;
  float** q_arr = thr_ptr_buf + num_thread * 6;

  for (int i = 0; i < num_thread; ++i) {
    // l
    pre_sum[i] = thr_buf + src_blk * i;
    // l^new
    sum[i] = thr_buf + src_blk * num_thread + src_blk * i;
    // m
    pre_max[i] = thr_buf + src_blk * num_thread * 2 + src_blk * i;
    // m^new
    max[i] = thr_buf + src_blk * num_thread * 3 + src_blk * i;
    // S
    qk_arr[i] = thr_buf + src_blk * num_thread * 4 + src_blk * tgt_blk * i;
    // PV
    exp_qkv_arr[i] = thr_buf + src_blk * num_thread * (4 + tgt_blk) +
                     src_blk * size_per_head * i;
    // Q
    q_arr[i] = thr_buf + src_blk * num_thread * (4 + tgt_blk + size_per_head) +
               src_blk * size_per_head * i;
  }

#pragma omp parallel for collapse(3) schedule(dynamic)
  for (uint64_t b = 0; b < batch_size; ++b) {
    for (int h = 0; h < q_num_heads; ++h) {
      for (int m = 0; m < max_src_len; m += src_blk) {
        int src_len = input_seq_lens[b];
        int tgt_len = input_seq_lens[b] + past_seq_lens[b];
        if (m >= src_len) {
          continue;
        }

        int tid = omp_get_thread_num();
        int q_real_blk = std::min(src_blk, src_len - m);
        uint64_t src_off = seq_start_loc[b] * q_stride + h * size_per_head;
        uint64_t out_off = seq_start_loc[b] * o_stride + h * size_per_head;
        const float* q_buf = query + src_off + m * q_stride;
        float* q = q_arr[tid];
        float* out = output + out_off + m * o_stride;

        // reset out
        for (int ii = 0; ii < q_real_blk; ++ii) {
#pragma omp simd
          for (int jj = 0; jj < size_per_head; ++jj) {
            out[ii * o_stride + jj] = 0;  // reset output
            // TODO: do we need make a copy, rather than using q_buf directly?
            q[ii * size_per_head + jj] = q_buf[ii * q_stride + jj];
          }
        }
        // reset sum
#pragma omp simd
        for (int ii = 0; ii < q_real_blk; ++ii) {
          pre_sum[tid][ii] = 0;
          sum[tid][ii] = 0;
          pre_max[tid][ii] = std::numeric_limits<float>::lowest();
          max[tid][ii] = std::numeric_limits<float>::lowest();
        }

        uint64_t tgt_off =
            seq_start_loc[b] * kv_stride + (h / num_group) * size_per_head;
        const float* k = key + tgt_off;
        const float* v = value + tgt_off;
        for (int n = 0; n < tgt_len; n += tgt_blk) {
          int kv_real_blk = std::min(tgt_blk, tgt_len - n);
          // mask out.
          if (m + q_real_blk - 1 < n) {
            break;
          }

          const float* k_blk = k + n * kv_stride;
          const float* v_blk = v + n * kv_stride;
          const float* mask_blk =
              mask + seq_start_loc[b] * tgt_len + m * tgt_len + n;
          vIncrementalTileAttention(
              q, k_blk, v_blk, mask_blk, q_real_blk, size_per_head, kv_real_blk,
              tgt_len, pre_sum[tid], sum[tid], pre_max[tid], max[tid], scale,
              qk_arr[tid], exp_qkv_arr[tid], out, size_per_head, kv_stride,
              kv_stride, o_stride);
        }
      }
    }
  }
}
#endif

template <>
void LogSoftmaxKernel(const float* input, float* output, int outer_dim,
                      int inner_dim) {
  parallel_for(outer_dim, [&](int out) {
    vLogSoftmax(inner_dim, input + out * inner_dim, output + out * inner_dim);
  });
}
template <>
void SoftmaxKernel(float* input, int* len_arr, int outer_dim, int inner_dim,
                   float temperature) {
  parallel_for(outer_dim, [&](int out) {
    vSoftmax(len_arr[out], input + out * inner_dim, temperature);
  });
}
template <typename T>
void UpdateKVLauncher(T* k, T* v, const T* step_k, const T* step_v,
                      int batch_size, int step, int max_length, int hidden_size,
                      int seq_len, int stride) {
  int N = batch_size * hidden_size * seq_len;
  parallel_for(N, [&](int tid) {
    int idx1 = tid / (seq_len * hidden_size);
    int idx2 = tid % (seq_len * hidden_size) / hidden_size;
    int idx3 = tid % hidden_size;
    int src_idx = idx1 * seq_len * stride + idx2 * stride + idx3;
    int dst_idx =
        (idx1 * max_length + step) * hidden_size + idx2 * hidden_size + idx3;
    k[dst_idx] = step_k[src_idx];
    v[dst_idx] = step_v[src_idx];
  });
}
template void UpdateKVLauncher<float>(float* k, float* v, const float* step_k,
                                      const float* step_v, int batch_size,
                                      int step, int max_length, int hidden_size,
                                      int seq_len, int stride);
template <>
void GetBatchArrayLauncher<float>(
    float* q, float* k, float* v, float* score, float* out, float** q_array,
    float** k_array, float** v_array, float** score_array, float** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head, int step,
    int q_stride, int kv_stride, int score_stride, int out_stride) {
  const int N = batch_size * beam_size * num_heads;
  parallel_for(N, [&](int tid) {
    int j = tid % num_heads;
    int i = tid / num_heads;
    q_array[tid] = q + i * q_stride + j * size_per_head;
    k_array[tid] = k + i / beam_size * kv_stride + j * size_per_head;
    v_array[tid] = v + i / beam_size * kv_stride + j * size_per_head;
    score_array[tid] = score + i * score_stride + j * step;
    out_array[tid] = out + i * out_stride + j * size_per_head;
  });
}

template <>
void MultiQueryGetBatchArrayLauncher<float>(
    float* q, float* k, float* v, float* score, float* out, float** q_array,
    float** k_array, float** v_array, float** score_array, float** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head,
    int group_num, int step, int q_stride, int kv_stride, int score_stride,
    int out_stride) {
  const int N = batch_size * beam_size * num_heads;
  parallel_for(N, [&](int tid) {
    int j = tid % num_heads;
    int i = tid / num_heads;
    q_array[tid] = q + i * q_stride + j * size_per_head;
    int group_now = j / (num_heads / group_num);
    k_array[tid] = k + i / beam_size * kv_stride + group_now * size_per_head;
    v_array[tid] = v + i / beam_size * kv_stride + group_now * size_per_head;
    score_array[tid] = score + i * score_stride + j * step;
    out_array[tid] = out + i * out_stride + j * size_per_head;
  });
}

template <>
void BatchGemmWraper<float>(void** matrix_C, void** matrix_A, void** matrix_B,
                            int m, int n, int k, bool transA, bool transB,
                            float alpha, float beta, int lda, int ldb, int ldc,
                            int batch) {
  CBLAS_TRANSPOSE transA_ = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB_ = transB ? CblasTrans : CblasNoTrans;
  parallel_for(batch, [&](int i) {
    float* A = (float*)matrix_A[i];
    float* B = (float*)matrix_B[i];
    float* C = (float*)matrix_C[i];
    cblas_sgemm(CblasRowMajor, transA_, transB_, m, n, k, alpha, A, lda, B, ldb,
                beta, C, ldc);
  });
}
template <>
void BatchSoftmax<float>(float* score, const float* mask, int batch_size,
                         int beam_size, int num_heads, int seq_len, int step) {
  const int N = batch_size * num_heads;
  if (mask) {
    parallel_for(N, [&](int i) {
      int m = i / num_heads;
      int n = i % num_heads;
      parallel_for(seq_len, [&](int j) {
        size_t k = m * seq_len * num_heads + j * num_heads + n;
        vSoftmaxMask(step, score + k * step,
                     mask + (m / beam_size * step + j) * step);
      });
    });
  } else {
    parallel_for(N, [&](int i) {
      int m = i / num_heads;
      int n = i % num_heads;
      parallel_for(seq_len, [&](int j) {
        size_t k = m * seq_len * num_heads + j * num_heads + n;
        vSoftmax(step, score + k * step);
      });
    });
  }
}

template <>
void BatchDecoderSoftmax<float>(float* score, const float* mask, int batch_size,
                                int beam_size, int num_heads, int seq_len,
                                int step, int input_len) {
  const int N = batch_size * num_heads;
  if (mask) {
    parallel_for(N, [&](int i) {
      int m = i / num_heads;
      int n = i % num_heads;
      std::vector<float> mask_in(step, 1.0f);
      memcpy(mask_in.data(),
             mask + (m / beam_size * input_len + input_len - 1) * input_len,
             input_len * sizeof(float));
      parallel_for(seq_len, [&](int j) {
        size_t k = m * seq_len * num_heads + j * num_heads + n;
        vSoftmaxMask(step, score + k * step, mask_in.data());
      });
    });
  }
}

template <>
void SimpleAdd<float>(float* out, const float* in1, const float* in2,
                      int count) {
  parallel_for(count, [&](int tid) { out[tid] = in1[tid] + in2[tid]; });
}
}  // namespace cpu
}  // namespace allspark
