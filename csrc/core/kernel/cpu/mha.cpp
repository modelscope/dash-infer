/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mha.cpp
 */

#include <cmath>

#include "cpu_common.h"
#include "cpu_kernel.h"
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
        int k = i * seq_length + j;
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
        int k = i * seq_length + j;
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
        int k = m * seq_length * num_heads + j * num_heads + n;
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
        int k = m * seq_length * num_heads + j * num_heads + n;
        ;
        vSoftmax(step, score + k * step);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_length,
                  size_per_head, step, 1, score_buf, step * num_heads,
                  v + kv_offset, kv_stride, 0, out + out_offset, hidden_size);
    });
  }
}
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
        int k = m * seq_len * num_heads + j * num_heads + n;
        vSoftmaxMask(step, score + k * step,
                     mask + (m / beam_size * step + j) * step);
      });
    });
  } else {
    parallel_for(N, [&](int i) {
      int m = i / num_heads;
      int n = i % num_heads;
      parallel_for(seq_len, [&](int j) {
        int k = m * seq_len * num_heads + j * num_heads + n;
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
        int k = m * seq_len * num_heads + j * num_heads + n;
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
