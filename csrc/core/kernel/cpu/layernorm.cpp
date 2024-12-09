/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    layernorm.cpp
 */

#include <cmath>

#include "cpu_common.h"
#include "cpu_kernel.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined ALLSPARK_USE_NEON_
#include "cpu_neon.h"
#endif
#include <iostream>
namespace allspark {
namespace cpu {
#if defined(__x86_64__) || defined(_M_X64)
void vNorm(int n, const float* input, float* array, const float* gamma,
           const float* beta, const float* bias, float eps, bool use_bias) {
  float mean = .0f, variance = .0f;
  // calculate mean
  float local_out[n];
  for (int i = 0; i < n; i++) {
    if (use_bias) {
      local_out[i] = input[i] + bias[i];
    } else {
      local_out[i] = input[i];
    }
    mean += local_out[i];
    variance += local_out[i] * local_out[i];
  }
  mean /= (float)n;
  variance /= (float)n;
  // calculate inverse of the variance
  variance = 1.f / sqrtf(variance - mean * mean + eps);
  // normalization
  for (int i = 0; i < n; i++) {
    array[i] = gamma[i] * (local_out[i] - mean) * variance + beta[i];
  }
}
static const uint32_t mask_f32[14] = {
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0,          0,          0,
    0,          0,          0,          0};
void layerNorm(int n, const float* input, float* array, const float* gamma,
               const float* beta, float eps) {
  __m256 mean, variance;
  __m256i mask;
  int i;
  float fmean, fvariance;
  // calculate mean
  mean = _mm256_setzero_ps();
  variance = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(input + i);
    mean = _mm256_add_ps(mean, src);
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_maskload_ps(input + i, mask);
    src = _mm256_blendv_ps(_mm256_setzero_ps(), src, _mm256_castsi256_ps(mask));
    mean = _mm256_add_ps(mean, src);
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  mean = _mm256_add_ps(mean, _mm256_permute2f128_ps(mean, mean, 0x1));
  mean = _mm256_add_ps(mean, _mm256_shuffle_ps(mean, mean, 0x4E));
  mean = _mm256_add_ps(mean, _mm256_shuffle_ps(mean, mean, 0xB1));
  fmean = _mm_cvtss_f32(_mm256_extractf128_ps(mean, 1));

  variance =
      _mm256_add_ps(variance, _mm256_permute2f128_ps(variance, variance, 0x1));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0x4E));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0xB1));
  fvariance = _mm_cvtss_f32(_mm256_extractf128_ps(variance, 1));

  fmean /= (float)n;
  fvariance /= (float)n;
  fvariance = 1.f / sqrtf(fvariance - fmean * fmean + eps);

  mean = _mm256_set1_ps(fmean);
  variance = _mm256_set1_ps(fvariance);
  // normalization
  for (i = 0; i < n - 7; i += 8) {
    __m256 result = _mm256_add_ps(
        _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(gamma + i),
                          _mm256_sub_ps(_mm256_loadu_ps(input + i), mean)),
            variance),
        _mm256_loadu_ps(beta + i));
    _mm256_storeu_ps(array + i, result);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 result = _mm256_add_ps(
        _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_maskload_ps(gamma + i, mask),
                _mm256_sub_ps(_mm256_maskload_ps(input + i, mask), mean)),
            variance),
        _mm256_maskload_ps(beta + i, mask));
    _mm256_maskstore_ps(array + i, mask, result);
  }
}
void layerNormNobeta(int n, const float* input, float* array,
                     const float* gamma, float eps) {
  __m256 variance;
  __m256i mask;
  int i;
  float fvariance;
  variance = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src = _mm256_loadu_ps(input + i);
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_maskload_ps(input + i, mask);
    src = _mm256_blendv_ps(_mm256_setzero_ps(), src, _mm256_castsi256_ps(mask));
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }

  variance =
      _mm256_add_ps(variance, _mm256_permute2f128_ps(variance, variance, 0x1));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0x4E));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0xB1));
  fvariance = _mm_cvtss_f32(_mm256_extractf128_ps(variance, 1));

  fvariance /= (float)n;
  fvariance = 1.f / sqrtf(fvariance + eps);

  variance = _mm256_set1_ps(fvariance);
  // normalization
  for (i = 0; i < n - 7; i += 8) {
    __m256 result = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_loadu_ps(gamma + i), _mm256_loadu_ps(input + i)),
        variance);
    _mm256_storeu_ps(array + i, result);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 result =
        _mm256_mul_ps(_mm256_mul_ps(_mm256_maskload_ps(gamma + i, mask),
                                    _mm256_maskload_ps(input + i, mask)),
                      variance);
    _mm256_maskstore_ps(array + i, mask, result);
  }
}
void layerNorm_bias(int n, const float* input, float* array, const float* gamma,
                    const float* beta, const float* bias, float eps) {
  __m256 mean, variance;
  __m256i mask;
  int i;
  float fmean, fvariance;
  // calculate mean
  mean = _mm256_setzero_ps();
  variance = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src =
        _mm256_add_ps(_mm256_loadu_ps(input + i), _mm256_loadu_ps(bias + i));
    mean = _mm256_add_ps(mean, src);
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_add_ps(_mm256_maskload_ps(input + i, mask),
                               _mm256_maskload_ps(bias + i, mask));

    src = _mm256_blendv_ps(_mm256_setzero_ps(), src, _mm256_castsi256_ps(mask));
    mean = _mm256_add_ps(mean, src);
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  mean = _mm256_add_ps(mean, _mm256_permute2f128_ps(mean, mean, 0x1));
  mean = _mm256_add_ps(mean, _mm256_shuffle_ps(mean, mean, 0x4E));
  mean = _mm256_add_ps(mean, _mm256_shuffle_ps(mean, mean, 0xB1));
  fmean = _mm_cvtss_f32(_mm256_extractf128_ps(mean, 1));

  variance =
      _mm256_add_ps(variance, _mm256_permute2f128_ps(variance, variance, 0x1));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0x4E));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0xB1));
  fvariance = _mm_cvtss_f32(_mm256_extractf128_ps(variance, 1));

  fmean /= (float)n;
  fvariance /= (float)n;
  fvariance = 1.f / sqrtf(fvariance - fmean * fmean + eps);

  mean = _mm256_set1_ps(fmean);
  variance = _mm256_set1_ps(fvariance);
  // normalization
  for (i = 0; i < n - 7; i += 8) {
    __m256 result = _mm256_add_ps(
        _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_loadu_ps(gamma + i),
                _mm256_sub_ps(_mm256_add_ps(_mm256_loadu_ps(input + i),
                                            _mm256_loadu_ps(bias + i)),
                              mean)),
            variance),
        _mm256_loadu_ps(beta + i));
    _mm256_storeu_ps(array + i, result);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 result = _mm256_add_ps(
        _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_maskload_ps(gamma + i, mask),
                _mm256_sub_ps(_mm256_add_ps(_mm256_maskload_ps(input + i, mask),
                                            _mm256_maskload_ps(bias + i, mask)),
                              mean)),
            variance),
        _mm256_maskload_ps(beta + i, mask));
    _mm256_maskstore_ps(array + i, mask, result);
  }
}
void layerNormNobeta_bias(int n, const float* input, float* array,
                          const float* gamma, const float* bias, float eps) {
  __m256 variance;
  __m256i mask;
  int i;
  float fvariance;
  variance = _mm256_setzero_ps();
  for (i = 0; i < n - 7; i += 8) {
    __m256 src =
        _mm256_add_ps(_mm256_loadu_ps(input + i), _mm256_loadu_ps(bias + i));
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 src = _mm256_add_ps(_mm256_maskload_ps(input + i, mask),
                               _mm256_maskload_ps(bias + i, mask));
    src = _mm256_blendv_ps(_mm256_setzero_ps(), src, _mm256_castsi256_ps(mask));
    __m256 input_2 = _mm256_mul_ps(src, src);
    variance = _mm256_add_ps(variance, input_2);
  }

  variance =
      _mm256_add_ps(variance, _mm256_permute2f128_ps(variance, variance, 0x1));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0x4E));
  variance =
      _mm256_add_ps(variance, _mm256_shuffle_ps(variance, variance, 0xB1));
  fvariance = _mm_cvtss_f32(_mm256_extractf128_ps(variance, 1));
  fvariance /= (float)n;
  fvariance = 1.f / sqrtf(fvariance + eps);
  variance = _mm256_set1_ps(fvariance);
  // normalization
  for (i = 0; i < n - 7; i += 8) {
    __m256 result =
        _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(gamma + i),
                                    _mm256_add_ps(_mm256_loadu_ps(input + i),
                                                  _mm256_loadu_ps(bias + i))),
                      variance);
    _mm256_storeu_ps(array + i, result);
  }
  if (i < n) {
    mask = _mm256_loadu_si256((__m256i*)(mask_f32 + 7 - (n - i)));
    __m256 result = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_maskload_ps(gamma + i, mask),
                      _mm256_add_ps(_mm256_maskload_ps(input + i, mask),
                                    _mm256_maskload_ps(bias + i, mask))),
        variance);
    _mm256_maskstore_ps(array + i, mask, result);
  }
}
#elif defined ALLSPARK_USE_NEON_
void layerNorm(int n, const float* input, float* array, const float* gamma,
               const float* beta, float eps) {
  float32x4_t sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_v[i] = vdupq_n_f32(0.0f);
  }
  float32x4_t square_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum_v[i] = vdupq_n_f32(0.0f);
  }

  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sum_v[i] = vaddq_f32(sum_v[i], regs.val[i]);
      square_sum_v[i] =
          vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
    }
  }
  float32_t sum = 0.0f;
  float32_t square_sum = 0.0f;
  for (; d < n; ++d) {
    float val = input[d];
    sum += val;
    square_sum += val * val;
  }

  sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
  sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
  sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum += sum_v[0][i];
  }

  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
  square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum += square_sum_v[0][i];
  }

  float mean = sum / n;
  float variance = square_sum / n;
  variance = 1.0f / std::sqrt(variance - mean * mean + eps);
  float32x4_t mean_v = vdupq_n_f32(mean);
  float32x4_t variance_v = vdupq_n_f32(variance);

  // normalization
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t input_v = vld1q_f32_x4(input + d);
    float32x4x4_t gamma_v = vld1q_f32_x4(gamma + d);
    float32x4x4_t beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
      input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
      input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
      input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
    }
    vst1q_f32_x4(array + d, input_v);
  }
  for (; d < n; ++d) {
    array[d] = (input[d] - mean) * gamma[d] * variance + beta[d];
  }
}
void layerNormNobeta(int n, const float* input, float* array,
                     const float* gamma, float eps) {
  //     float32x4_t sum_v[4];
  // #pragma unroll
  //     for (int i = 0; i < 4; ++i) {
  //         sum_v[i] = vdupq_n_f32(0.0f);
  //     }

  float32x4_t square_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum_v[i] = vdupq_n_f32(0.0f);
  }

  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // sum_v[i] = vaddq_f32(sum_v[i], regs.val[i]);
      square_sum_v[i] =
          vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
    }
  }
  float32_t sum = 0.0f;
  float32_t square_sum = 0.0f;
  for (; d < n; ++d) {
    float val = input[d];
    sum += val;
    square_sum += val * val;
  }

  //     sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
  //     sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
  //     sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
  // #pragma unroll
  //     for (int i = 0; i < 4; ++i) {
  //         sum += sum_v[0][i];
  //     }

  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
  square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum += square_sum_v[0][i];
  }

  // float mean = sum / n;
  float variance = square_sum / n;
  variance = 1.0f / std::sqrt(variance + eps);
  // variance = 1.0f / std::sqrt(variance - mean * mean + eps);
  // float32x4_t mean_v = vdupq_n_f32(mean);
  float32x4_t variance_v = vdupq_n_f32(variance);

  // normalization
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t input_v = vld1q_f32_x4(input + d);
    float32x4x4_t gamma_v = vld1q_f32_x4(gamma + d);
    // float32x4x4_t beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
      input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
      input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
      // input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
    }
    vst1q_f32_x4(array + d, input_v);
  }
  for (; d < n; ++d) {
    array[d] = input[d] * gamma[d] * variance;  // + beta[d];
    // array[d] = (input[d] - mean) * gamma[d] * variance;  // + beta[d];
  }
}
void layerNorm_bias(int n, const float* input, float* array, const float* gamma,
                    const float* beta, const float* bias, float eps) {
  float32x4_t sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_v[i] = vdupq_n_f32(0.0f);
  }
  float32x4_t square_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum_v[i] = vdupq_n_f32(0.0f);
  }

  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(input + d);
    float32x4x4_t bias_v = vld1q_f32_x4(bias + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vaddq_f32(regs.val[i], bias_v.val[i]);
      sum_v[i] = vaddq_f32(sum_v[i], regs.val[i]);
      square_sum_v[i] =
          vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
    }
  }
  float32_t sum = 0.0f;
  float32_t square_sum = 0.0f;
  for (; d < n; ++d) {
    float val = input[d] + bias[d];
    sum += val;
    square_sum += val * val;
  }

  sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
  sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
  sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum += sum_v[0][i];
  }

  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
  square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum += square_sum_v[0][i];
  }

  float mean = sum / n;
  float variance = square_sum / n;
  variance = 1.0f / std::sqrt(variance - mean * mean + eps);
  float32x4_t mean_v = vdupq_n_f32(mean);
  float32x4_t variance_v = vdupq_n_f32(variance);

  // normalization
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t input_v = vld1q_f32_x4(input + d);
    float32x4x4_t bias_v = vld1q_f32_x4(bias + d);
    float32x4x4_t gamma_v = vld1q_f32_x4(gamma + d);
    float32x4x4_t beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      input_v.val[i] = vaddq_f32(input_v.val[i], bias_v.val[i]);
      input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
      input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
      input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
      input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
    }
    vst1q_f32_x4(array + d, input_v);
  }
  for (; d < n; ++d) {
    array[d] = (input[d] + bias[d] - mean) * gamma[d] * variance + beta[d];
  }
}
void layerNormNobeta_bias(int n, const float* input, float* array,
                          const float* gamma, const float* bias, float eps) {
  //     float32x4_t sum_v[4];
  // #pragma unroll
  //     for (int i = 0; i < 4; ++i) {
  //         sum_v[i] = vdupq_n_f32(0.0f);
  //     }

  float32x4_t square_sum_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum_v[i] = vdupq_n_f32(0.0f);
  }

  int d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t regs = vld1q_f32_x4(input + d);
    float32x4x4_t bias_v = vld1q_f32_x4(bias + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      regs.val[i] = vaddq_f32(regs.val[i], bias_v.val[i]);
      // sum_v[i] = vaddq_f32(sum_v[i], regs.val[i]);
      square_sum_v[i] =
          vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
    }
  }
  float32_t sum = 0.0f;
  float32_t square_sum = 0.0f;
  for (; d < n; ++d) {
    float val = input[d] + bias[d];
    sum += val;
    square_sum += val * val;
  }

  //     sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
  //     sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
  //     sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
  // #pragma unroll
  //     for (int i = 0; i < 4; ++i) {
  //         sum += sum_v[0][i];
  //     }

  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
  square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
  square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    square_sum += square_sum_v[0][i];
  }

  // float mean = sum / n;
  float variance = square_sum / n;
  variance = 1.0f / std::sqrt(variance + eps);
  // variance = 1.0f / std::sqrt(variance - mean * mean + eps);
  // float32x4_t mean_v = vdupq_n_f32(mean);
  float32x4_t variance_v = vdupq_n_f32(variance);

  // normalization
  d = 0;
  for (; d <= n - 16; d += 16) {
    float32x4x4_t input_v = vld1q_f32_x4(input + d);
    float32x4x4_t bias_v = vld1q_f32_x4(bias + d);
    float32x4x4_t gamma_v = vld1q_f32_x4(gamma + d);
    // float32x4x4_t beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      input_v.val[i] = vaddq_f32(input_v.val[i], bias_v.val[i]);
      // input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
      input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
      input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
      // input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
    }
    vst1q_f32_x4(array + d, input_v);
  }
  for (; d < n; ++d) {
    array[d] = (input[d] + bias[d]) * gamma[d] * variance;  // + beta[d];
    // array[d] = (input[d] + bias[d] - mean) * gamma[d] * variance; // +
    // beta[d];
  }
}
#endif

template <>
void LayerNormKernel<float>(float* data_out, const float* data_in,
                            const float* bias, const float* gamma,
                            const float* beta, int m, int n, float eps) {
  if (bias == nullptr) {
    parallel_for(m, [&](int i) {
      int offset = i * n;
      layerNorm(n, data_in + offset, data_out + offset, gamma, beta, eps);
      // vNorm(n, data_in + offset, data_out + offset, gamma, beta,
      //         bias + offset, eps, bias != nullptr);
    });
  } else {
    parallel_for(m, [&](int i) {
      int offset = i * n;
      layerNorm_bias(n, data_in + offset, data_out + offset, gamma, beta,
                     bias + offset, eps);
    });
  }
}
template <>
void LayerNormNoBetaKernel<float>(float* data_out, const float* data_in,
                                  const float* bias, const float* gamma, int m,
                                  int n, float eps) {
  if (bias == nullptr) {
    parallel_for(m, [&](int i) {
      int offset = i * n;
      layerNormNobeta(n, data_in + offset, data_out + offset, gamma, eps);
    });
  } else {
    parallel_for(m, [&](int i) {
      int offset = i * n;
      layerNormNobeta_bias(n, data_in + offset, data_out + offset, gamma,
                           bias + offset, eps);
    });
  }
}

}  // namespace cpu
}  // namespace allspark