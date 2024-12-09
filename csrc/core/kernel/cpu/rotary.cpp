/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <string.h>

#include "cpu_avx.h"
#elif defined ALLSPARK_USE_NEON_
#include "cpu_neon.h"
#endif

namespace allspark {
namespace cpu {

#if defined(__x86_64__) || defined(_M_X64)
template <typename T>
void RotaryKernelLauncher(T* out, T* in, float* inv_freq, int* batch_offset,
                          int batch_size, int seq_len, int num_head,
                          int size_per_head, int* step_list, int stride,
                          int xlogn) {
  // OPT TBD
  int N = batch_size * seq_len * num_head;
  parallel_for(N, [&](int tid) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * size_per_head;
    T* data_out = out + batch * seq_len * stride + seq_pos * stride +
                  head * size_per_head;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    int pos = seq_pos + step_list[batch] + offset;
    float scale = 1.0;
    if (xlogn > 0 && pos > xlogn) {
      scale = logf(pos) / logf(xlogn);
    }
    if (inv_freq == nullptr) {
      // no inv_freq provided, copy input to output
      memcpy(data_out, data_in, size_per_head * sizeof(T));
    } else {
      const int hex_sign_mask = 0x80000000;
      const float sign_mask = *(float*)&hex_sign_mask;
      __m256 scale_v = _mm256_set1_ps(scale);
      __m256 pos_v = _mm256_set1_ps(pos);
      __m256 sign_mask_v = _mm256_set1_ps(sign_mask);
      int half_inner = size_per_head / 2;
      for (int i = 0; i < 2; i++) {
        int d = 0;
        for (; d <= half_inner - 8; d += 8) {
          __m256 inv_freq_v = _mm256_loadu_ps(inv_freq + d);

          __m256 data_in_v_1;
          __m256 data_in_v_2;
          __m256 output_v;
          if (i == 0) {
            data_in_v_1 = _mm256_loadu_ps(data_in + d);
            data_in_v_2 = _mm256_loadu_ps(data_in + d + half_inner);
          } else {
            data_in_v_1 = _mm256_loadu_ps(data_in + d + half_inner);
            data_in_v_2 = _mm256_loadu_ps(data_in + d);
          }
          __m256 rad_v = _mm256_mul_ps(inv_freq_v, pos_v);
          __m256 sin_v;
          __m256 cos_v;
          sincos256_ps(rad_v, &sin_v, &cos_v);
          __m256 v1_v = _mm256_mul_ps(data_in_v_1, cos_v);
          __m256 v2_v;
          if (i == 0) {
            v2_v = _mm256_mul_ps(data_in_v_2, sin_v);
            v2_v = _mm256_xor_ps(v2_v, sign_mask_v);
          } else {
            v2_v = _mm256_mul_ps(data_in_v_2, sin_v);
          }
          output_v = _mm256_add_ps(v1_v, v2_v);
          output_v = _mm256_mul_ps(output_v, scale_v);
          if (i == 0) {
            _mm256_storeu_ps(data_out + d, output_v);
          } else {
            _mm256_storeu_ps(data_out + d + half_inner, output_v);
          }
        }
        for (; d < half_inner; ++d) {
          float sin_ = std::sin(inv_freq[d % half_inner] * pos);
          float cos_ = std::cos(inv_freq[d % half_inner] * pos);
          float v1 = data_in[d + i * half_inner] * cos_;
          float v2;
          if (i == 0) {
            v2 = -data_in[d + half_inner] * sin_;
          } else {
            v2 = data_in[d] * sin_;
          }
          data_out[d + i * half_inner] = static_cast<T>(v1 + v2) * scale;
        }
      }
    }
  });
}
#elif defined(ALLSPARK_USE_NEON_)
template <typename T>
void RotaryKernelLauncher(T* out, T* in, float* inv_freq, int* batch_offset,
                          int batch_size, int seq_len, int num_head,
                          int size_per_head, int* step_list, int stride,
                          int xlogn) {
  int N = batch_size * seq_len * num_head;
  parallel_for(N, [&](int tid) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * size_per_head;
    T* data_out = out + batch * seq_len * stride + seq_pos * stride +
                  head * size_per_head;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    int pos = seq_pos + step_list[batch] + offset;
    float scale = 1.0;
    if (xlogn > 0 && pos > xlogn) {
      scale = logf(pos) / logf(xlogn);
    }
    if (inv_freq == nullptr) {
      // no inv_freq provided, copy input to output
      memcpy(data_out, data_in, size_per_head * sizeof(T));
    } else {
      // use inv_freq for calculation
      float32x4_t pos_v[4];
      float32x4_t scale_v = vdupq_n_f32(scale);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pos_v[i] = vdupq_n_f32(pos);
      }
      for (int i = 0; i < 2; i++) {
        int d = 0;
        int half_inner = size_per_head / 2;
        for (; d <= half_inner - 16; d += 16) {
          float32x4x4_t inv_freq_neon = vld1q_f32_x4(inv_freq + d);
          float32x4x4_t data_in_neon =
              vld1q_f32_x4(data_in + d + i * half_inner);
          float32x4x4_t data_in_neon_2;
          float32x4x4_t output_neon;
          if (i == 0) {
            data_in_neon_2 = vld1q_f32_x4(data_in + d + half_inner);
          } else {
            data_in_neon_2 = vld1q_f32_x4(data_in + d);
          }
          for (int j = 0; j < 4; j++) {
            float32x4_t rad_sin_neon =
                vmulq_f32(inv_freq_neon.val[j], pos_v[j]);
            float32x4_t pi_2_neon = vdupq_n_f32(M_PI / 2);
            float32x4_t rad_cos_neon = vsubq_f32(pi_2_neon, rad_sin_neon);
            float32x4_t sin_neon = vsinq_f32(rad_sin_neon);
            float32x4_t cos_neon = vsinq_f32(rad_cos_neon);
            float32x4_t v1_neon = vmulq_f32(data_in_neon.val[j], cos_neon);
            float32x4_t v2_neon;
            if (i == 0) {
              v2_neon = vnegq_f32(vmulq_f32(data_in_neon_2.val[j], sin_neon));
            } else {
              v2_neon = vmulq_f32(data_in_neon_2.val[j], sin_neon);
            }
            output_neon.val[j] = vaddq_f32(v1_neon, v2_neon);
            output_neon.val[j] = vmulq_f32(output_neon.val[j], scale_v);
          }
          vst1q_f32_x4(data_out + d + i * half_inner, output_neon);
        }
        for (; d < half_inner; ++d) {
          float sin_ = std::sin(inv_freq[d % half_inner] * pos);
          float cos_ = std::cos(inv_freq[d % half_inner] * pos);
          float v1 = data_in[d + i * half_inner] * cos_;
          float v2;
          if (i == 0) {
            v2 = -data_in[d + half_inner] * sin_;
          } else {
            v2 = data_in[d] * sin_;
          }
          data_out[d + i * half_inner] = static_cast<T>(v1 + v2) * scale;
        }
      }
    }
  });
}
#endif

template void RotaryKernelLauncher<float>(float* out, float* in,
                                          float* inv_freq, int* batch_offset,
                                          int batch_size, int seq_len,
                                          int num_head, int size_per_head,
                                          int* step_list, int stride,
                                          int xlogn);

template <typename T>
void RotaryEmbedding2D(T* out, T* in, float* inv_freq, int* batch_offset,
                       int batch, int seq_len, int num_head, int inner,
                       int step, int stride, int input_len) {
  int N = batch * seq_len * num_head;
  parallel_for(N, [&](int tid) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int half_inner = inner / 2;

      int pos_1 = seq_pos + step + offset;
      if (pos_1 > input_len - 2) {
        pos_1 = input_len - 2;
      }
      for (int i = 0; i < half_inner; i++) {
        float sin_ = std::sin(inv_freq[i % (half_inner / 2)] * pos_1);
        float cos_ = std::cos(inv_freq[i % (half_inner / 2)] * pos_1);
        float v1 = data_in[i] * cos_;
        float v2;
        if (i < half_inner / 2) {
          v2 = -data_in[i + half_inner / 2] * sin_;
        } else {
          v2 = data_in[i - half_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
      int pos_2 = seq_pos + step + offset - (input_len - 2);
      if (pos_2 < 0) {
        pos_2 = 0;
      }

      for (int i = half_inner; i < inner; i++) {
        float sin_ =
            std::sin(inv_freq[(i - half_inner) % (half_inner / 2)] * pos_2);
        float cos_ =
            std::cos(inv_freq[(i - half_inner) % (half_inner / 2)] * pos_2);
        float v1 = data_in[i] * cos_;
        float v2;
        if ((i - half_inner) < half_inner / 2) {
          v2 = -data_in[i + half_inner / 2] * sin_;
        } else {
          v2 = data_in[i - half_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
    }
  });
}
template void RotaryEmbedding2D<float>(float* out, float* in, float* inv_freq,
                                       int* batch_offset, int batch,
                                       int seq_len, int num_head, int inner,
                                       int step, int stride, int input_len);

template <typename T>
void RotaryEmbeddingHalfInner(T* out, T* in, float* inv_freq, int* batch_offset,
                              int batch_size, int seq_len, int num_head,
                              int inner, int* step_list, int stride) {
  int N = batch_size * seq_len * num_head;
  parallel_for(N, [&](int tid) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int half_inner = inner / 2;
      int pos = seq_pos + step_list[batch] + offset;
      for (int i = 0; i < half_inner / 2; i++) {
        float sin_ = std::sin(inv_freq[i] * (float)pos);
        float cos_ = std::cos(inv_freq[i] * (float)pos);
        float v1 = data_in[i * 2];
        float v2 = data_in[i * 2 + 1];
        data_out[i * 2] = static_cast<T>(v1 * cos_ - v2 * sin_);
        data_out[i * 2 + 1] = static_cast<T>(v2 * cos_ + v1 * sin_);
      }
      for (int i = half_inner; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    }
  });
}

template void RotaryEmbeddingHalfInner<float>(
    float* out, float* in, float* inv_freq, int* batch_offset, int batch_size,
    int seq_len, int num_head, int inner, int* step_list, int stride);
template <typename T>
void RotaryPctKernelLauncher(T* out, T* in, float* inv_freq, int* batch_offset,
                             int batch, int seq_len, int num_head, int inner,
                             int step, int stride, float pct) {
  int N = batch * seq_len * num_head;
  parallel_for(N, [&](int tid) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    int pos = seq_pos + step + offset;
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int real_inner = round(inner * pct);
      for (int i = 0; i < real_inner; i++) {
        float sin_ = std::sin(inv_freq[i % (real_inner / 2)] * pos);
        float cos_ = std::cos(inv_freq[i % (real_inner / 2)] * pos);
        float v1 = data_in[i] * cos_;
        float v2;
        if (i < real_inner / 2) {
          v2 = -data_in[i + real_inner / 2] * sin_;
        } else {
          v2 = data_in[i - real_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
      for (int i = real_inner; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    }
  });
}
template void RotaryPctKernelLauncher<float>(float* output, float* input,
                                             float* inv_freq, int* batch_offset,
                                             int batch, int seq_len, int head,
                                             int size_per_head, int step,
                                             int stride, float pct);
}  // namespace cpu
}  // namespace allspark