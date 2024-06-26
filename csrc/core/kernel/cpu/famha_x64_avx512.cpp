/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    famha.cpp
 */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
#include <immintrin.h>

#include <limits>

#include "cpu_common.h"
#include "cpu_kernel.h"

namespace allspark {
namespace cpu {

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

}  // namespace cpu
}  // namespace allspark
#endif
