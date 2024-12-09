/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mha.cu
 */

#include "attention/softmax.hpp"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "interface/allspark_check.h"
#include "reduce.cuh"
namespace allspark {
namespace cuda {

template <typename T>
__global__ static void get_batch_array_kernel(
    T* q, T* k, T* v, T* score, T* out, T** q_array, T** k_array, T** v_array,
    T** score_array, T** out_array, int N, int beam_size, int num_heads,
    int size_per_head, int step, int q_stride, int kv_stride, int score_stride,
    int out_stride) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int j = tid % num_heads;
    int i = tid / num_heads;
    q_array[tid] = q + i * q_stride + j * size_per_head;
    k_array[tid] = k + i / beam_size * kv_stride + j * size_per_head;
    v_array[tid] = v + i / beam_size * kv_stride + j * size_per_head;
    score_array[tid] = score + i * score_stride + j * step;
    out_array[tid] = out + i * out_stride + j * size_per_head;
  }
}

template <typename T>
void GetBatchArrayLauncher(T* q, T* k, T* v, T* score, T* out, T** q_array,
                           T** k_array, T** v_array, T** score_array,
                           T** out_array, int batch_size, int beam_size,
                           int num_heads, int size_per_head, int step,
                           int q_stride, int kv_stride, int score_stride,
                           int out_stride, cudaStream_t stream) {
  const int N = batch_size * beam_size * num_heads;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  get_batch_array_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      q, k, v, score, out, q_array, k_array, v_array, score_array, out_array, N,
      beam_size, num_heads, size_per_head, step, q_stride, kv_stride,
      score_stride, out_stride);
}

template void GetBatchArrayLauncher<float>(
    float* q, float* k, float* v, float* score, float* out, float** q_array,
    float** k_array, float** v_array, float** score_array, float** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head, int step,
    int q_stride, int kv_stride, int score_stride, int out_stride,
    cudaStream_t stream);
#ifdef ENABLE_FP16
template void GetBatchArrayLauncher<half>(
    half* q, half* k, half* v, half* score, half* out, half** q_array,
    half** k_array, half** v_array, half** score_array, half** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head, int step,
    int q_stride, int kv_stride, int score_stride, int out_stride,
    cudaStream_t stream);
#endif
template void GetBatchArrayLauncher<hie::bfloat16>(
    hie::bfloat16* q, hie::bfloat16* k, hie::bfloat16* v, hie::bfloat16* score,
    hie::bfloat16* out, hie::bfloat16** q_array, hie::bfloat16** k_array,
    hie::bfloat16** v_array, hie::bfloat16** score_array,
    hie::bfloat16** out_array, int batch_size, int beam_size, int num_heads,
    int size_per_head, int step, int q_stride, int kv_stride, int score_stride,
    int out_stride, cudaStream_t stream);

template <typename T>
__global__ static void multi_query_get_batch_array_kernel(
    T* q, T* k, T* v, T* score, T* out, T** q_array, T** k_array, T** v_array,
    T** score_array, T** out_array, int N, int beam_size, int num_heads,
    int size_per_head, int group_num, int step, int q_stride, int kv_stride,
    int score_stride, int out_stride) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int j = tid % num_heads;
    int i = tid / num_heads;
    q_array[tid] = q + i * q_stride + j * size_per_head;
    int group_now = j / (num_heads / group_num);
    k_array[tid] = k + i / beam_size * kv_stride + group_now * size_per_head;
    v_array[tid] = v + i / beam_size * kv_stride + group_now * size_per_head;
    score_array[tid] = score + i * score_stride + j * step;
    out_array[tid] = out + i * out_stride + j * size_per_head;
  }
}

template <typename T>
void MultiQueryGetBatchArrayLauncher(
    T* q, T* k, T* v, T* score, T* out, T** q_array, T** k_array, T** v_array,
    T** score_array, T** out_array, int batch_size, int beam_size,
    int num_heads, int size_per_head, int group_num, int step, int q_stride,
    int kv_stride, int score_stride, int out_stride, cudaStream_t stream) {
  const int N = batch_size * beam_size * num_heads;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  multi_query_get_batch_array_kernel<<<block_num, THREAD_PER_BLOCK, 0,
                                       stream>>>(
      q, k, v, score, out, q_array, k_array, v_array, score_array, out_array, N,
      beam_size, num_heads, size_per_head, group_num, step, q_stride, kv_stride,
      score_stride, out_stride);
}

template void MultiQueryGetBatchArrayLauncher<float>(
    float* q, float* k, float* v, float* score, float* out, float** q_array,
    float** k_array, float** v_array, float** score_array, float** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head,
    int group_num, int step, int q_stride, int kv_stride, int score_stride,
    int out_stride, cudaStream_t stream);
#ifdef ENABLE_FP16
template void MultiQueryGetBatchArrayLauncher<half>(
    half* q, half* k, half* v, half* score, half* out, half** q_array,
    half** k_array, half** v_array, half** score_array, half** out_array,
    int batch_size, int beam_size, int num_heads, int size_per_head,
    int group_num, int step, int q_stride, int kv_stride, int score_stride,
    int out_stride, cudaStream_t stream);
#endif
template void MultiQueryGetBatchArrayLauncher<hie::bfloat16>(
    hie::bfloat16* q, hie::bfloat16* k, hie::bfloat16* v, hie::bfloat16* score,
    hie::bfloat16* out, hie::bfloat16** q_array, hie::bfloat16** k_array,
    hie::bfloat16** v_array, hie::bfloat16** score_array,
    hie::bfloat16** out_array, int batch_size, int beam_size, int num_heads,
    int size_per_head, int group_num, int step, int q_stride, int kv_stride,
    int score_stride, int out_stride, cudaStream_t stream);
template <typename T>
__global__ static void softmax_kernel(T* qk_buf_, const int step) {
  float tmp = -1e20f;
  float qk = 0.0f;
  int qk_offset = blockIdx.x * step + threadIdx.x;
  __shared__ float s_sum, s_max;
  if (threadIdx.x < step) {
    qk = static_cast<float>(qk_buf_[qk_offset]);
    tmp = qk;
  }
  float max_val = tmp;
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  if (threadIdx.x < step) {
    qk = expf(tmp - s_max);
  }
  float sum_val = qk;
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    qk_buf_[qk_offset] = qk / s_sum;
  }
}
// add mask + softmax
template <typename T>
__global__ void softmax_with_mask_kernel(T* qk_buf_, const float* mask,
                                         const int beam_size, const int step) {
  float tmp = -1e20f;
  float qk = 0.0f;
  int qk_offset =
      ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * step +
      threadIdx.x;
  int mask_offset =
      (blockIdx.z / beam_size * step + blockIdx.y) * step + threadIdx.x;
  __shared__ float s_sum, s_max;
  if (threadIdx.x < step) {
    qk = static_cast<float>(qk_buf_[qk_offset]);
    float mask_val = (1 - __ldg(&mask[mask_offset])) * -100000.f;
    tmp = qk + mask_val;
  }
  float max_val = tmp;
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  if (threadIdx.x < step) {
    qk = expf(tmp - s_max);
  }
  float sum_val = qk;
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    qk_buf_[qk_offset] = (T)(qk / s_sum);
  }
}
template <typename T, int BLOCK = 1024, int UNROLL>
__global__ void softmax_kernel_UNROLL(T* qk_buf_, const float* mask,
                                      const int beam_size, const int step) {
  float tmp[UNROLL];
  float qk[UNROLL];
  __shared__ float s_sum, s_max;
  float max_val = -1e20f;
  float sum_val = 0.0f;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset = blockIdx.x * step + tid;
      qk[i] = static_cast<float>(qk_buf_[qk_offset]);
      tmp[i] = qk[i];
      max_val = max(max_val, tmp[i]);
    }
  }
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      qk[i] = expf(tmp[i] - s_max);
      sum_val += qk[i];
    }
  }
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset = blockIdx.x * step + tid;
      qk_buf_[qk_offset] = (T)(qk[i] / s_sum);
    }
  }
}
template <typename T, int BLOCK = 1024, int UNROLL>
__global__ void softmax_with_mask_kernel_UNROLL(T* qk_buf_, const float* mask,
                                                const int beam_size,
                                                const int step) {
  float tmp[UNROLL];
  float qk[UNROLL];
  __shared__ float s_sum, s_max;
  float max_val = -1e20f;
  float sum_val = 0.0f;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset =
          ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
              step +
          tid;
      int mask_offset =
          (blockIdx.z / beam_size * step + blockIdx.y) * step + tid;
      qk[i] = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = (1 - __ldg(&mask[mask_offset])) * -100000.f;
      tmp[i] = qk[i] + mask_val;
      max_val = max(max_val, tmp[i]);
    }
  }
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      qk[i] = expf(tmp[i] - s_max);
      sum_val += qk[i];
    }
  }
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset =
          ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
              step +
          tid;
      qk_buf_[qk_offset] = (T)(qk[i] / s_sum);
    }
  }
}

namespace opt {
/**
 * @brief One block compute one inner line.
 *
 * data_in : [BatchSize, Seq, NumHead, Step]
 * mask    : [BatchSize, Seq, Step]
 * data_out: [BatchSize, Seq, NumHead, Step]
 *
 * block.x = NumHead
 * block.y = Seq
 * block.z = BatchSize
 *
 */
template <typename T, int BLOCK, int UNROLL>
__global__ __launch_bounds__(BLOCK) void softmax_one_pass(
    const T* data_in, const float* mask, T* data_out, const int batch_size,
    const int seq_len, const int num_heads, const int step) {
  const int tid = threadIdx.x;
  const uint64_t offset = blockIdx.x * step + blockIdx.y * num_heads * step +
                          blockIdx.z * uint64_t(seq_len) * num_heads * step;
  const uint64_t mask_offset =
      blockIdx.z * uint64_t(seq_len) * step + blockIdx.y * step;

  const T* data_in_ptr = data_in + offset;
  const float* mask_ptr = mask + mask_offset;
  T* data_out_ptr = data_out + offset;

  float ld_reg[UNROLL] = {float(0)};
  float mask_reg[UNROLL] = {float(0)};
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int idx = tid + i * BLOCK;
    ld_reg[i] = idx < step ? float(data_in_ptr[idx]) : float(-INFINITY);
    mask_reg[i] = idx < step ? mask_ptr[idx] : float(0);
  }

  // Reduce Max
  float fmax = float(-INFINITY);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    ld_reg[i] = ld_reg[i] + ((1 - mask_reg[i]) * (-100000.0f));
    fmax = max(fmax, ld_reg[i]);
  }
  blockReduce<float, ReduceOp::kMAX>(&fmax);
  __shared__ float s_max;
  if (tid == 0) {
    s_max = fmax;
  }
  __syncthreads();
  fmax = s_max;

  // Exp-Sum
  float exp_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int idx = tid + i * BLOCK;
    if (idx < step) {
      ld_reg[i] = __expf(ld_reg[i] - fmax);
      exp_sum += ld_reg[i];
    }
  }
  blockReduce<float, ReduceOp::kSUM>(&exp_sum);
  __shared__ float s_sum;
  if (tid == 0) {
    s_sum = 1.0f / (exp_sum + 1e-12f);
  }
  __syncthreads();
  exp_sum = s_sum;

  float st_reg[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    st_reg[i] = ld_reg[i] * exp_sum;
  }
// Store
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int idx = tid + i * BLOCK;
    if (idx < step) {
      data_out_ptr[idx] = T(st_reg[i]);
    }
  }
}
}  // namespace opt

template <typename T>
void SoftmaxKernelLauncher(T* qk_buf, const float* mask, int batch_size,
                           int beam_size, int num_heads, int seq_len, int step,
                           cudaStream_t stream) {
  dim3 block, grid;
  if (mask != nullptr) {
    grid.x = num_heads;
    grid.y = seq_len;
    grid.z = batch_size;
    if (step <= 1024) {
      const int UNROLL = 8;
      const int BLOCK = 1024 / UNROLL;
      dim3 g_dim(num_heads, seq_len, batch_size);
      opt::softmax_one_pass<T, BLOCK, UNROLL><<<g_dim, BLOCK, 0, stream>>>(
          qk_buf, mask, qk_buf, batch_size, seq_len, num_heads, step);
    } else if (step <= 2048) {
      const int UNROLL = 8;
      const int BLOCK = 2048 / UNROLL;
      dim3 g_dim(num_heads, seq_len, batch_size);
      opt::softmax_one_pass<T, BLOCK, UNROLL><<<g_dim, BLOCK, 0, stream>>>(
          qk_buf, mask, qk_buf, batch_size, seq_len, num_heads, step);
    } else if (step <= 4096) {
      const int UNROLL = 8;
      const int BLOCK = 4096 / UNROLL;
      dim3 g_dim(num_heads, seq_len, batch_size);
      opt::softmax_one_pass<T, BLOCK, UNROLL><<<g_dim, BLOCK, 0, stream>>>(
          qk_buf, mask, qk_buf, batch_size, seq_len, num_heads, step);
    } else if (step <= 8192) {
      const int UNROLL = 8;
      const int BLOCK = 8192 / UNROLL;
      dim3 g_dim(num_heads, seq_len, batch_size);
      opt::softmax_one_pass<T, BLOCK, UNROLL><<<g_dim, BLOCK, 0, stream>>>(
          qk_buf, mask, qk_buf, batch_size, seq_len, num_heads, step);
    } else {
      softmax_4d_fallback(stream, attention::toDataType<T>::cuda_type, qk_buf,
                          qk_buf, mask, 1.f, batch_size, seq_len, num_heads,
                          step, batch_size / beam_size, seq_len, 1, step, false,
                          true, false, false, 0, 0);
      // throw AsException("max_length > 8192 not supported yet in
      // SoftmaxKernelLauncher()");
    }
  } else {
    grid.x = batch_size * num_heads * seq_len;
    if (step <= 1024) {
      block.x = (step + 31) / 32 * 32;
      softmax_kernel<<<grid, block, 0, stream>>>(qk_buf, step);
    } else if (step <= 2048) {
      const int unroll = 2;
      block.x = 1024;
      softmax_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step);
    } else if (step <= 3072) {
      const int unroll = 3;
      block.x = 1024;
      softmax_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step);
    } else if (step <= 4096) {
      const int unroll = 4;
      block.x = 1024;
      softmax_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step);
    } else if (step <= 8192) {
      const int unroll = 8;
      block.x = 1024;
      softmax_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step);
    } else {
      softmax_4d_fallback(stream, attention::toDataType<T>::cuda_type, qk_buf,
                          qk_buf, nullptr, 1.f, batch_size, seq_len, num_heads,
                          step, 1, 1, 1, 1, false, true, false, false, 0, 0);
      // throw AsException("max_length > 8192 not supported yet in
      // SoftmaxKernelLauncher()");
    }
  }
}

template void SoftmaxKernelLauncher<float>(float* qk_buf, const float* mask,
                                           int batch_size, int beam_size,
                                           int num_heads, int seq_len, int step,
                                           cudaStream_t stream);
#ifdef ENABLE_FP16
template void SoftmaxKernelLauncher<half>(half* qk_buf, const float* mask,
                                          int batch_size, int beam_size,
                                          int num_heads, int seq_len, int step,
                                          cudaStream_t stream);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
template void SoftmaxKernelLauncher<hie::bfloat16>(
    hie::bfloat16* qk_buf, const float* mask, int batch_size, int beam_size,
    int num_heads, int seq_len, int step, cudaStream_t stream);
#endif  // ENABLE_BF16

template <typename T>
__device__ __forceinline__ float logn_basic_load_and_scale(
    const T* qkbuf, const float* maskp, int32_t bidx, int32_t xidx,
    int32_t nidx, int32_t sidx, int32_t batch, int32_t xseql, int32_t nhead,
    int32_t cstep, float scale) {
  if (bidx < batch && xidx < xseql && nidx < nhead && sidx < cstep) {
    // valid indexing.
    int64_t qkbuf_offset = bidx * int64_t(xseql) * nhead * int64_t(cstep) +
                           int64_t(xidx) * nhead * int64_t(cstep) +
                           nidx * int64_t(cstep) + int64_t(sidx);
    int64_t maskp_offset = bidx * int64_t(xseql) * int64_t(cstep) +
                           int64_t(xidx) * int64_t(cstep) + int64_t(sidx);
    float qkbuf_load = static_cast<float>(qkbuf[qkbuf_offset]);
    float maskp_load = maskp ? (1 - maskp[maskp_offset]) * (-1e15) : 0.f;
    float score = scale * qkbuf_load + maskp_load;
    return score;
  } else {
    return -INFINITY;
  }
}

// naive impl... optimize later.
template <typename T>
__global__ void logn_softmax_inplace_no_unroll_kernel(
    T* qkbuf,            //  [batch, xseql, nhead, cstep]
    const float* maskp,  //  [batch, xseql, cstep]
    int32_t batch, int32_t nhead, int32_t xseql, int32_t cstep, int32_t xlogn) {
  __shared__ float shr[32];
  __shared__ float block_max, block_sum, block_div;
  // block.z batch, block.y nhead, block.x xseql
  int32_t bidx = blockIdx.z;
  int32_t nidx = blockIdx.y;
  int32_t xidx = blockIdx.x;
  float scale = xidx > xlogn ? logf(xidx) / logf(xlogn) : 1.f;

  // find block max.
  float tidx_max = -INFINITY;
  for (int64_t sidx = threadIdx.x; sidx < cstep; sidx += blockDim.x) {
    // too lazy to optimize this.
    float score =
        logn_basic_load_and_scale(qkbuf, maskp, bidx, xidx, nidx, sidx, batch,
                                  xseql, nhead, cstep, scale);
    tidx_max = tidx_max > score ? tidx_max : score;
  }
  // blockReduce<float, ReduceOp::kMAX>(&tidx_max);
  warpReduce<float, ReduceOp::kMAX>(&tidx_max);
  if (threadIdx.x % 32 == 0) shr[threadIdx.x / 32] = tidx_max;
  __syncthreads();
  if (threadIdx.x == 0) {
    block_max = tidx_max;
    for (int32_t warp = 1; warp < blockDim.x / 32; warp++) {
      block_max = block_max > shr[warp] ? block_max : shr[warp];
    }
  }
  __syncthreads();

  // calculate exp sum.
  float tidx_sum = 0.f;
  for (int64_t sidx = threadIdx.x; sidx < cstep; sidx += blockDim.x) {
    float score =
        logn_basic_load_and_scale(qkbuf, maskp, bidx, xidx, nidx, sidx, batch,
                                  xseql, nhead, cstep, scale);
    tidx_sum += expf(score - block_max);
  }
  // blockReduce<float, ReduceOp::kSUM>(&tidx_sum);
  warpReduce<float, ReduceOp::kSUM>(&tidx_sum);
  if (threadIdx.x % 32 == 0) shr[threadIdx.x / 32] = tidx_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    block_sum = 0.f;
    for (int32_t warp = 0; warp < blockDim.x / 32; warp++) {
      block_sum += shr[warp];
    }
    block_div = 1.f / (block_sum + 1e-12f);
  }
  __syncthreads();

  // store softmax
  for (int64_t sidx = threadIdx.x; sidx < cstep; sidx += blockDim.x) {
    float score =
        logn_basic_load_and_scale(qkbuf, maskp, bidx, xidx, nidx, sidx, batch,
                                  xseql, nhead, cstep, scale);
    float scexp = expf(score - block_max) * block_div;
    // write back
    int64_t index = bidx * int64_t(xseql) * nhead * int64_t(cstep) +
                    int64_t(xidx) * nhead * int64_t(cstep) +
                    nidx * int64_t(cstep) + int64_t(sidx);
    if (bidx < batch && xidx < xseql && nidx < nhead && sidx < cstep) {
      qkbuf[index] = static_cast<T>(scexp);
    }
  }
}

template <typename T>
void LognSoftmaxKernelLauncher(T* qkbuf /* score of qxk */,
                               const float* maskp /* padding mask */,
                               int32_t batch, int32_t nhead, int32_t xseql,
                               int32_t cstep, int32_t xlogn,
                               cudaStream_t stream) {
  // if logn enable, use this launcher.
  // first we have to check if it is true that input sequence length longer than
  // module basic training embedding length. in logn logic, only sequence length
  // longer than module basic training embedding length have to calculate
  // log(index, xlogn), otherwise just multiply 1.f instead and keep softmax.
  if (xseql > xlogn && xlogn != 0) {
    // into logn logic.
    // otherwise back to normal logic is better.
    // no optimize kernel selection logic, cause xlogn usually longer than 8k,
    // and whatever the kernel logic is, it will be slow.
    dim3 kgrid, kblock;
    kgrid.x = xseql;
    kgrid.y = nhead;
    kgrid.z = batch;
    kblock.x = 1024;
    // printf("LognSoftmaxKernelLauncher<<<[%d, %d, %d], [%d, %d, %d]>>> (",
    //     kgrid.x, kgrid.y, kgrid.z, kblock.x, kblock.y, kblock.z);
    // printf("batch=%d, nhead=%d, xseql=%d, cstep=%d, xlogn=%d)\n",
    //     batch, nhead, xseql, cstep, xlogn);
    logn_softmax_inplace_no_unroll_kernel<T><<<kgrid, kblock, 0, stream>>>(
        qkbuf, maskp, batch, nhead, xseql, cstep, xlogn);
  } else {
    SoftmaxKernelLauncher<T>(qkbuf, maskp, batch, 1, nhead, xseql, cstep,
                             stream);
  }
}

template void LognSoftmaxKernelLauncher<float>(
    float* qkbuf /* score of qxk */, const float* maskp /* padding mask */,
    int32_t batch, int32_t nhead, int32_t xseql, int32_t cstep, int32_t xlogn,
    cudaStream_t stream);
#ifdef ENABLE_FP16
template void LognSoftmaxKernelLauncher<half>(
    half* qkbuf /* score of qxk */, const float* maskp /* padding mask */,
    int32_t batch, int32_t nhead, int32_t xseql, int32_t cstep, int32_t xlogn,
    cudaStream_t stream);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
template void LognSoftmaxKernelLauncher<hie::bfloat16>(
    hie::bfloat16* qkbuf /* score of qxk */,
    const float* maskp /* padding mask */, int32_t batch, int32_t nhead,
    int32_t xseql, int32_t cstep, int32_t xlogn, cudaStream_t stream);
#endif  // ENABLE_BF16

template <typename T>
__global__ void decoder_softmax_with_mask_kernel(T* qk_buf_, const float* mask,
                                                 const int beam_size,
                                                 const int step,
                                                 const int input_len) {
  float tmp = -1e20f;
  float qk = 0.0f;
  int qk_offset =
      ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * step +
      threadIdx.x;
  int mask_offset =
      (blockIdx.z / beam_size * input_len + input_len - 1) * input_len +
      threadIdx.x;
  __shared__ float s_sum, s_max;
  if (threadIdx.x < step) {
    qk = static_cast<float>(qk_buf_[qk_offset]);
    float mask_val = 0;
    if (mask && threadIdx.x < input_len) {  // if valid and not nullptr.
      mask_val = (1 - __ldg(&mask[mask_offset])) * -100000.f;
    }
    tmp = qk + mask_val;
  }
  float max_val = tmp;
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  if (threadIdx.x < step) {
    qk = expf(tmp - s_max);
  }
  float sum_val = qk;
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    qk_buf_[qk_offset] = (T)(qk / s_sum);
  }
}
template <typename T, int BLOCK = 1024, int UNROLL>
__global__ void decoder_softmax_with_mask_kernel_UNROLL(T* qk_buf_,
                                                        const float* mask,
                                                        const int beam_size,
                                                        const int step,
                                                        const int input_len) {
  float tmp[UNROLL];
  float qk[UNROLL];
  __shared__ float s_sum, s_max;
  float max_val = -1e20f;
  float sum_val = 0.0f;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset =
          ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
              step +
          tid;
      int mask_offset =
          (blockIdx.z / beam_size * input_len + input_len - 1) * input_len +
          tid;
      qk[i] = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = 0;
      if (mask && tid < input_len) {  // if valid and not nullptr.
        mask_val = (1 - __ldg(&mask[mask_offset])) * -100000.f;
      }
      tmp[i] = qk[i] + mask_val;
      max_val = max(max_val, tmp[i]);
    }
  }
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      qk[i] = expf(tmp[i] - s_max);
      sum_val += qk[i];
    }
  }
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int tid = threadIdx.x + i * BLOCK;
    if (tid < step) {
      int qk_offset =
          ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
              step +
          tid;
      qk_buf_[qk_offset] = (T)(qk[i] / s_sum);
    }
  }
}
template <typename T>
void DecoderSoftmaxKernelLauncher(T* qk_buf, const float* mask, int batch_size,
                                  int beam_size, int num_heads, int seq_len,
                                  int step, int input_len,
                                  cudaStream_t stream) {
  dim3 block, grid;
  if (seq_len != 1) {
    throw AsException("Decoder Softmax not support sequence length != 1");
  }
  if (mask != nullptr) {
    grid.x = num_heads;
    grid.y = seq_len;  // seq_len=1
    grid.z = batch_size;
    if (step <= 1024) {
      block.x = (step + 31) / 32 * 32;
      decoder_softmax_with_mask_kernel<<<grid, block, 0, stream>>>(
          qk_buf, mask, beam_size, step, input_len);
    } else if (step <= 2048) {
      const int unroll = 2;
      block.x = 1024;
      decoder_softmax_with_mask_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step,
                                       input_len);
    } else if (step <= 4096) {
      const int unroll = 4;
      block.x = 1024;
      decoder_softmax_with_mask_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step,
                                       input_len);
    } else if (step <= 8192) {
      const int unroll = 8;
      block.x = 1024;
      decoder_softmax_with_mask_kernel_UNROLL<T, 1024, unroll>
          <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step,
                                       input_len);
    } else {
      softmax_4d_fallback(stream, attention::toDataType<T>::cuda_type, qk_buf,
                          qk_buf, mask, 1.f, batch_size, 1, num_heads, step,
                          batch_size / beam_size, input_len, 1, input_len, true,
                          true, false, false, 0, 0);
      // throw AsException("max_length > 8192 not supported yet in
      // DecoderSoftmaxKernelLauncher()");
    }
  } else {
    softmax_4d_fallback(stream, attention::toDataType<T>::cuda_type, qk_buf,
                        qk_buf, nullptr, 1.f, batch_size, 1, num_heads, step, 1,
                        1, 1, 1, true, true, false, false, 0, 0);
    // throw AsException("mask should not be nullptr in
    // DecoderSoftmaxKernelLauncher()");
  }
}
template void DecoderSoftmaxKernelLauncher<float>(
    float* qk_buf, const float* mask, int batch_size, int beam_size,
    int num_heads, int seq_len, int step, int input_len, cudaStream_t stream);
#ifdef ENABLE_FP16
template void DecoderSoftmaxKernelLauncher<half>(
    half* qk_buf, const float* mask, int batch_size, int beam_size,
    int num_heads, int seq_len, int step, int input_len, cudaStream_t stream);
#endif
template void DecoderSoftmaxKernelLauncher<hie::bfloat16>(
    hie::bfloat16* qk_buf, const float* mask, int batch_size, int beam_size,
    int num_heads, int seq_len, int step, int input_len, cudaStream_t stream);
template <typename T>
__global__ static void add_mask_kernel(T* score, const float* mask, int stride,
                                       int seq_len, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int seq_idx = tid % seq_len;
    int batch_idx = tid / stride;
    int mask_offset = batch_idx * seq_len * seq_len + seq_idx;
    float mask_val = (1 - __ldg(&mask[mask_offset])) * -100000.f;
    score[tid] = score[tid] + (T)mask_val;
  }
}
// score [batch * beam_size, num_heads, 1, enc_seq_len]
// mask [batch, enc_seq_len, enc_seq_len]
template <typename T>
void AddMaskLauncher(T* score, const float* mask, int batch_size, int beam_size,
                     int num_heads, int enc_seq_len, cudaStream_t stream) {
  const int N = batch_size * beam_size * num_heads * enc_seq_len;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  add_mask_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      score, mask, beam_size * num_heads * enc_seq_len, enc_seq_len, N);
}

template void AddMaskLauncher<float>(float* score, const float* mask,
                                     int batch_size, int beam_size,
                                     int num_heads, int enc_seq_len,
                                     cudaStream_t stream);
#ifdef ENABLE_FP16
template void AddMaskLauncher<half>(half* score, const float* mask,
                                    int batch_size, int beam_size,
                                    int num_heads, int enc_seq_len,
                                    cudaStream_t stream);
#endif
template void AddMaskLauncher<hie::bfloat16>(hie::bfloat16* score,
                                             const float* mask, int batch_size,
                                             int beam_size, int num_heads,
                                             int enc_seq_len,
                                             cudaStream_t stream);
template <>
void BatchGemmWraper<float>(void** matrix_C, void** matrix_A, void** matrix_B,
                            int m, int n, int k, bool transA, bool transB,
                            float alpha, float beta, int lda, int ldb, int ldc,
                            int batch, cublasHandle_t handle) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasGemmBatchedEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                      CUDA_R_32F, ldb, matrix_A, CUDA_R_32F, lda, &beta,
                      matrix_C, CUDA_R_32F, ldc, batch, CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
}
#ifdef ENABLE_FP16
template <>
void BatchGemmWraper<half>(void** matrix_C, void** matrix_A, void** matrix_B,
                           int m, int n, int k, bool transA, bool transB,
                           float alpha, float beta, int lda, int ldb, int ldc,
                           int batch, cublasHandle_t handle) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasGemmBatchedEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                      CUDA_R_16F, ldb, matrix_A, CUDA_R_16F, lda, &beta,
                      matrix_C, CUDA_R_16F, ldc, batch, CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
#endif
template <>
void BatchGemmWraper<hie::bfloat16>(void** matrix_C, void** matrix_A,
                                    void** matrix_B, int m, int n, int k,
                                    bool transA, bool transB, float alpha,
                                    float beta, int lda, int ldb, int ldc,
                                    int batch, cublasHandle_t handle) {
  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasGemmBatchedEx(handle, transB_, transA_, n, m, k, &alpha, matrix_B,
                      CUDA_R_16BF, ldb, matrix_A, CUDA_R_16BF, lda, &beta,
                      matrix_C, CUDA_R_16BF, ldc, batch, CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
template <typename T>
__global__ void update_kv_kernel(T* dst_k, T* dst_v, const T* src_k,
                                 const T* src_v, int step, int max_length,
                                 int hidden_size, int seq_len, int stride,
                                 int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    int idx1 = tid / (seq_len * hidden_size);                // batch
    int idx2 = tid % (seq_len * hidden_size) / hidden_size;  // seqlen
    int idx3 = tid % hidden_size;                            // hidden
    int src_idx = idx1 * seq_len * stride + idx2 * stride + idx3;
    int dst_idx =
        (idx1 * max_length + step) * hidden_size + idx2 * hidden_size + idx3;
    dst_k[dst_idx] = src_k[src_idx];
    dst_v[dst_idx] = src_v[src_idx];
  }
}

template <typename T>
void UpdateKVLauncher(T* k, T* v, const T* step_k, const T* step_v,
                      int batch_size, int step, int max_length, int hidden_size,
                      int seq_len, int stride, cudaStream_t stream) {
  int N = batch_size * seq_len * hidden_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  update_kv_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      k, v, step_k, step_v, step, max_length, hidden_size, seq_len, stride, N);
}

template void UpdateKVLauncher<float>(float* k, float* v, const float* step_k,
                                      const float* step_v, int batch_size,
                                      int step, int max_length, int hidden_size,
                                      int seq_len, int stride,
                                      cudaStream_t stream);
#ifdef ENABLE_FP16
template void UpdateKVLauncher<half>(half* k, half* v, const half* step_k,
                                     const half* step_v, int batch_size,
                                     int step, int max_length, int hidden_size,
                                     int seq_len, int stride,
                                     cudaStream_t stream);
#endif
template void UpdateKVLauncher<hie::bfloat16>(
    hie::bfloat16* k, hie::bfloat16* v, const hie::bfloat16* step_k,
    const hie::bfloat16* step_v, int batch_size, int step, int max_length,
    int hidden_size, int seq_len, int stride, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
