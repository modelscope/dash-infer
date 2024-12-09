/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_a8w8_perc_kernel.cu
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../cuda_kernel.h"
#include "allspark.pb.h"
#include "moe_a8w8_kernel.h"
namespace allspark {
namespace cuda {

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
__device__ __forceinline__ int64_t index(int64_t total_col, int64_t row,
                                         int64_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}

__global__ void get_reorder_qdata_kernel(
    int64_t N, int8_t* reorder_qdata, float* reorder_scale,
    int32_t* reorder_red_sum, int8_t* input, float* input_scale,
    int32_t* input_red_sum, int64_t* experts_idx, int64_t* experts_seq,
    int* total_tokens_post_pad, int padding_val, int topk, int hidden_size,
    int block_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int64_t batch_num = tid / hidden_size;
    if (batch_num >= total_tokens_post_pad[0]) {  // vaild data
      reorder_qdata[tid] = 0;
      reorder_scale[batch_num] = 0;
      reorder_red_sum[batch_num] = 0;
    } else {
      int64_t col = tid % hidden_size;
      int src_num = experts_seq[batch_num];
      if (src_num == padding_val) {  // for padding data
        reorder_qdata[tid] = 0;
        reorder_scale[batch_num] = 0;
        reorder_red_sum[batch_num] = 0;
      } else {
        src_num = src_num / topk;
        reorder_qdata[tid] = input[src_num * hidden_size + col];
        if (col == 0) {
          reorder_scale[batch_num] = input_scale[src_num];
          reorder_red_sum[batch_num] = input_red_sum[src_num];
        }
      }
    }
  }
}

void GetReorderQData(int8_t* reorder_qdata, float* reorder_scale,
                     int32_t* reorder_red_sum, int8_t* input_q,
                     float* input_scale, int32_t* input_red_sum,
                     int64_t* experts_idx, int64_t* experts_seq,
                     int* total_tokens_post_pad, int max_total_tokens,
                     int padding_val, int topk, int hidden_size, int block_size,
                     cudaStream_t stream) {
  const int64_t N = (int64_t)max_total_tokens * hidden_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  get_reorder_qdata_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, reorder_qdata, reorder_scale, reorder_red_sum, input_q, input_scale,
      input_red_sum, experts_idx, experts_seq, total_tokens_post_pad,
      padding_val, topk, hidden_size, block_size);
}

template <typename T>
__global__ void MOE_get_qweight_data_kernel(int N, int64_t* experts_idx,
                                            int* total_tokens_post_pad,
                                            T* scale, T* scale_in, T* zero,
                                            T* zero_in, int max_block,
                                            int layout_size, int block_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_tokens = max_block * block_size;
  if (total_tokens_post_pad != nullptr) {
    total_tokens = total_tokens_post_pad[0];
  }
  if (tid < N) {
    int64_t block_idx = tid / layout_size;
    int64_t col = tid % layout_size;
    int dst_num = 0;
    if (experts_idx == nullptr) {
      dst_num = tid;
    } else {
      if (block_idx * block_size >= total_tokens) {
        dst_num = 0;
      } else {
        dst_num = experts_idx[block_idx];
      }
    }
    scale[tid] = scale_in[dst_num * layout_size + col];
    zero[tid] = zero_in[dst_num * layout_size + col];
  }
}

template <typename T>
void GetReorderQWeightData(int64_t* experts_idx, int* total_tokens_post_pad,
                           T* scale, T* scale_in, T* zero, T* zero_in,
                           int max_block, int layout_size, int block_size,
                           cudaStream_t stream) {
  const int N = max_block * layout_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  MOE_get_qweight_data_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, experts_idx, total_tokens_post_pad, scale, scale_in, zero, zero_in,
      max_block, layout_size, block_size);
}

template <typename FT, typename QT>
__global__ void MOE_qweight_get_batch_array_kernel(
    int N, int64_t* experts_idx, int* total_tokens_post_pad, QT* data,
    void** data_array, FT* scale, void** scale_array, FT* zero,
    void** zero_array, int max_block, int layout_size_qdata,
    int layout_size_scale, int block_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_tokens = max_block * block_size;
  if (total_tokens_post_pad != nullptr) {
    total_tokens = total_tokens_post_pad[0];
  }
  if (tid < N) {
    int dst_num = 0;
    if (experts_idx == nullptr) {
      dst_num = tid;
    } else {
      if (tid * block_size >= total_tokens) {
        dst_num = 0;
      } else {
        dst_num = experts_idx[tid];
      }
    }
    data_array[tid] = data + dst_num * layout_size_qdata;
    scale_array[tid] = scale + dst_num * layout_size_scale;
    zero_array[tid] = zero + dst_num * layout_size_scale;
  }
}

template <typename FT, typename QT>
void MOEQWeightGetBatchArrayLauncher(
    int64_t* experts_idx, int* total_tokens_post_pad, QT* qdata,
    void** qdata_array, FT* scale, void** scale_array, FT* zero,
    void** zero_array, int max_block, int layout_size_qdata,
    int layout_size_scale, int block_size, cudaStream_t stream) {
  const int N = max_block;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  MOE_qweight_get_batch_array_kernel<<<block_num, THREAD_PER_BLOCK, 0,
                                       stream>>>(
      N, experts_idx, total_tokens_post_pad, qdata, qdata_array, scale,
      scale_array, zero, zero_array, max_block, layout_size_qdata,
      layout_size_scale, block_size);
}

//========================

template void GetReorderQWeightData<half>(int64_t* experts_idx,
                                          int* total_tokens_post_pad,
                                          half* scale, half* scale_in,
                                          half* zero, half* zero_in,
                                          int max_block, int layout_size,
                                          int block_size, cudaStream_t stream);

template void GetReorderQWeightData<hie::bfloat16>(
    int64_t* experts_idx, int* total_tokens_post_pad, hie::bfloat16* scale,
    hie::bfloat16* scale_in, hie::bfloat16* zero, hie::bfloat16* zero_in,
    int max_block, int layout_size, int block_size, cudaStream_t stream);

template void MOEQWeightGetBatchArrayLauncher<half, int8_t>(
    int64_t* experts_idx, int* total_tokens_post_pad, int8_t* qdata,
    void** qdata_array, half* scale, void** scale_array, half* zero,
    void** zero_array, int max_block, int layout_size_qdata,
    int layout_size_scale, int block_size, cudaStream_t stream);
template void MOEQWeightGetBatchArrayLauncher<hie::bfloat16, int8_t>(
    int64_t* experts_idx, int* total_tokens_post_pad, int8_t* qdata,
    void** qdata_array, hie::bfloat16* scale, void** scale_array,
    hie::bfloat16* zero, void** zero_array, int max_block,
    int layout_size_qdata, int layout_size_scale, int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark