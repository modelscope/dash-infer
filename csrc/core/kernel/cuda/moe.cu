/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe.cu
 */

#include <math.h>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
__device__ __forceinline__ int64_t index(int64_t total_col, int64_t row,
                                         int64_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
__global__ void run_kernel(int64_t* expert_ids, int64_t* sorted_token_ids,
                           int64_t* indice_source, int* topk_ids, int batch,
                           int num_experts, int top_k, int block_size,
                           int* total_tokens_post_pad) {
  int64_t numel = (int64_t)batch * top_k;
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;
  extern __shared__ int64_t shared_mem[];
  int64_t* tokens_cnts =
      shared_mem;  // 2d tensor with shape (num_experts + 1, num_experts)
  int64_t* cumsum =
      shared_mem + (num_experts + 1) *
                       num_experts;  // 1d tensor with shape (num_experts + 1)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
  for (int i = 1; i <= blockDim.x; ++i) {
    tokens_cnts[index(num_experts, i, threadIdx.x)] +=
        tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                          block_size) *
                      block_size;
    }
    *total_tokens_post_pad = (int)cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
       i += block_size) {
    expert_ids[i / block_size] = threadIdx.x;
  }
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += 1) {
    sorted_token_ids[i] = numel;
  }
  __syncthreads();
  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int64_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int64_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    indice_source[i] = rank_post_pad;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}
void ReorderAndPaddingMOE(int64_t* experts_idx, int64_t* experts_seq,
                          int64_t* indice_source, int* input, int batch,
                          int num_experts, int top_k, int block_size,
                          int* total_token_post_pad, cudaStream_t stream) {
  const int64_t shared_mem =
      ((num_experts + 1) * num_experts + (num_experts + 1)) * sizeof(int64_t);
  run_kernel<<<1, num_experts, shared_mem, stream>>>(
      experts_idx, experts_seq, indice_source, input, batch, num_experts, top_k,
      block_size, total_token_post_pad);
}
template <typename T>
__global__ void get_reorder_data_kernel(int64_t N, T* reorder_data, T* input,
                                        int64_t* experts_idx,
                                        int64_t* experts_seq,
                                        int* total_tokens_post_pad,
                                        int padding_val, int topk,
                                        int hidden_size, int block_size) {
  int64_t tid = (int64_t)threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int64_t batch_num = tid / hidden_size;
    if (batch_num >= total_tokens_post_pad[0]) {  // vaild data
      reorder_data[tid] = (T)0.0;
    } else {
      int64_t col = tid % hidden_size;
      int64_t src_num = experts_seq[batch_num];
      if (src_num == padding_val) {  // for padding data
        reorder_data[tid] = (T)0.0;
      } else {
        src_num = src_num / topk;
        reorder_data[tid] = input[src_num * hidden_size + col];
      }
    }
  }
}
template <typename T>
void GetReorderData(T* reorder_data, T* input, int64_t* experts_idx,
                    int64_t* experts_seq, int* total_tokens_post_pad,
                    int max_total_tokens, int padding_val, int topk,
                    int hidden_size, int block_size, cudaStream_t stream) {
  const int64_t N = (int64_t)max_total_tokens * hidden_size;
  const int64_t block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  get_reorder_data_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, reorder_data, input, experts_idx, experts_seq, total_tokens_post_pad,
      padding_val, topk, hidden_size, block_size);
}
template void GetReorderData<float>(float* reorder_data, float* input,
                                    int64_t* experts_idx, int64_t* experts_seq,
                                    int* total_tokens_post_pad,
                                    int max_total_tokens, int expert_num,
                                    int topk, int hidden_size, int block_size,
                                    cudaStream_t stream);

#ifdef ENABLE_FP16
template void GetReorderData<half>(half* reorder_data, half* input,
                                   int64_t* experts_idx, int64_t* experts_seq,
                                   int* total_tokens_post_pad,
                                   int max_total_tokens, int expert_num,
                                   int topk, int hidden_size, int block_size,
                                   cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void GetReorderData<hie::bfloat16>(
    hie::bfloat16* reorder_data, hie::bfloat16* input, int64_t* experts_idx,
    int64_t* experts_seq, int* total_tokens_post_pad, int max_total_tokens,
    int expert_num, int topk, int hidden_size, int block_size,
    cudaStream_t stream);
#endif
template <typename T>
__global__ void MOE_get_batch_array_kernel(int N, int64_t* experts_idx,
                                           int* total_tokens_post_pad, T* data,
                                           void** data_array, int max_block,
                                           int layout_size, int block_size) {
  int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  int64_t total_tokens = (int64_t)max_block * block_size;
  if (total_tokens_post_pad != nullptr) {
    total_tokens = total_tokens_post_pad[0];
  }
  if (tid < N) {
    int64_t dst_num = 0;
    if (experts_idx == nullptr) {
      dst_num = tid;
    } else {
      if (tid * block_size >= total_tokens) {
        dst_num = 0;
      } else {
        dst_num = experts_idx[tid];
      }
    }
    data_array[tid] = data + dst_num * layout_size;
  }
}
template <typename T>
void MOEGetBatchArrayLauncher(int64_t* experts_idx, int* total_tokens_post_pad,
                              T* data, void** data_array, int max_block,
                              int layout_size, int block_size,
                              cudaStream_t stream) {
  const int N = max_block;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  MOE_get_batch_array_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, experts_idx, total_tokens_post_pad, data, data_array, max_block,
      layout_size, block_size);
}
template void MOEGetBatchArrayLauncher<float>(int64_t* experts_idx,
                                              int* total_tokens_post_pad,
                                              float* data, void** data_array,
                                              int max_block, int layout_size,
                                              int block_size,
                                              cudaStream_t stream);
#ifdef ENABLE_FP16
template void MOEGetBatchArrayLauncher<half>(int64_t* experts_idx,
                                             int* total_tokens_post_pad,
                                             half* data, void** data_array,
                                             int max_block, int layout_size,
                                             int block_size,
                                             cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void MOEGetBatchArrayLauncher<hie::bfloat16>(
    int64_t* experts_idx, int* total_tokens_post_pad, hie::bfloat16* data,
    void** data_array, int max_block, int layout_size, int block_size,
    cudaStream_t stream);
#endif
template void MOEGetBatchArrayLauncher<int8_t>(int64_t* experts_idx,
                                               int* total_tokens_post_pad,
                                               int8_t* data, void** data_array,
                                               int max_block, int layout_size,
                                               int block_size,
                                               cudaStream_t stream);
template void MOEGetBatchArrayLauncher<int32_t>(
    int64_t* experts_idx, int* total_tokens_post_pad, int32_t* data,
    void** data_array, int max_block, int layout_size, int block_size,
    cudaStream_t stream);
template <typename T>
__global__ void mul_and_silu_kernel(int N, T* output, T* gate_out,
                                    T* up_proj_out, int m, int n) {
  int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    T x = gate_out[tid];
    T y = up_proj_out[tid];
    output[tid] = x * (1.0f / (1.0f + expf((T)(-x)))) * y;
  }
}
template <typename T>
void MulAndSilu(T* output, T* gate_out, T* up_proj_out, int m, int n,
                cudaStream_t stream) {
  const int64_t N = (int64_t)m * n;
  const int64_t block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  mul_and_silu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, gate_out, up_proj_out, m, n);
}
template void MulAndSilu<float>(float* output, float* gate_out,
                                float* up_proj_out, int m, int n,
                                cudaStream_t stream);
#ifdef ENABLE_FP16
template void MulAndSilu<half>(half* output, half* gate_out, half* up_proj_out,
                               int m, int n, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void MulAndSilu<hie::bfloat16>(hie::bfloat16* output,
                                        hie::bfloat16* gate_out,
                                        hie::bfloat16* up_proj_out, int m,
                                        int n, cudaStream_t stream);
#endif
template <typename T>
__global__ void finalize_kernel(T* output, T* expanded_permuted_rows,
                                float* experts_score, int64_t* indice_source,
                                int* expert_for_source_row, int* num_valid_ptr,
                                int k, int cols) {
  int64_t const original_row = blockIdx.x;
  int64_t const offset = original_row * cols;
  T* reduced_row_ptr = output + offset;
  const int64_t num_valid = *num_valid_ptr;
  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    float thread_output{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int64_t const expanded_original_row = original_row * k + k_idx;
      int64_t const expanded_permuted_row =
          indice_source[expanded_original_row];
      if (expanded_permuted_row >= num_valid) {
        // should impossable
        continue;
      }
      // int64_t const k_offset = original_row * k + k_idx;
      // int64_t const expert_idx = expert_for_source_row[k_offset];
      float const row_scale = (float)experts_score[expanded_original_row];

      // T const* bias_ptr = bias + expert_idx * cols;
      // float const bias_value = bias ? static_cast<float>(bias_ptr[tid]) :
      // 0.f;

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows + expanded_permuted_row * cols;
      float const row_value =
          static_cast<float>(expanded_permuted_rows_row_ptr[tid]);

      thread_output =
          static_cast<float>(thread_output) + row_scale * (row_value);
    }
    reduced_row_ptr[tid] = static_cast<T>(thread_output);
  }
}
template <typename T>
void FinalizeMoeRoutingKernelLauncher(
    T* output, T* fianl_result, float* experts_score, int64_t* indice_source,
    int* expert_for_source_row, int* total_tokens_pad_ptr, int total_token,
    int top_k, int hidden_size, cudaStream_t stream) {
  const int block_num = total_token;
  finalize_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      output, fianl_result, experts_score, indice_source, expert_for_source_row,
      total_tokens_pad_ptr, top_k, hidden_size);
}
template void FinalizeMoeRoutingKernelLauncher<float>(
    float* output, float* fianl_result, float* experts_score,
    int64_t* indice_source, int* expert_for_source_row,
    int* total_tokens_pad_ptr, int total_token, int top_k, int hidden_size,
    cudaStream_t stream);
#ifdef ENABLE_FP16
template void FinalizeMoeRoutingKernelLauncher<half>(
    half* output, half* fianl_result, float* experts_score,
    int64_t* indice_source, int* expert_for_source_row,
    int* total_tokens_pad_ptr, int total_token, int top_k, int hidden_size,
    cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void FinalizeMoeRoutingKernelLauncher<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* fianl_result, float* experts_score,
    int64_t* indice_source, int* expert_for_source_row,
    int* total_tokens_pad_ptr, int total_token, int top_k, int hidden_size,
    cudaStream_t stream);
#endif
template <typename T>
__global__ void finalize_new_kernel(T* output, T* final_result,
                                    float* experts_score, int* mid_row_indices,
                                    int* final_row_indices, int total_token,
                                    int k, int cols) {
  int64_t const original_row = blockIdx.x;
  int64_t const offset = original_row * cols;
  T* reduced_row_ptr = output + offset;
  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    float thread_output{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int64_t origin_row_idx = original_row * k + k_idx;
      int64_t mid_row_idx = mid_row_indices[origin_row_idx];
      int64_t final_row_idx = final_row_indices[mid_row_idx];
      float const row_scale = (float)experts_score[origin_row_idx];
      float const row_value =
          static_cast<float>(final_result[final_row_idx * cols + tid]);
      thread_output =
          static_cast<float>(thread_output) + row_scale * (row_value);
    }
    reduced_row_ptr[tid] = static_cast<T>(thread_output);
  }
}
template <typename T>
void FinalizeMoeRoutingNewKernelLauncher(T* output, T* fianl_result,
                                         float* experts_score,
                                         int* mid_row_indices,
                                         int* final_row_indices,
                                         int total_token, int top_k,
                                         int hidden_size, cudaStream_t stream) {
  const int64_t block_num = total_token;
  finalize_new_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      output, fianl_result, experts_score, mid_row_indices, final_row_indices,
      total_token, top_k, hidden_size);
}
template void FinalizeMoeRoutingNewKernelLauncher<float>(
    float* output, float* fianl_result, float* experts_score,
    int* mid_row_indices, int* final_row_indices, int total_token, int top_k,
    int hidden_size, cudaStream_t stream);
#ifdef ENABLE_FP16
template void FinalizeMoeRoutingNewKernelLauncher<half>(
    half* output, half* fianl_result, float* experts_score,
    int* mid_row_indices, int* final_row_indices, int total_token, int top_k,
    int hidden_size, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void FinalizeMoeRoutingNewKernelLauncher<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* fianl_result, float* experts_score,
    int* mid_row_indices, int* final_row_indices, int total_token, int top_k,
    int hidden_size, cudaStream_t stream);
#endif
__global__ void get_expert_kernel(int N, int* expert_indices,
                                  const int* in_expert_indices,
                                  const int* row_indices, int total_token,
                                  int topk, int num_expert) {
  int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int64_t row_id = row_indices[tid];
    int64_t expert_id = in_expert_indices[tid];
    if (row_id < N && expert_id < num_expert) {
      expert_indices[row_id] = expert_id;
    }
  }
}
void GetExpertByIndice(int* expert_indices, const int* in_expert_indices,
                       const int* row_indices, int total_token, int topk,
                       int num_expert, cudaStream_t stream) {
  const int64_t N = (int64_t)total_token * topk;
  const int64_t block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  get_expert_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, expert_indices, in_expert_indices, row_indices, total_token, topk,
      num_expert);
}
}  // namespace cuda
}  // namespace allspark
