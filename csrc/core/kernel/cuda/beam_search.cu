/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    beam_search.cu
 */

#include <limits>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "utility/check_cuda.h"
namespace allspark {
namespace cuda {

template <typename T>
__global__ static void beam_score_init_kernel(T* beam_score, T* hyps_beam_score,
                                              int64_t* hyps_beam_idx,
                                              int* eos_count, int beam_size,
                                              int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int beam_id = tid % beam_size;
    if (beam_id == 0) {
      beam_score[tid] = (T)0.0;
      eos_count[tid / beam_size] = 0;
    } else {
      beam_score[tid] = -1e9f;
    }
    hyps_beam_score[tid] = -1e9f;
    hyps_beam_idx[tid] = 0;
  }
}

template <typename T>
void BeamScoreInitLauncher(T* beam_score, T* hyps_beam_score,
                           int64_t* hyps_beam_idx, int* eos_count,
                           int batch_size, int beam_size, cudaStream_t stream) {
  int N = batch_size * beam_size;
  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  beam_score_init_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      beam_score, hyps_beam_score, hyps_beam_idx, eos_count, beam_size, N);
}

template void BeamScoreInitLauncher(float* beam_score, float* hyps_beam_score,
                                    int64_t* hyps_beam_idx, int* eos_count,

                                    int batch_size, int beam_size,
                                    cudaStream_t stream);
#ifdef ENABLE_FP16
template void BeamScoreInitLauncher(half* beam_score, half* hyps_beam_score,
                                    int64_t* hyps_beam_idx, int* eos_count,
                                    int batch_size, int beam_size,
                                    cudaStream_t stream);
#endif
template void BeamScoreInitLauncher(hie::bfloat16* beam_score,
                                    hie::bfloat16* hyps_beam_score,
                                    int64_t* hyps_beam_idx, int* eos_count,
                                    int batch_size, int beam_size,
                                    cudaStream_t stream);
template <typename T>
__global__ static void add_score_kernel(T* next_score, const T* beam_score,
                                        int length, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int batch_idx = tid / length;
    next_score[tid] += beam_score[batch_idx];
  }
}

template <typename T>
void AddScoreLauncher(T* next_score, const T* beam_score, int batch, int length,
                      cudaStream_t stream) {
  int N = batch * length;
  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  add_score_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      next_score, beam_score, length, N);
}

template void AddScoreLauncher(float* next_score, const float* beam_score,
                               int batch, int length, cudaStream_t stream);
#ifdef ENABLE_FP16
template void AddScoreLauncher(half* next_score, const half* beam_score,
                               int batch, int length, cudaStream_t stream);
#endif
template void AddScoreLauncher(hie::bfloat16* next_score,
                               const hie::bfloat16* beam_score, int batch,
                               int length, cudaStream_t stream);

template <typename T>
__global__ static void update_beamscore_kernel(
    T* beam_score, int64_t* beam_next_token, int* beam_idx, const T* topk_score,
    const int* topk_indice, int beam_size, int vocab_size, int eos,
    int* eos_count, int* hyps_beam_idx, T* hyps_beam_score, int64_t* hyps_id,
    int64_t* in_ids, int cur_len, int loop_len, float length_penalty, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    if (hyps_beam_score[(tid + 1) * beam_size - 1] > (T)-1000.0) {
      return;
    }
    int len = beam_size * 2;
    int count = 0;
    int cur_eos_count = 0;
    for (int i = 0; i < len; ++i) {
      if (count == beam_size) {
        break;
      }
      int src_idx = tid * len + i;
      int dst_idx = tid * beam_size + count;
      int token = topk_indice[src_idx] % vocab_size;
      if (token != eos) {
        beam_next_token[dst_idx] = token;
        beam_score[dst_idx] = topk_score[src_idx];
        beam_idx[dst_idx] = topk_indice[src_idx] / vocab_size;
        ++count;
      } else if (i < beam_size) {
        ++cur_eos_count;
        T score = topk_score[src_idx] / (T)powf(cur_len + 1, length_penalty);
        // insert sort
        for (int j = beam_size - 1; j >= 0; --j) {
          int cur_idx = tid * beam_size + j;
          if (hyps_beam_score[cur_idx] < score) {
            if (j == 0) {
              hyps_beam_score[cur_idx] = score;
              hyps_beam_idx[cur_idx] = topk_indice[src_idx] / vocab_size;
              for (int k = 0; k < cur_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] =
                    in_ids[(tid * beam_size + hyps_beam_idx[cur_idx]) *
                               loop_len +
                           k];
              }
              hyps_id[cur_idx * loop_len + cur_len] = eos;
              for (int k = cur_len + 1; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = 0;
              }
            } else {
              hyps_beam_score[cur_idx] = hyps_beam_score[cur_idx - 1];
              hyps_beam_idx[cur_idx] = hyps_beam_idx[cur_idx - 1];
              for (int k = 0; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = hyps_id[(cur_idx - 1) * loop_len + k];
              }
            }

          } else {
            if (j != beam_size - 1) {
              cur_idx += 1;
              hyps_beam_score[cur_idx] = score;
              hyps_beam_idx[cur_idx] = topk_indice[src_idx] / vocab_size;
              for (int k = 0; k < cur_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] =
                    in_ids[(tid * beam_size + hyps_beam_idx[cur_idx]) *
                               loop_len +
                           k];
              }
              hyps_id[cur_idx * loop_len + cur_len] = eos;
              for (int k = cur_len + 1; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = 0;
              }
            }
            break;
          }
        }
      }
    }
    eos_count[tid] = cur_eos_count;
    // check if finalize, compare beam_score & hyps_beam_score
    if (cur_len == loop_len - 1) {
      for (int i = 0; i < count; ++i) {
        if (eos_count[tid] >= beam_size) {
          break;
        }
        eos_count[tid]++;
        int dst_idx = tid * beam_size + i;
        T score = beam_score[dst_idx] / (T)powf(cur_len + 1, length_penalty);
        for (int j = beam_size - 1; j >= 0; --j) {
          int cur_idx = tid * beam_size + j;
          if (hyps_beam_score[cur_idx] < score) {
            if (j == 0) {
              hyps_beam_score[cur_idx] = score;
              hyps_beam_idx[cur_idx] = beam_idx[dst_idx];
              for (int k = 0; k < cur_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] =
                    in_ids[(tid * beam_size + hyps_beam_idx[cur_idx]) *
                               loop_len +
                           k];
              }
              hyps_id[cur_idx * loop_len + cur_len] = beam_next_token[dst_idx];
              for (int k = cur_len + 1; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = 0;
              }
            } else {
              hyps_beam_score[cur_idx] = hyps_beam_score[cur_idx - 1];
              hyps_beam_idx[cur_idx] = hyps_beam_idx[cur_idx - 1];
              for (int k = 0; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = hyps_id[(cur_idx - 1) * loop_len + k];
              }
            }

          } else {
            if (j != beam_size - 1) {
              cur_idx += 1;
              hyps_beam_score[cur_idx] = score;
              hyps_beam_idx[cur_idx] = beam_idx[dst_idx];
              for (int k = 0; k < cur_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] =
                    in_ids[(tid * beam_size + hyps_beam_idx[cur_idx]) *
                               loop_len +
                           k];
              }
              hyps_id[cur_idx * loop_len + cur_len] = beam_next_token[dst_idx];
              for (int k = cur_len + 1; k < loop_len; k++) {
                int now_idx = cur_idx * loop_len + k;
                hyps_id[now_idx] = 0;
              }
            }
            break;
          }
        }
      }
    }
  }
}

template <typename T>
void UpdateBeamScoreLauncher(T* beam_score, int64_t* beam_next_token,
                             int* beam_idx, const T* topk_score,
                             const int* topk_indice, int batch_size,
                             int beam_size, int vocab_size, int eos,
                             int* eos_count, int* hyps_beam_idx,
                             T* hyps_beam_score, int64_t* hyps_id,
                             int64_t* in_ids, int cur_len, int loop_len,
                             float length_penalty, cudaStream_t stream) {
  int N = batch_size;
  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  update_beamscore_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      beam_score, beam_next_token, beam_idx, topk_score, topk_indice, beam_size,
      vocab_size, eos, eos_count, hyps_beam_idx, hyps_beam_score, hyps_id,
      in_ids, cur_len, loop_len, length_penalty, N);
}

template void UpdateBeamScoreLauncher(
    float* beam_score, int64_t* beam_next_token, int* beam_idx,
    const float* topk_score, const int* topk_indice, int batch_size,
    int beam_size, int vocab_size, int eos, int* eos_count, int* hyps_beam_idx,
    float* hyps_beam_score, int64_t* hyps_id, int64_t* in_ids, int cur_len,
    int loop_len, float length_penalty, cudaStream_t stream);
#ifdef ENABLE_FP16
template void UpdateBeamScoreLauncher(
    half* beam_score, int64_t* beam_next_token, int* beam_idx,
    const half* topk_score, const int* topk_indice, int batch_size,
    int beam_size, int vocab_size, int eos, int* eos_count, int* hyps_beam_idx,
    half* hyps_beam_score, int64_t* hyps_id, int64_t* in_ids, int cur_len,
    int loop_len, float length_penalty, cudaStream_t stream);
#endif
template void UpdateBeamScoreLauncher(
    hie::bfloat16* beam_score, int64_t* beam_next_token, int* beam_idx,
    const hie::bfloat16* topk_score, const int* topk_indice, int batch_size,
    int beam_size, int vocab_size, int eos, int* eos_count, int* hyps_beam_idx,
    hie::bfloat16* hyps_beam_score, int64_t* hyps_id, int64_t* in_ids,
    int cur_len, int loop_len, float length_penalty, cudaStream_t stream);
template <typename T>
__global__ static void reorder_kv_cache_kernel(T* k, T* v, const T* old_k,
                                               const T* old_v,
                                               const int* beam_idx,
                                               int batch_size, int beam_size,
                                               int inner_dim, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int idx1 = tid / inner_dim / beam_size;
    int beam = tid / inner_dim % beam_size;
    int idx2 = tid % inner_dim;
    k[tid] = old_k[(idx1 * beam_size + beam_idx[idx1 * beam_size + beam]) *
                       inner_dim +
                   idx2];
    v[tid] = old_v[(idx1 * beam_size + beam_idx[idx1 * beam_size + beam]) *
                       inner_dim +
                   idx2];
  }
}
template <typename T>
__global__ static void reorder_kv_cache_kernel_nobeam(
    T* k, T* v, const T* old_k, const T* old_v, const int* beam_idx,
    int batch_size, int beam_size, int inner_dim, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int idx1 = tid / inner_dim / beam_size;
    int idx2 = tid % inner_dim;
    k[tid] = old_k[(idx1)*inner_dim + idx2];
    v[tid] = old_v[(idx1)*inner_dim + idx2];
  }
}
template <typename T>
void ReorderKVCacheLauncher(T* k, T* v, const T* old_k, const T* old_v,
                            const int* beam_idx, int batch_size, int beam_size,
                            int inner_dim, cudaStream_t stream) {
  int N = batch_size * beam_size * inner_dim;
  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  if (beam_idx == nullptr) {
    reorder_kv_cache_kernel_nobeam<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        k, v, old_k, old_v, beam_idx, batch_size, beam_size, inner_dim, N);
  } else {
    reorder_kv_cache_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        k, v, old_k, old_v, beam_idx, batch_size, beam_size, inner_dim, N);
  }
}
template void ReorderKVCacheLauncher(float* k, float* v, const float* old_k,
                                     const float* old_v, const int* beam_idx,
                                     int batch_size, int beam_size,
                                     int inner_dim, cudaStream_t stream);
#ifdef ENABLE_FP16
template void ReorderKVCacheLauncher(half* k, half* v, const half* old_k,
                                     const half* old_v, const int* beam_idx,
                                     int batch_size, int beam_size,
                                     int inner_dim, cudaStream_t stream);
#endif
template void ReorderKVCacheLauncher(hie::bfloat16* k, hie::bfloat16* v,
                                     const hie::bfloat16* old_k,
                                     const hie::bfloat16* old_v,
                                     const int* beam_idx, int batch_size,
                                     int beam_size, int inner_dim,
                                     cudaStream_t stream);
template <typename T>
__global__ static void rep_logits_processor(
    T* score_in, T* score_out, const int64_t* in_ids, int* cur_len_list,
    int max_len, int vocab_size, float* repetition_penalty_list,
    int* suppress_repetition_in_generation_list, int* input_len_list, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int batch = tid / max_len;
    int idx2 = tid % max_len;
    int src_idx = batch * max_len + idx2;
    if (suppress_repetition_in_generation_list[batch] != 0) {
      if (idx2 < input_len_list[batch]) {
        return;
      }
    }
    if (src_idx >= cur_len_list[batch]) {
      return;
    }
    const int vocab_offset = in_ids[src_idx];
    if (vocab_offset < 0 || vocab_offset >= vocab_size) return;
    int dst_idx = batch * vocab_size + vocab_offset;
    float rep_penalty = repetition_penalty_list[batch];
    if (score_in[dst_idx] < (T)(0.f)) {
      score_out[dst_idx] = score_in[dst_idx] * (T)rep_penalty;
    } else {
      score_out[dst_idx] = score_in[dst_idx] / (T)rep_penalty;
    }
  }
}
template <typename T>
__global__ static void penalty_logits_processor(int* token_count, T* score_out,
                                                float* frequency_penalty_list,
                                                float* presence_penalty_list,
                                                int vocab_size, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int batch = tid / vocab_size;
    float total_penalty = token_count[tid] * frequency_penalty_list[batch];
    if (token_count[tid] > 0) {
      total_penalty += presence_penalty_list[batch];
    }
    score_out[tid] -= (T)total_penalty;
  }
}
__global__ static void token_count_processor(int* token_count,
                                             const int64_t* in_ids,
                                             int* cur_len_list,
                                             int* input_len_list, int max_len,
                                             int vocab_size, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int batch = tid / max_len;
    int idx2 = tid % max_len;
    int src_idx = batch * max_len + idx2;
    if (idx2 < input_len_list[batch] || idx2 >= cur_len_list[batch]) {
      // only count generate ids
      return;
    }
    const int vocab_offset = in_ids[src_idx];
    if (vocab_offset < 0 || vocab_offset >= vocab_size) return;
    int dst_idx = batch * vocab_size + vocab_offset;
    atomicAdd(&token_count[dst_idx], 1);
  }
}
template <typename T>
__global__ static void n_gram_logits_processor(T* score, const int64_t* in_ids,
                                               int* cur_len_list,
                                               int* ngram_list, int max_len,
                                               int vocab_size, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int batch = tid / max_len;
    int idx2 = tid % max_len;
    int src_idx = batch * max_len + idx2;
    int cur_len = cur_len_list[batch];
    int ngram = ngram_list[batch];
    if (idx2 < cur_len && ngram > 0 && idx2 + ngram - 2 < cur_len - 1) {
      bool flag = true;
      for (int i = 0; i < ngram - 1; i++) {
        if (in_ids[src_idx + i] !=
            in_ids[batch * max_len + cur_len - ngram + i + 1]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        int dst_idx = batch * vocab_size + in_ids[src_idx + ngram - 1];
        score[dst_idx] = -1e9f;
      }
    }
  }
}

template <typename T>
__global__ static void min_length_logits_processor(T* score, int* cur_len_list,
                                                   int* min_length_list,
                                                   int* eos_list,
                                                   int vocab_size, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    if (cur_len_list[tid] < min_length_list[tid]) {
      score[tid * vocab_size + eos_list[tid]] = -1e9f;
    }
  }
}

template <typename T>
__global__ static void no_bad_ids_processor(T* score, int* bad_ids,
                                            int bad_id_len,
                                            const int64_t* in_ids, int cur_len,
                                            int max_len, int vocab_size) {
  if (threadIdx.x == 0) {
    bool flag = true;
    for (int i = 0; i < bad_id_len - 1; ++i) {
      int in_idx = blockIdx.x * max_len + cur_len - bad_id_len + 1 + i;
      if (bad_ids[i] != in_ids[in_idx]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      score[blockIdx.x * vocab_size + bad_ids[bad_id_len - 1]] = -1e9f;  //-inf;
    }
  }
}

template <typename T>
void LogitsProcessor(T* score, const int64_t* in_ids, int batch_size,
                     int max_len, int vocab_size, BatchGencfg batch_gencfg,
                     void* ws_ptr, size_t ws_bytes, cudaStream_t stream) {
  // int N = batch_size * cur_len;
  // int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

  // float repetition_penalty = gen_cfg.repetition_penalty;
  // float presence_penalty = gen_cfg.presence_penalty;
  // float frequency_penalty = gen_cfg.frequency_penalty;
  // int no_repeat_ngram_size = gen_cfg.no_repeat_ngram_size;
  // int min_length = gen_cfg.min_length;
  // int eos_token_id = gen_cfg.eos_token_id;
  int* cur_len_list = (int*)(batch_gencfg.cur_len_list);
  int* input_len_list = (int*)(batch_gencfg.input_len_list);
  int* no_repeat_ngram_size_list =
      (int*)(batch_gencfg.no_repeat_ngram_size_list);
  int* min_length_list = (int*)(batch_gencfg.min_length_list);
  int* eos_token_id_list = (int*)(batch_gencfg.eos_token_id_list);
  float* repetition_penalty_list =
      (float*)(batch_gencfg.repetition_penalty_list);
  float* frequency_penalty_list = (float*)(batch_gencfg.frequency_penalty_list);
  float* presence_penalty_list = (float*)(batch_gencfg.presence_penalty_list);
  int* suppress_repetition_in_generation_list =
      (int*)(batch_gencfg.suppress_repetition_in_generation_list);
  // ######## repetition_penalty
  {
    T* score_in = (T*)ws_ptr;
    AS_CHECK_CUDA(cudaMemcpyAsync(score_in, score,
                                  batch_size * vocab_size * sizeof(T),
                                  cudaMemcpyDeviceToDevice, stream));
    int N = batch_size * max_len;
    int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    rep_logits_processor<T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        score_in, score, in_ids, cur_len_list, max_len, vocab_size,
        repetition_penalty_list, suppress_repetition_in_generation_list,
        input_len_list, N);
  }
  // #########frequency_penalty&presence_penalty
  int* token_count = (int*)ws_ptr;
  {
    int N = batch_size * max_len;
    int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    cudaMemsetAsync(token_count, 0, batch_size * vocab_size * sizeof(int),
                    stream);
    token_count_processor<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        token_count, in_ids, cur_len_list, input_len_list, max_len, vocab_size,
        N);
    N = batch_size * vocab_size;
    block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    penalty_logits_processor<T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        token_count, score, frequency_penalty_list, presence_penalty_list,
        vocab_size, N);
  }
  // #########no_repeat_ngram_size

  {
    int N = batch_size * max_len;
    int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    n_gram_logits_processor<T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        score, in_ids, cur_len_list, no_repeat_ngram_size_list, max_len,
        vocab_size, N);
  }
  // #########min_length
  {
    int N = batch_size;
    int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    min_length_logits_processor<T><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        score, cur_len_list, min_length_list, eos_token_id_list, vocab_size, N);
  }
  // ########## bad_words_ids
  // TODO
  // if (bad_words_ids != nullptr && bad_ids_size.size() > 0) {
  //   int num_bad_words = bad_ids_size.size();
  //   int* cur_bad_words = bad_words_ids;
  //   for (int i = 0; i < num_bad_words; ++i) {
  //     if (cur_len >= bad_ids_size[i] - 1) {
  //       no_bad_ids_processor<T><<<batch_size, 1, 0, stream>>>(
  //           score, cur_bad_words, bad_ids_size[i], in_ids, cur_len, max_len,
  //           vocab_size);
  //     }
  //     cur_bad_words += bad_ids_size[i];
  //   }
  // }
}

template void LogitsProcessor<float>(float* score, const int64_t* in_ids,
                                     int batch_size, int max_len,
                                     int vocab_size, BatchGencfg batch_gencfg,
                                     void* ws_ptr, size_t ws_bytes,
                                     cudaStream_t stream);

#ifdef ENABLE_FP16
template void LogitsProcessor<half>(half* score, const int64_t* in_ids,
                                    int batch_size, int max_len, int vocab_size,
                                    BatchGencfg batch_gencfg, void* ws_ptr,
                                    size_t ws_bytes, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void LogitsProcessor<hie::bfloat16>(
    hie::bfloat16* score, const int64_t* in_ids, int batch_size, int max_len,
    int vocab_size, BatchGencfg batch_gencfg, void* ws_ptr, size_t ws_bytes,
    cudaStream_t stream);
#endif
template <typename T>
__global__ static void copy_multi_beam_kernel(T* A, T* B, int batch_size,
                                              int beam_size, int inner_dim,
                                              int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    int idx1 = tid / inner_dim / beam_size;
    int idx2 = tid % inner_dim;
    B[tid] = A[(idx1)*inner_dim + idx2];
  }
}
template <>
void CopyMultiBeam<int64_t>(void* A, void* B, int batch_size, int beam_size,
                            int inner_dim, cudaStream_t stream) {
  int N = batch_size * beam_size * inner_dim;
  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  copy_multi_beam_kernel<int64_t><<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      (int64_t*)A, (int64_t*)B, batch_size, beam_size, inner_dim, N);
}
}  // namespace cuda
}  // namespace allspark
