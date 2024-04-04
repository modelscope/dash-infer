/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    beam_search.cpp
 */

#include <math.h>

#include <cstring>
#include <iostream>
#include <set>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void BeamScoreInitLauncher(T* beam_score, T* hyps_beam_score,
                           int64_t* hyps_beam_idx, int* eos_count,
                           int batch_size, int beam_size) {
  int N = batch_size * beam_size;
  parallel_for(N, [&](int i) {
    int beam_id = i % beam_size;
    if (beam_id == 0) {
      beam_score[i] = 0;
      eos_count[i / beam_size] = 0;
    } else {
      beam_score[i] = -1e9f;
    }
    hyps_beam_score[i] = -1e9f;
    hyps_beam_idx[i] = 0;
  });
}
template void BeamScoreInitLauncher<float>(float* beam_score,
                                           float* hyps_beam_score,
                                           int64_t* hyps_beam_idx,
                                           int* eos_count, int batch_size,
                                           int beam_size);
template <typename T>
void add_score_kernel(T* next_score, const T* beam_score, int length, int N) {
  parallel_for(N, [&](int tid) {
    int batch_idx = tid / length;
    next_score[tid] += beam_score[batch_idx];
  });
}
template <typename T>
void AddScoreLauncher(T* next_score, const T* beam_score, int batch,
                      int length) {
  int N = batch * length;
  add_score_kernel(next_score, beam_score, length, N);
}

template <typename T>
void UpdateBeamScoreLauncher(T* beam_score, int64_t* beam_next_token,
                             int* beam_idx, const T* topk_score,
                             const int64_t* topk_indice, int batch_size,
                             int beam_size, int vocab_size, int eos,
                             int* eos_count, int* hyps_beam_idx,
                             T* hyps_beam_score, int64_t* hyps_id,
                             int64_t* in_ids, int cur_len, int loop_len,
                             float length_penalty) {
  int N = batch_size;
  parallel_for(N, [&](int tid) {
    if (hyps_beam_score[(tid + 1) * beam_size - 1] > (T)-1000.0) {
      return;
    }
    int len = beam_size * 2;
    int count = 0;
    int cur_eos_count = 0;
    bool tmp_update_hyps = false;
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
  });
}

template void UpdateBeamScoreLauncher<float>(
    float* beam_score, int64_t* beam_next_token, int* beam_idx,
    const float* topk_score, const int64_t* topk_indice, int batch_size,
    int beam_size, int vocab_size, int eos, int* eos_count, int* hyps_beam_idx,
    float* hyps_beam_score, int64_t* hyps_id, int64_t* in_ids, int cur_len,
    int loop_len, float length_penalty);

template <typename T>
void ReorderKVCacheLauncher(T* k, T* v, const T* old_k, const T* old_v,
                            const int* beam_idx, int batch_size, int beam_size,
                            int inner_dim) {
  int N = batch_size * beam_size * inner_dim;
  if (beam_idx == nullptr) {
    parallel_for(N, [&](int tid) {
      int idx1 = tid / inner_dim / beam_size;
      int beam = tid / inner_dim % beam_size;
      int idx2 = tid % inner_dim;
      k[tid] = old_k[(idx1)*inner_dim + idx2];
      v[tid] = old_v[(idx1)*inner_dim + idx2];
    });
  } else {
    parallel_for(N, [&](int tid) {
      int idx1 = tid / inner_dim / beam_size;
      int beam = tid / inner_dim % beam_size;
      int idx2 = tid % inner_dim;
      k[tid] = old_k[(idx1 * beam_size + beam_idx[idx1 * beam_size + beam]) *
                         inner_dim +
                     idx2];
      v[tid] = old_v[(idx1 * beam_size + beam_idx[idx1 * beam_size + beam]) *
                         inner_dim +
                     idx2];
    });
  }
}
template void ReorderKVCacheLauncher<float>(float* k, float* v,
                                            const float* old_k,
                                            const float* old_v,
                                            const int* beam_idx, int batch_size,
                                            int beam_size, int inner_dim);

template <typename T>
void rep_logits_processor(T* score_in, T* score_out, const int64_t* in_ids,
                          int64_t in_ids_len, int cur_len, int max_len,
                          int vocab_size, float rep_penalty,
                          GenerateConfig gen_cfg, int N) {
  for (int tid = 0; tid < N; tid++) {
    int idx1 = tid / cur_len;
    int idx2 = tid % cur_len;
    int src_idx = idx1 * max_len + idx2;
    if (gen_cfg.suppress_repetition_in_generation) {
      if (idx2 < gen_cfg.input_len) {
        return;
      }
    }
    if (src_idx > in_ids_len) {
      return;
    }
    const int vocab_offset = in_ids[src_idx];
    if (vocab_offset < 0 || vocab_offset >= vocab_size) return;
    int dst_idx = idx1 * vocab_size + vocab_offset;

    if (score_in[dst_idx] < (T)(0.f)) {
      score_out[dst_idx] = score_in[dst_idx] * (T)rep_penalty;
    } else {
      score_out[dst_idx] = score_in[dst_idx] / (T)rep_penalty;
    }
  }
}
template <typename T>
void presence_logits_processor(T* score_in, T* score_out, const int64_t* in_ids,
                               int64_t in_ids_len, int cur_len, int max_len,
                               int vocab_size, float presence_penalty,
                               GenerateConfig gen_cfg, int N) {
  for (int tid = 0; tid < N; tid++) {
    int idx1 = tid / cur_len;
    int idx2 = tid % cur_len;
    int src_idx = idx1 * max_len + idx2;
    if (gen_cfg.suppress_repetition_in_generation) {
      if (idx2 < gen_cfg.input_len) {
        return;
      }
    }
    if (src_idx > in_ids_len) {
      return;
    }
    const int vocab_offset = in_ids[src_idx];
    if (vocab_offset < 0 || vocab_offset >= vocab_size) return;
    int dst_idx = idx1 * vocab_size + vocab_offset;

    if (score_in[dst_idx] < (T)(0.f)) {
      score_out[dst_idx] = score_in[dst_idx] - (T)presence_penalty;
    } else {
      score_out[dst_idx] = score_in[dst_idx] - (T)presence_penalty;
    }
  }
}
template <typename T>
void n_gram_logits_processor(T* score, const int64_t* in_ids, int batch_size,
                             int cur_len, int max_len, int vocab_size,
                             int ngram, int N) {
  parallel_for(N, [&](int tid) {
    int idx1 = tid / cur_len;
    int idx2 = tid % cur_len;
    int src_idx = idx1 * max_len + idx2;
    if (idx2 + ngram - 2 < cur_len - 1) {
      bool flag = true;
      for (int i = 0; i < ngram - 1; i++) {
        if (in_ids[src_idx + i] !=
            in_ids[idx1 * max_len + cur_len - ngram + i + 1]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        int dst_idx = idx1 * vocab_size + in_ids[src_idx + ngram - 1];
        score[dst_idx] = -1e9f;
      }
    }
  });
}
template <typename T>
void min_length_logits_processor(T* score, int eos, int vocab_size, int N) {
  parallel_for(N, [&](int tid) { score[tid * vocab_size + eos] = -1e9f; });
}
template <typename T>
void no_bad_ids_processor(T* score, int* bad_ids, int bad_id_len,
                          const int64_t* in_ids, int cur_len, int max_len,
                          int vocab_size, int batch_size) {
  parallel_for(batch_size, [&](int batch) {
    bool flag = true;
    for (int i = 0; i < bad_id_len - 1; ++i) {
      int in_idx = batch * max_len + cur_len - bad_id_len + 1 + i;
      if (bad_ids[i] != in_ids[in_idx]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      score[batch * vocab_size + bad_ids[bad_id_len - 1]] = -1e9f;  //-inf;
    }
  });
}
template <typename T>
void LogitsProcessor(T* score, const int64_t* in_ids, int64_t in_ids_len,
                     int batch_size, int cur_len, int max_len, int vocab_size,
                     int* bad_words_ids, std::vector<int>& bad_ids_size,
                     GenerateConfig gen_cfg, void* ws_ptr, size_t ws_bytes) {
  int N = batch_size * cur_len;

  float repetition_penalty = gen_cfg.repetition_penalty;
  float presence_penalty = gen_cfg.presence_penalty;
  int no_repeat_ngram_size = gen_cfg.no_repeat_ngram_size;
  int min_length = gen_cfg.min_length;
  int eos_token_id = gen_cfg.eos_token_id;

  T* score_in = (T*)ws_ptr;

  if (std::fabs(repetition_penalty - 1.0f) > 1e-9) {
    memcpy(score_in, score, vocab_size * sizeof(T));
    rep_logits_processor<T>(score_in, score, in_ids, in_ids_len, cur_len,
                            max_len, vocab_size, repetition_penalty, gen_cfg,
                            N);
  }

  if (std::fabs(presence_penalty - 0.0f) > 1e-9) {
    memcpy(score_in, score, vocab_size * sizeof(T));
    presence_logits_processor<T>(score_in, score, in_ids, in_ids_len, cur_len,
                                 max_len, vocab_size, presence_penalty, gen_cfg,
                                 N);
  }

  if (no_repeat_ngram_size != 0 && cur_len > no_repeat_ngram_size) {
    n_gram_logits_processor<T>(score, in_ids, batch_size, cur_len, max_len,
                               vocab_size, no_repeat_ngram_size, N);
  }

  if (cur_len < min_length) {
    int N = batch_size;
    min_length_logits_processor<T>(score, eos_token_id, vocab_size, N);
  }

  if (bad_words_ids != nullptr && bad_ids_size.size() > 0) {
    int num_bad_words = bad_ids_size.size();
    int* cur_bad_words = bad_words_ids;
    for (int i = 0; i < num_bad_words; ++i) {
      if (cur_len >= bad_ids_size[i] - 1) {
        no_bad_ids_processor<T>(score, cur_bad_words, bad_ids_size[i], in_ids,
                                cur_len, max_len, vocab_size, batch_size);
      }
      cur_bad_words += bad_ids_size[i];
    }
  }
}

template void LogitsProcessor<float>(float* score, const int64_t* in_ids,
                                     int64_t in_ids_len, int batch_size,
                                     int cur_len, int max_len, int vocab_size,
                                     int* bad_words_ids,
                                     std::vector<int>& bad_ids_size,
                                     GenerateConfig gen_cfg, void* ws_ptr,
                                     size_t ws_bytes);

template void AddScoreLauncher<float>(float* next_score,
                                      const float* beam_score, int batch,
                                      int length);
template <>
void CopyMultiBeam<float>(void* A_, void* B_, int batch_size, int beam_size,
                          int inner_dim) {
  int type_size = sizeof(float);
  float* A = static_cast<float*>(A_);
  float* B = static_cast<float*>(B_);
  for (int i = 0; i < batch_size; i++)
    for (int j = 0; j < beam_size; j++) {
      for (int k = 0; k < inner_dim; k++) {
        B[(i * beam_size + j) * inner_dim + k] = A[i * inner_dim + k];
      }
    }
}
template <>
void CopyMultiBeam<int64_t>(void* A_, void* B_, int batch_size, int beam_size,
                            int inner_dim) {
  int type_size = sizeof(int64_t);
  int64_t* A = static_cast<int64_t*>(A_);
  int64_t* B = static_cast<int64_t*>(B_);
  for (int i = 0; i < batch_size; i++)
    for (int j = 0; j < beam_size; j++) {
      memcpy(B + (i * beam_size + j) * inner_dim, A + i * inner_dim,
             inner_dim * sizeof(int64_t));
    }
}
}  // namespace cpu
}  // namespace allspark
