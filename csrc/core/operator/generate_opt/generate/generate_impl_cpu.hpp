/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_impl_cpu.hpp
 */

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <algorithm>
#include <limits>
#include <random>

#include "generate_op.h"  // NOLINT

#ifdef ENABLE_MULTINUMA
#include <mpi.h>

#include "cpu/mpi_utils.hpp"
#endif

namespace allspark {

AsStatus copy_matrix_cpu(DataType dtype, void* in_ptr, void* new_ptr, int M,
                         int N, int lda, int ldb, const DeviceContext* ctx) {
  auto functor = [&]<typename T>() {
    T* A = static_cast<T*>(in_ptr);
    T* B = static_cast<T*>(new_ptr);
    cpu::CopyMatrix(M, N, A, lda, B, ldb);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus logprobs_cpu(DataType dtype, void* in_logits, int64_t* out_tokens,
                      void* token_logprobs, void* logprobs, void* topk_value,
                      int* topk_indice, int batch_size, int length,
                      RuntimeContext* runtime_ctx, void* ws_ptr,
                      size_t ws_bytes, const DeviceContext* ctx) {
  auto functor = [&]<typename T>() {
    T* typed_topk_value = static_cast<T*>(topk_value);
    T* typed_in_logits = static_cast<T*>(in_logits);
    T* typed_logprobs = static_cast<T*>(logprobs);
    int top_logprobs = ctx->GetMaxTopLogprobs();
    cpu::LogSoftmaxKernel(typed_in_logits, typed_logprobs, batch_size, length);
    cpu::TopKKernel(typed_topk_value, topk_indice, typed_logprobs, batch_size,
                    length, top_logprobs);
    runtime_ctx->logprobs_indice_host.reserve(batch_size * top_logprobs);
    runtime_ctx->logprobs_value_host.reserve(batch_size * top_logprobs);
    runtime_ctx->token_logprobs_host.reserve(batch_size);
    for (int i = 0; i < batch_size * top_logprobs; i++) {
      runtime_ctx->logprobs_indice_host[i] = topk_indice[i];
      runtime_ctx->logprobs_value_host[i] = (float)typed_topk_value[i];
    }
    for (int i = 0; i < batch_size; i++) {
      runtime_ctx->token_logprobs_host[i] =
          (float)(typed_logprobs[i * length + out_tokens[i]]);
    }
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

void gen_beam_init_cpu(DataType dtype, void* beam_score, void* hyps_beam_score,
                       int64_t* hyps_beam_idx, int* eos_count, int batch_size,
                       int beam_size, const DeviceContext* ctx) {
  auto functor = [&]<typename T>() {
    T* typed_beam_score = static_cast<T*>(beam_score);
    T* typed_hyps_beam_score = static_cast<T*>(hyps_beam_score);
    cpu::BeamScoreInitLauncher(typed_beam_score, typed_hyps_beam_score,
                               hyps_beam_idx, eos_count, batch_size, beam_size);
  };
  DispatchCPU(dtype, functor);
}

AsStatus gen_process_logits_cpu(DataType dtype, int64_t* ori_in_tokens,
                                void* ori_in_logits, int batch_size, int length,
                                const DeviceContext* ctx,
                                RuntimeContext* runtime_ctx,
                                BatchGencfg batch_gencfg, void* ws_ptr,
                                size_t ws_bytes) {
  // for batch
  std::vector<int> bad_words_ids_size(0);
  if (runtime_ctx->is_context) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
    int64_t* in_tokens = static_cast<int64_t*>(ori_in_tokens);
    void* in_logits = (ori_in_logits);
    auto functor = [&]<typename T>() {
      T* typed_in = static_cast<T*>(in_logits);
      int cur_len = gen_ctx->step + gen_ctx->in_length_bias;
      int64_t in_ids_len = batch_size * ctx->GetModelMaxLength();

      cpu::LogitsProcessor(typed_in, in_tokens, in_ids_len, batch_size, cur_len,
                           ctx->GetModelMaxLength(), length, nullptr,
                           bad_words_ids_size, gen_ctx->gen_cfg, ws_ptr,
                           ws_bytes);
    };
    DispatchCPU(dtype, functor);
  } else {
    for (int i = 0; i < batch_size; i++) {
      std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(i);
      int64_t* in_tokens = static_cast<int64_t*>(ori_in_tokens) +
                           gen_ctx->current_batch * ctx->GetModelMaxLength();
      void* in_logits = ((char*)ori_in_logits + i * length * SizeofType(dtype));
      auto functor = [&]<typename T>() {
        T* typed_in = static_cast<T*>(in_logits);
        int cur_len = gen_ctx->step + gen_ctx->in_length_bias;
        int64_t in_ids_len = batch_size * ctx->GetModelMaxLength();

        cpu::LogitsProcessor(typed_in, in_tokens, in_ids_len, 1, cur_len,
                             ctx->GetModelMaxLength(), length, nullptr,
                             bad_words_ids_size, gen_ctx->gen_cfg, ws_ptr,
                             ws_bytes);
      };
      DispatchCPU(dtype, functor);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus gen_sample_cpu(DataType dtype, int64_t* total_out_tokens,
                        void* total_topk_value, void* total_topp_value,
                        int* total_topk_indice, void* total_in_logits,
                        void** sample_states_vec_unused, int total_batch_size,
                        int max_k, int length, int* k_arr_list,
                        float* p_arr_list, float* temperature_arr_unused,
                        const DeviceContext* ctx, RuntimeContext* runtime_ctx,
                        void* ws_ptr, size_t ws_bytes, void* device_prop) {
  constexpr int batch_size = 1;
  for (int i = 0; i < total_batch_size; i++) {
    std::shared_ptr<GenerateContext> gen_ctx;
    if (runtime_ctx->is_context) {
      gen_ctx = runtime_ctx->GetContextGenCtx();
    } else {
      gen_ctx = runtime_ctx->GetGenCtx(i);
    }
    void* sample_states = gen_ctx->sample_state->GetDataPtr();
    float temperature = gen_ctx->gen_cfg.temperature;
    int* k_arr = k_arr_list + i;
    float* p_arr = p_arr_list + i;

    if (max_k <= 0) max_k = length;
    if (*k_arr <= 0) *k_arr = length;

    void* in_logits =
        (void*)((char*)total_in_logits + i * length * SizeofType(dtype));
    int64_t* out_tokens = total_out_tokens + i;

    auto functor = [&]<typename T>() {
      T* typed_topk_value = static_cast<T*>(total_topk_value);
      T* typed_topp_value = static_cast<T*>(total_topp_value);
      T* typed_in = static_cast<T*>(in_logits);

      cpu::TopKKernel(typed_topk_value, total_topk_indice, typed_in, batch_size,
                      length, max_k);
      if (p_arr != nullptr) {
        copy_matrix_cpu(dtype, typed_topk_value, typed_topp_value, batch_size,
                        max_k, max_k, max_k, ctx);
        cpu::SoftmaxKernel(typed_topp_value, k_arr, batch_size, max_k,
                           temperature);
        cpu::TopPKernel(typed_topp_value, k_arr, p_arr, batch_size, max_k);
      }
      cpu::SoftmaxKernel(typed_topk_value, k_arr, batch_size, max_k,
                         temperature);
      cpu::SampleKernel<T>(out_tokens, sample_states, typed_topk_value,
                           total_topk_indice, batch_size, k_arr, max_k);
    };
    DispatchCPU(dtype, functor);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

void gen_sample_init_cpu(void* sample_state, unsigned long long seed,
                         int batch_size, const DeviceContext* ctx) {
  cpu::SampleKernelInitLauncher(sample_state, seed, batch_size);
}

AsStatus fill_generated_ids_cpu(RuntimeContext* runtime_ctx,
                                std::shared_ptr<AsTensor>& dec_ids,
                                int rank_id) {
  if (rank_id != 0) {
    return AsStatus::ALLSPARK_SUCCESS;
  }

  int batch_size = 1;
  if (!runtime_ctx->is_context) {
    batch_size = runtime_ctx->GetGenCtxListSize();
  }

  std::vector<int64_t*> ptrs_host(batch_size);
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<GenerateContext> gen_ctx;
    if (runtime_ctx->is_context) {
      gen_ctx = runtime_ctx->GetContextGenCtx();
    } else {
      gen_ctx = runtime_ctx->GetGenCtx(i);
    }

    int generated_len = gen_ctx->step + gen_ctx->in_length_bias + 1;
    std::shared_ptr<AsTensor> generated_ids_tensor =
        gen_ctx->request->interim.at("generated_ids");
    generated_ids_tensor->SetShape(Shape{1, generated_len});
    ptrs_host[i] = static_cast<int64_t*>(generated_ids_tensor->GetDataPtr()) +
                   generated_len - 1;
  }

  // dec_ids -> generated_ids
  cpu::CopyToVarsKernelLauncher(
      ptrs_host.data(), static_cast<const int64_t*>(dec_ids->GetDataPtr()),
      batch_size);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus fill_max_dec_ids_cpu(RuntimeContext* runtime_ctx,
                              std::shared_ptr<AsTensor>& max_dec_ids,
                              const DeviceContext* ctx) {
  int batch_size = 1;
  if (!runtime_ctx->is_context) {
    batch_size = runtime_ctx->GetGenCtxListSize();
  }

  // memset(max_dec_ids->GetDataPtr(), 0, max_dec_ids->GetSizeInByte());

  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<GenerateContext> gen_ctx;
    if (runtime_ctx->is_context) {
      gen_ctx = runtime_ctx->GetContextGenCtx();
    } else {
      gen_ctx = runtime_ctx->GetGenCtx(i);
    }

    std::shared_ptr<AsTensor> generated_ids_tensor =
        gen_ctx->request->interim.at("generated_ids");

    void* dst_data = (void*)((int64_t*)max_dec_ids->GetDataPtr() +
                             gen_ctx->current_batch * ctx->GetModelMaxLength());
    void* src_data = generated_ids_tensor->GetDataPtr();
    size_t nbytes = generated_ids_tensor->GetSizeInByte();
    memcpy(dst_data, src_data, nbytes);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

#ifdef ENABLE_MULTINUMA
AsStatus MpiBcast(std::shared_ptr<AsTensor> tensor) {
  void* out = tensor->GetDataPtr();
  int count = tensor->GetShape().Count();
  DataType dtype = tensor->GetDataType();
  MPI_Datatype mpi_dtype = GetMpiType(dtype);
  MPI_Bcast(out, count, mpi_dtype, 0, MPI_COMM_WORLD);

  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

}  // namespace allspark
