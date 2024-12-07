/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_impl_gpu.hpp
 */

#include <check_cuda.h>
#include <core/kernel/cuda/sample.h>
#include <core/kernel/kernel.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#include <algorithm>
#include <limits>
#include <random>

#include "float16.h"
#include "generate_op.h"  // NOLINT
namespace allspark {
// uncomment this line if want to enable radix top-k instead of heap topk.
#define CONFIG_ENABLE_RADIX_TOPK
// uncomment this line if want to enable true top-p.
AsStatus copy_matrix_gpu(DataType dtype, void* in_ptr, void* new_ptr, int M,
                         int N, int lda, int ldb, const DeviceContext* ctx) {
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  auto functor = [&]<typename T>() {
    T* A = static_cast<T*>(in_ptr);
    T* B = static_cast<T*>(new_ptr);
    cuda::CopyMatrix(M, N, A, lda, B, ldb, cu_stream);
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus logprobs_gpu(DataType dtype, void* in_logits, int64_t* out_tokens,
                      void* token_logprobs, void* logprobs, void* topk_value,
                      int* topk_indice, int batch_size, int length,
                      RuntimeContext* runtime_ctx, void* ws_ptr,
                      size_t ws_bytes, const DeviceContext* ctx) {
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_topk_value = static_cast<T*>(topk_value);
    T* typed_in_logits = static_cast<T*>(in_logits);
    T* typed_logprobs = static_cast<T*>(logprobs);
    float* float_logprobs = static_cast<float*>(logprobs);
    float* float_token_logprobs = static_cast<float*>(token_logprobs);
    int top_logprobs = ctx->GetMaxTopLogprobs();
    cuda::StridedLogSoftmaxLauncher(typed_logprobs, typed_in_logits, nullptr,
                                    nullptr, ws_ptr, ws_bytes, batch_size,
                                    length, cu_stream);
    cuda::SelectBatchTokenLogprobLauncher(typed_logprobs, float_token_logprobs,
                                          out_tokens, batch_size, length,
                                          cu_stream);
#ifdef CONFIG_ENABLE_RADIX_TOPK
    /* radix-based top-k */
    cuda::TopKRadixKernelLauncher(typed_topk_value, topk_indice, typed_logprobs,
                                  ws_ptr, batch_size, length, top_logprobs,
                                  cu_stream);
#else
    /* heap-based top-k */
    cuda::TopKKernelLauncher(typed_topk_value, topk_indice, typed_logprobs,
                             batch_size, length, top_logprobs, cu_stream);
#endif

    cuda::ToFloatKernelLauncher(float_logprobs, typed_topk_value,
                                batch_size * top_logprobs, cu_stream);
    runtime_ctx->logprobs_indice_host.reserve(batch_size * top_logprobs);
    runtime_ctx->logprobs_value_host.reserve(batch_size * top_logprobs);
    runtime_ctx->token_logprobs_host.reserve(batch_size);
    cudaMemcpyAsync(runtime_ctx->logprobs_indice_host.data(), topk_indice,
                    batch_size * top_logprobs * sizeof(int),
                    cudaMemcpyDeviceToHost, cu_stream);
    cudaMemcpyAsync(runtime_ctx->logprobs_value_host.data(), float_logprobs,
                    batch_size * top_logprobs * sizeof(float),
                    cudaMemcpyDeviceToHost, cu_stream);
    cudaMemcpyAsync(runtime_ctx->token_logprobs_host.data(),
                    float_token_logprobs, batch_size * sizeof(float),
                    cudaMemcpyDeviceToHost, cu_stream);
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
void gen_beam_init_gpu(DataType dtype, void* beam_score, void* hyps_beam_score,
                       int64_t* hyps_beam_idx, int* eos_count, int batch_size,
                       int beam_size, const DeviceContext* ctx) {
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_beam_score = static_cast<T*>(beam_score);
    T* typed_hyps_beam_score = static_cast<T*>(hyps_beam_score);
    cuda::BeamScoreInitLauncher(typed_beam_score, typed_hyps_beam_score,
                                hyps_beam_idx, eos_count, batch_size, beam_size,
                                cu_stream);
  };
  DispatchCUDA(dtype, functor);
}
AsStatus gen_process_logits_gpu(DataType dtype, int64_t* in_tokens,
                                void* in_logits, int batch_size, int vocab_size,
                                const DeviceContext* ctx,
                                RuntimeContext* runtime_ctx,
                                BatchGencfg batch_gencfg, void* ws_ptr,
                                size_t ws_bytes) {
  const cudaStream_t gpu_stream =
      static_cast<const CUDAContext*>(ctx)->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_in_logits = static_cast<T*>(in_logits);
    // int cur_len = gen_ctx->step + gen_ctx->in_length_bias;
    // the input and output of LogitsProcessor are typed_in_logits, it
    // perform inplace process
    cuda::LogitsProcessor(typed_in_logits, in_tokens, batch_size,
                          ctx->GetModelMaxLength(), vocab_size, batch_gencfg,
                          ws_ptr, ws_bytes, gpu_stream);
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus gen_sample_gpu(DataType dtype, int64_t* out_tokens, void* topk_value,
                        void* topp_value, int* topk_indice, void* in_logits,
                        void* sample_states, int batch_size, int max_k,
                        int length, int* k_arr, float* p_arr,
                        float* temperature_arr, const DeviceContext* ctx,
                        RuntimeContext* runtime_ctx, void* ws_ptr,
                        size_t ws_bytes, void* device_prop) {
  if (max_k > length) {
    LOG(ERROR) << "Top-k: max_k should not exceed length" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  if (p_arr == nullptr) {
    LOG(ERROR) << "Top-p: p_arr should not be nullptr" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  if (temperature_arr == nullptr) {
    LOG(ERROR) << "Top-p: temperature_arr should not be nullptr" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  std::vector<void*> sample_states_vec(batch_size);
  bool use_torch_sample = ctx->GetUseTorchSample();
  if (use_torch_sample) {
    for (int i = 0; i < batch_size; i++) {
      GenerateContext* gen_ctx = runtime_ctx->is_context
                                     ? runtime_ctx->GetContextGenCtx()
                                     : runtime_ctx->GetGenCtx(i);
      sample_states_vec[i] = gen_ctx->sample_state->GetDataPtr();
    }
  }

  auto functor = [&]<typename T>() {
    cudaStream_t gpu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
    T* typed_topk_value = static_cast<T*>(topk_value);
    T* typed_topp_value = static_cast<T*>(topp_value);
    T* typed_in_logits = static_cast<T*>(in_logits);
    const T* topp_input_logits = typed_in_logits;
    if (max_k == 1) {
      // greedy
      cuda::TopKKernelLauncher(typed_topk_value, topk_indice, typed_in_logits,
                               batch_size, length, max_k, gpu_stream);
      if (!use_torch_sample) {
        cuda::SampleKernelLauncher<T>(out_tokens, sample_states,
                                      typed_topk_value, topk_indice, batch_size,
                                      k_arr, max_k, gpu_stream, device_prop);
      } else {
        cuda::SampleTorchKernelLauncher<T>(
            out_tokens, sample_states_vec, typed_topk_value, topk_indice,
            batch_size, k_arr, max_k, gpu_stream, device_prop);
      }
    } else {
      // step 1: top-k
      int topp_length = length;
      bool topp_input_is_sorted = false;
      if (max_k != length) {
#ifdef CONFIG_ENABLE_RADIX_TOPK
        /* radix-based top-k */
        cuda::TopKRadixKernelLauncher(typed_topk_value, topk_indice,
                                      typed_in_logits, ws_ptr, batch_size,
                                      length, max_k, gpu_stream);
#else
        /* heap-based top-k */
        cuda::TopKKernelLauncher(typed_topk_value, topk_indice, typed_in_logits,
                                 batch_size, length, max_k, gpu_stream);
#endif
        topp_input_logits = typed_topk_value;
        topp_length = max_k;
        topp_input_is_sorted = true;
      }
      // step 2: top-p & softmax to generate sampling probability
      DLOG(INFO) << "top-p with max_k=" << max_k << std::endl;
      hiednnCudaHandle_t hiednn_handle =
          static_cast<const CUDAContext*>(ctx)->GetHiednnHandle();
      cuda::TopPSoftmaxLauncher(
          k_arr, typed_topk_value, topk_indice, topp_input_logits, p_arr,
          temperature_arr, typed_topp_value, ws_ptr, ws_bytes, batch_size,
          topp_length, topp_input_is_sorted, hiednn_handle, gpu_stream);

      // step 3: sample token
      if (!use_torch_sample) {
        cuda::SampleKernelLauncher<T>(
            out_tokens, sample_states, typed_topk_value, topk_indice,
            batch_size, k_arr, topp_length, gpu_stream, device_prop);
      } else {
        cuda::SampleTorchKernelLauncher<T>(
            out_tokens, sample_states_vec, typed_topk_value, topk_indice,
            batch_size, k_arr, topp_length, gpu_stream, device_prop);
      }
    }
    // TODO: for later Batch GenOp:
    // update each gen_ctx.sample_state by sample_states
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
void gen_torch_sample_init_gpu(void* sample_states, unsigned long long seed,
                               int batch_size, const DeviceContext* ctx) {
  const cudaStream_t gpu_stream =
      static_cast<const CUDAContext*>(ctx)->GetStream();
  cuda::SampleTorchKernelInitLauncher(sample_states, seed, batch_size,
                                      gpu_stream);
}
void gen_sample_init_gpu(void* sample_states, unsigned long long seed,
                         int batch_size, const DeviceContext* ctx) {
  const cudaStream_t gpu_stream =
      static_cast<const CUDAContext*>(ctx)->GetStream();
  cuda::SampleKernelInitLauncher(sample_states, seed, batch_size, gpu_stream);
}
AsStatus NcclBcast(std::shared_ptr<AsTensor> tensor,
                   const CUDAContext* cuda_ctx) {
  void* out = tensor->GetDataPtr();
  int count = tensor->GetShape().Count();
  DataType dtype = tensor->GetDataType();
  ncclDataType_t nccl_dtype = GetNcclType(dtype);
  AS_CHECK_NCCL(ncclBcast(out, count, nccl_dtype, 0, cuda_ctx->GetNCCLComm(),
                          cuda_ctx->GetStream()));
  return AsStatus::ALLSPARK_SUCCESS;
}
}  // namespace allspark
