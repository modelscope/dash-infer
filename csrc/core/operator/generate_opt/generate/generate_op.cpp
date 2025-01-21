/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_op.cpp
 */

#include "generate_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <algorithm>
#include <limits>
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <curand_kernel.h>

#include <cuda/nccl_utils.hpp>

#include "generate_impl_gpu.hpp"
#endif
#include <cpu/cpu_context.h>

#include <random>

#include "generate_impl_cpu.hpp"

#ifdef ENABLE_JSON_MODE
#include <utility/format_enforcer.h>
#endif

/**
 * @brief Define this macro to constrain max_k to 1024, and forbid max_k == 0
 * (only influences SampleOp).
 */
namespace allspark {
#ifdef ENABLE_JSON_MODE
AsStatus GenerateOp::FormatModelOutput(std::shared_ptr<GenerateContext> gen_ctx,
                                       char* in_ptr, int current_batch,
                                       bool is_context) {
  std::shared_ptr<util::FormatEnforcer> formatter = gen_ctx->format_enforcer;
  if (is_context == false) {
    int64_t* generated_ids = static_cast<int64_t*>(
        gen_ctx->request->interim.at("generated_ids")->GetDataPtr());
    // get last generated token
    // gen_ctx->step is already increased after 'decoder' graph, so use step-1
    // here
    formatter->gen_sequence.emplace_back(generated_ids[gen_ctx->step - 1]);
  }
  FrozenTokenVector& allowed_tokens =
      formatter->token_enforcer_->get_allowed_tokens(formatter->gen_sequence);
  if (allowed_tokens.empty()) {
    LOG(ERROR) << "LMFE: Parser reached state with no allowed tokens, "
                  "generated sequence:\n"
               << formatter->tokenizer_data_->decode(formatter->gen_sequence);
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  void* batch_in_ptr =
      (void*)(in_ptr + current_batch * vocab_size_ * SizeofType(dtype_));
  switch (dtype_) {
    case DataType::FLOAT32:
      AS_CHECK_STATUS(util::FormatEnforcer::process_logits<float>(
          allowed_tokens, static_cast<float*>(batch_in_ptr), ctx_,
          SizeofType(dtype_), vocab_size_));
      break;
    case DataType::FLOAT16:
      AS_CHECK_STATUS(util::FormatEnforcer::process_logits<half>(
          allowed_tokens, static_cast<half*>(batch_in_ptr), ctx_,
          SizeofType(dtype_), vocab_size_));
      break;
    case DataType::BFLOAT16:
      AS_CHECK_STATUS(util::FormatEnforcer::process_logits<hie::bfloat16>(
          allowed_tokens, static_cast<hie::bfloat16*>(batch_in_ptr), ctx_,
          SizeofType(dtype_), vocab_size_));
      break;
    default:
      LOG(ERROR)
          << "GenerateOp::FormatModelOutput got unsupported data type!\n";
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif  // ENABLE_JSON_MODE

GenerateOp::~GenerateOp() {
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    AS_CHECK_CUDA(cudaFreeHost(dec_ids_host_));
  }
#endif
}

AsStatus GenerateOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      copy_matrix = copy_matrix_gpu;
      sample_init_launcher = gen_sample_init_gpu;
      kernel_launcher = gen_sample_gpu;
      fill_max_dec_ids_launcher = fill_max_dec_ids_gpu;
      process_logits_launcher = gen_process_logits_gpu;
      beam_init_launcher = gen_beam_init_gpu;
      logprobs_launcher = logprobs_gpu;
      device_prop_ = std::make_unique<AsTensor>(
          "device_property", DeviceType::CPU, DataType::INT8, DataMode::DENSE,
          Shape{sizeof(cudaDeviceProp)});
      cudaGetDeviceProperties((cudaDeviceProp*)device_prop_->GetDataPtr(),
                              rank_info_.rank_id);

      const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx_);
      rank_id_ = cuda_ctx->GetRank();
      nrank_ = cuda_ctx->GetNranks();
      break;
    }
#endif
    case DeviceType::CPU: {
      copy_matrix = copy_matrix_cpu;
      sample_init_launcher = gen_sample_init_cpu;
      kernel_launcher = gen_sample_cpu;
      fill_max_dec_ids_launcher = fill_max_dec_ids_cpu;
      process_logits_launcher = gen_process_logits_cpu;
      beam_init_launcher = gen_beam_init_cpu;
      logprobs_launcher = logprobs_cpu;

      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      rank_id_ = cpu_ctx->GetRank();
      nrank_ = cpu_ctx->GetNranks();
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  const int max_batch = ctx_->GetModelMaxBatch();
  logprobs_ =
      std::make_unique<AsTensor>("logprobs", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch, default_k_});
  token_logprobs_ =
      std::make_unique<AsTensor>("token_logprobs", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch, 1});
  topk_list_ = std::make_unique<AsTensor>("topk_list", backend, DataType::INT32,
                                          DataMode::DENSE, Shape{max_batch});
  topp_list_ =
      std::make_unique<AsTensor>("topp_list", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch});
  temperature_list_ =
      std::make_unique<AsTensor>("temperature_list", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch});

  dec_ids_ = std::make_shared<AsTensor>("dec_ids", backend, DataType::INT64,
                                        DataMode::DENSE, Shape{max_batch});
  max_dec_ids_ = std::make_shared<AsTensor>(
      "max_dec_ids", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_batch, static_cast<long>(ctx_->GetModelMaxLength())});
  gen_ids_ptr_ =
      std::make_shared<AsTensor>("gen_ids_ptr", backend, DataType::POINTER,
                                 DataMode::DENSE, Shape{max_batch});

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (ctx_->GetUseTorchSample()) {
      sample_states_ = std::make_unique<AsTensor>(
          "sample_states", DeviceType::CPU, DataType::POINTER, DataMode::DENSE,
          Shape{max_batch});
    } else {
      sample_states_ = std::make_unique<AsTensor>(
          "sample_states", DeviceType::CUDA, DataType::POINTER, DataMode::DENSE,
          Shape{max_batch});
    }

    AS_CHECK_CUDA(cudaMallocHost(&dec_ids_host_, sizeof(int64_t) * max_batch,
                                 cudaHostAllocDefault));
  }
#endif

  if (out_names_.size() > 3) {
    tensor_map_->at(out_names_[3])->SetDataType(dtype_);
  }
  presence_penalty_list = std::make_unique<AsTensor>(
      "presence_penalty_list", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{max_batch});
  repetition_penalty_list = std::make_unique<AsTensor>(
      "repetition_penalty_list", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{max_batch});
  frequency_penalty_list = std::make_unique<AsTensor>(
      "frequency_penalty_list", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{max_batch});
  no_repeat_ngram_size_list = std::make_unique<AsTensor>(
      "no_repeat_ngram_size_list", backend, DataType::INT32, DataMode::DENSE,
      Shape{max_batch});
  min_length_list =
      std::make_unique<AsTensor>("min_length_list", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{max_batch});
  eos_token_id_list =
      std::make_unique<AsTensor>("eos_token_id_list", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{max_batch});
  cur_len_list =
      std::make_unique<AsTensor>("cur_len_list", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{max_batch});
  input_len_list =
      std::make_unique<AsTensor>("input_len_list", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{max_batch});
  suppress_repetition_in_generation_list = std::make_unique<AsTensor>(
      "suppress_repetition_in_generation_list", backend, DataType::INT32,
      DataMode::DENSE, Shape{max_batch});
  return AsStatus::ALLSPARK_SUCCESS;
}
void GenerateOp::build_batch_gencfg(RuntimeContext* runtime_ctx,
                                    BatchGencfg& batch_gencfg,
                                    const DeviceContext* ctx) {
  int batch_size = 0;
  if (runtime_ctx->is_context) {
    batch_size = 1;
  } else {
    batch_size = runtime_ctx->GetGenCtxListSize();
  }
  std::vector<float> host_repetition_penalty_list(batch_size);
  std::vector<float> host_presence_penalty_list(batch_size);
  std::vector<float> host_frequency_penalty_list(batch_size);
  std::vector<int> host_no_repeat_ngram_size_list(batch_size);
  std::vector<int> host_min_length_list(batch_size);
  std::vector<int> host_eos_token_id_list(batch_size);
  std::vector<int> host_cur_len_list(batch_size);
  std::vector<int> host_input_len_list(batch_size);
  std::vector<int> host_suppress_repetition_in_generation_list(batch_size);
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<GenerateContext> gen_ctx;
    if (runtime_ctx->is_context) {
      gen_ctx = runtime_ctx->GetContextGenCtx();
    } else {
      gen_ctx = runtime_ctx->GetGenCtx(i);
    }
    GenerateConfig gen_cfg = gen_ctx->gen_cfg;
    host_repetition_penalty_list[i] = gen_cfg.repetition_penalty;
    host_presence_penalty_list[i] = gen_cfg.presence_penalty;
    host_frequency_penalty_list[i] = gen_cfg.frequency_penalty;
    host_no_repeat_ngram_size_list[i] = gen_cfg.no_repeat_ngram_size;
    host_min_length_list[i] = gen_cfg.min_length;
    host_eos_token_id_list[i] = gen_cfg.eos_token_id;
    host_cur_len_list[i] = gen_ctx->step + gen_ctx->in_length_bias;
    host_input_len_list[i] = gen_ctx->input_len;
    host_suppress_repetition_in_generation_list[i] =
        gen_cfg.suppress_repetition_in_generation ? 1 : 0;
  }
  repetition_penalty_list->CopyDataFrom(host_repetition_penalty_list.data(),
                                        sizeof(float) * batch_size,
                                        DeviceType::CPU, ctx);
  presence_penalty_list->CopyDataFrom(host_presence_penalty_list.data(),
                                      sizeof(float) * batch_size,
                                      DeviceType::CPU, ctx);
  frequency_penalty_list->CopyDataFrom(host_frequency_penalty_list.data(),
                                       sizeof(float) * batch_size,
                                       DeviceType::CPU, ctx);
  no_repeat_ngram_size_list->CopyDataFrom(host_no_repeat_ngram_size_list.data(),
                                          sizeof(int) * batch_size,
                                          DeviceType::CPU, ctx);
  min_length_list->CopyDataFrom(host_min_length_list.data(),
                                sizeof(int) * batch_size, DeviceType::CPU, ctx);
  eos_token_id_list->CopyDataFrom(host_eos_token_id_list.data(),
                                  sizeof(int) * batch_size, DeviceType::CPU,
                                  ctx);
  cur_len_list->CopyDataFrom(host_cur_len_list.data(), sizeof(int) * batch_size,
                             DeviceType::CPU, ctx);
  input_len_list->CopyDataFrom(host_input_len_list.data(),
                               sizeof(int) * batch_size, DeviceType::CPU, ctx);
  suppress_repetition_in_generation_list->CopyDataFrom(
      host_suppress_repetition_in_generation_list.data(),
      sizeof(int) * batch_size, DeviceType::CPU, ctx);

  batch_gencfg.batch_size = batch_size;
  batch_gencfg.repetition_penalty_list = repetition_penalty_list->GetDataPtr();
  batch_gencfg.presence_penalty_list = presence_penalty_list->GetDataPtr();
  batch_gencfg.frequency_penalty_list = frequency_penalty_list->GetDataPtr();
  batch_gencfg.no_repeat_ngram_size_list =
      no_repeat_ngram_size_list->GetDataPtr();
  batch_gencfg.min_length_list = min_length_list->GetDataPtr();
  batch_gencfg.eos_token_id_list = eos_token_id_list->GetDataPtr();
  batch_gencfg.cur_len_list = cur_len_list->GetDataPtr();
  batch_gencfg.input_len_list = input_len_list->GetDataPtr();
  batch_gencfg.suppress_repetition_in_generation_list =
      suppress_repetition_in_generation_list->GetDataPtr();
}
AsStatus GenerateOp::Reshape(RuntimeContext* runtime_ctx) {
  const int max_batch = ctx_->GetModelMaxBatch();
  switch (runtime_ctx->generate_method) {
    case 0: {
      DeviceType backend = ctx_->GetDeviceType();
      const Shape& in_shape = tensor_map_->at(in_names_[0])->GetShape();
      batch_size_ = in_shape[0];
      seq_len_ = in_shape[1];
      vocab_size_ = in_shape[2];
      beam_size_ = 1;
      max_k_ = 0;
      need_logprobs_ = false;
      std::vector<int> topk_vec(batch_size_);
      std::vector<float> topp_vec(batch_size_);
      std::vector<float> temperatures(batch_size_);
      for (int i = 0; i < batch_size_; i++) {
        std::shared_ptr<GenerateContext> gen_ctx;
        if (runtime_ctx->is_context) {
          gen_ctx = runtime_ctx->GetContextGenCtx();
        } else {
          gen_ctx = runtime_ctx->GetGenCtx(i);
        }
        if (gen_ctx->gen_cfg.logprobs == true) {
          need_logprobs_ = true;
        }
        int real_k =
            gen_ctx->gen_cfg.top_k == 0 ? vocab_size_ : gen_ctx->gen_cfg.top_k;
        max_k_ = std::max(max_k_, real_k);
        topk_vec[i] = real_k;
        topp_vec[i] = gen_ctx->gen_cfg.top_p;
        temperatures[i] = gen_ctx->gen_cfg.temperature;
#ifdef ENABLE_JSON_MODE
#ifdef ENABLE_CUDA
        if (gen_ctx->gen_cfg.response_format.count("type") &&
            gen_ctx->gen_cfg.response_format["type"] == "json_object") {
          if (backend == DeviceType::CUDA) {
            const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx_);
            if (cuda_ctx->GetRank() == 0) {
              if (gen_ctx->format_enforcer->scores_buf_ == nullptr) {
                AS_CHECK_CUDA(cudaMallocHost(
                    (void**)&gen_ctx->format_enforcer->scores_buf_,
                    sizeof(float) * vocab_size_));
              }
            }
          }
        }
#endif  // ENABLE_CUDA
#endif  // ENABLE_JSON_MODE
      }

      // assert: 1 / min ~= 8.5e37 < max ~= 3.4e38
      if (*std::min_element(temperatures.begin(), temperatures.end()) <
          std::numeric_limits<float>::min()) {
        LOG(ERROR) << "GenerateOp: temperatures should be positive "
                      "normalized floats";
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      topk_list_->CopyDataFrom(topk_vec.data(), sizeof(int) * batch_size_,
                               DeviceType::CPU, ctx_);
      topp_list_->CopyDataFrom(topp_vec.data(), sizeof(float) * batch_size_,
                               DeviceType::CPU, ctx_);
      temperature_list_->CopyDataFrom(temperatures.data(),
                                      sizeof(float) * batch_size_,
                                      DeviceType::CPU, ctx_);

      build_batch_gencfg(runtime_ctx, batch_gencfg_, ctx_);

      AS_CHECK_STATUS(dec_ids_->SetShape(Shape{batch_size_, 1}));
      AS_CHECK_STATUS(gen_ids_ptr_->SetShape(Shape{batch_size_, 1}));

#ifdef CONFIG_SAMPLE_CONSTRAIN_MAX_K
      if (max_k_ == 0) {
        max_k_ = 1024;
      }
      if (max_k_ > 1024) {
        LOG(ERROR) << "gen_cfg.topk must < 1024" << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
#endif  // #ifdef CONFIG_SAMPLE_CONSTRAIN_MAX_K
        // warm up with max batch
      /* max_k_ == vocab_size_ means no top-k step, but topk_value_ and
       * topk_indice_ should still reshape to the top-p output shape,
       * i.e., {batch, length} */
      const bool run_topk = max_k_ != vocab_size_;
      const int topp_length = run_topk ? max_k_ : vocab_size_;
      // warm up with max length
      size_t total_ws_bytes = 0;

#ifdef ENABLE_CUDA
      if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_ENABLE_RADIX_TOPK
        // workspace: gpu radix top-k
        /// NOTE: with warm-up, we had better always set top-k workspace
        // if (max_k_ > 0) {
        size_t topk_ws_bytes;
        DispatchCUDA(dtype_, [&]<typename T>() {
          // set workspace with max batch max length
          cuda::TopKRadixGetWorkspaceSize<T>(&topk_ws_bytes, max_batch,
                                             vocab_size_);
        });
        if (topk_ws_bytes > std::numeric_limits<dim_t>::max()) {
          LOG(ERROR) << "radix top-k workspace size too large in "
                        "SampleOp::Reshape()";
          return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
        }
        total_ws_bytes = std::max(total_ws_bytes, topk_ws_bytes);
        // }
#endif  // CONFIG_ENABLE_RADIX_TOPK
        // workspace: gpu top-p
        size_t topp_ws_bytes;
        /// NOTE: with warm-up, we had better assume input is unsorted
        // whether top-k ensures order of top-p inputs
        constexpr bool topp_is_sorted = false;
        DispatchCUDA(dtype_, [&]<typename T>() {
          // set workspace with max batch max length
          cuda::TopPSoftmaxGetWorkspaceSize<T>(&topp_ws_bytes, max_batch,
                                               vocab_size_, topp_is_sorted);
        });

        if (topp_ws_bytes > std::numeric_limits<dim_t>::max()) {
          LOG(ERROR) << "top-p workspace size too large in "
                        "SampleOp::Reshape()";
          return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
        }
        total_ws_bytes = std::max(total_ws_bytes, topp_ws_bytes);
      }
#endif  // ENABLE_CUDA

      int64_t final_ws_bytes =
          total_ws_bytes +
          (int64_t)(max_batch * vocab_size_) *
              (SizeofType(dtype_) + SizeofType(DataType::INT32) +
               SizeofType(dtype_));
      AS_CHECK_STATUS(
          tensor_map_->at("workspace")
              ->SetShape(Shape{static_cast<dim_t>(final_ws_bytes)}));
      topk_value_ptr_ =
          (char*)tensor_map_->at("workspace")->GetDataPtr() + total_ws_bytes;
      topk_indice_ptr_ =
          (char*)topk_value_ptr_ + max_batch * vocab_size_ * SizeofType(dtype_);
      topp_value_ptr_ = (char*)topk_indice_ptr_ +
                        max_batch * vocab_size_ * SizeofType(DataType::INT32);
      // LOG(INFO) << "total_ws_bytes = " << total_ws_bytes;
      // alloc for logprobs_
      if (need_logprobs_) {
        AS_CHECK_STATUS(logprobs_->SetShape(Shape{max_batch, vocab_size_}));
        AS_CHECK_STATUS(token_logprobs_->SetShape(Shape{max_batch, 1}));
      }

      break;
    }
    default: {
      LOG(ERROR) << "GenerateOp::Reshape encounter bad generate method "
                 << runtime_ctx->generate_method << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GenerateOp::RunSample(RuntimeContext* runtime_ctx) {
  const DeviceType device = ctx_->GetDeviceType();
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  char* in_ptr = (char*)in_tensor->GetDataPtr() +
                 (seq_len_ - 1) * vocab_size_ * SizeofType(dtype_);
  void* out_ptr = dec_ids_->GetDataPtr();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  size_t ws_bytes = tensor_map_->at("workspace")->GetSizeInByte();
  void** sample_states = nullptr;

  void* device_prop_ptr = nullptr;
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    device_prop_ptr = device_prop_ ? device_prop_->GetDataPtr() : nullptr;
  }
#endif
  if (runtime_ctx->is_context) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
    sample_init_launcher(gen_ctx->sample_state->GetDataPtr(),
                         gen_ctx->gen_cfg.seed, 1, ctx_);
  }

#ifdef ENABLE_CUDA
  if (device == DeviceType::CUDA) {
    int batch_size = batch_size_;
    if (runtime_ctx->is_context) {
      batch_size = 1;
    }

    if (ctx_->GetUseTorchSample() == false) {
      std::vector<void*> sample_states_vec(batch_size);
      for (int i = 0; i < batch_size; i++) {
        std::shared_ptr<GenerateContext> gen_ctx = nullptr;
        if (runtime_ctx->is_context) {
          gen_ctx = runtime_ctx->GetContextGenCtx();
        } else {
          gen_ctx = runtime_ctx->GetGenCtx(i);
        }
        sample_states_vec[i] = gen_ctx->sample_state->GetDataPtr();
      }
      cudaStream_t stream = static_cast<const CUDAContext*>(ctx_)->GetStream();
      AS_CHECK_CUDA(cudaMemcpyAsync(sample_states_->GetDataPtr(),
                                    sample_states_vec.data(),
                                    sample_states_vec.size() * sizeof(void*),
                                    cudaMemcpyHostToDevice, stream));
    } else {
      for (int i = 0; i < batch_size; i++) {
        std::shared_ptr<GenerateContext> gen_ctx = nullptr;
        if (runtime_ctx->is_context) {
          gen_ctx = runtime_ctx->GetContextGenCtx();
        } else {
          gen_ctx = runtime_ctx->GetGenCtx(i);
        }
        ((void**)(sample_states_->GetDataPtr()))[i] =
            gen_ctx->sample_state->GetDataPtr();
      }
    }
    sample_states = (void**)sample_states_->GetDataPtr();
  }
#endif

  {
    fill_max_dec_ids_launcher(runtime_ctx, max_dec_ids_, ctx_);

    int batch_size;
    int64_t* max_dec_ids;
    char* batch_in_ptr = in_ptr;
    if (runtime_ctx->is_context) {
      batch_size = 1;
      max_dec_ids = static_cast<int64_t*>(max_dec_ids_->GetDataPtr()) +
                    runtime_ctx->current_batch * ctx_->GetModelMaxLength();
    } else {
      batch_size = batch_size_;
      max_dec_ids = static_cast<int64_t*>(max_dec_ids_->GetDataPtr());
    }

    process_logits_launcher(dtype_, max_dec_ids, batch_in_ptr, batch_size,
                            vocab_size_, ctx_, runtime_ctx, batch_gencfg_,
                            ws_ptr, ws_bytes);
  }

#ifdef ENABLE_JSON_MODE
  bool do_format = false;
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx->GetRank() == 0) {
      do_format = true;
    }
  } else {
    do_format = rank_id_ == 0;
  }
#else
  do_format = rank_id_ == 0;
#endif
  if (do_format == true) {
    if (runtime_ctx->is_context) {
      std::shared_ptr<GenerateContext> gen_ctx =
          runtime_ctx->GetContextGenCtx();
      if (gen_ctx->gen_cfg.response_format.count("type") &&
          gen_ctx->gen_cfg.response_format["type"] == "json_object") {
        AS_CHECK_STATUS(
            FormatModelOutput(gen_ctx, in_ptr, 0, runtime_ctx->is_context));
      }
    } else {
      for (int i = 0; i < batch_size_; i++) {
        std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(i);
        if (gen_ctx->gen_cfg.response_format.count("type") &&
            gen_ctx->gen_cfg.response_format["type"] == "json_object") {
          AS_CHECK_STATUS(
              FormatModelOutput(gen_ctx, in_ptr, i, runtime_ctx->is_context));
        }
      }
    }
  }
#endif  // ENABLE_JSON_MODE

  // don't remove this sync, otherwise wrong token will generate.
  ctx_->Synchronize();
  kernel_launcher(dtype_, static_cast<int64_t*>(out_ptr), topk_value_ptr_,
                  topp_value_ptr_, static_cast<int*>(topk_indice_ptr_), in_ptr,
                  sample_states, batch_size_, max_k_, vocab_size_,
                  static_cast<int*>(topk_list_->GetDataPtr()),
                  static_cast<float*>(topp_list_->GetDataPtr()),
                  static_cast<float*>(temperature_list_->GetDataPtr()), ctx_,
                  runtime_ctx, ws_ptr, ws_bytes, device_prop_ptr);

  if (need_logprobs_) {
    logprobs_launcher(dtype_, in_ptr, static_cast<int64_t*>(out_ptr),
                      token_logprobs_->GetDataPtr(), logprobs_->GetDataPtr(),
                      topk_value_ptr_, static_cast<int*>(topk_indice_ptr_),
                      batch_size_, vocab_size_, runtime_ctx, ws_ptr, ws_bytes,
                      ctx_);
  }

#ifdef ENABLE_CUDA
  if (device == DeviceType::CUDA) {
    const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx->GetNranks() > 1) {
      AsStatus status = NcclBcast(dec_ids_, cuda_ctx);
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in sampling " << std::endl;
        return status;
      }
      ctx_->Synchronize();
    }

    fill_generated_ids_gpu(runtime_ctx, dec_ids_, gen_ids_ptr_, dec_ids_host_,
                           cuda_ctx, rank_id_);
  }
#endif

  if (device == DeviceType::CPU) {
    if (nrank_ > 1) {
#ifdef ENABLE_MULTINUMA
      AsStatus status = MpiBcast(dec_ids_);

      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in sampling " << std::endl;
        return status;
      }
#else
      LOG(ERROR) << "Multi-NUMA codes are not compiled" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
#endif
    }

    fill_generated_ids_cpu(runtime_ctx, dec_ids_, rank_id_);
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GenerateOp::Forward(RuntimeContext* runtime_ctx) {
  switch (runtime_ctx->generate_method) {
    case 0: {
      return RunSample(runtime_ctx);
    }
    case 1: {
      LOG(ERROR) << "BeamSearch Not Support" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    default: {
      LOG(ERROR) << "GenerateOp::Forward encounter bad generate method "
                 << runtime_ctx->generate_method << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
REGISTER_OP(GenerateOp, CUDA, GenerateOp)
REGISTER_OP(GenerateOp, CPU, GenerateOp)
}  // namespace allspark
