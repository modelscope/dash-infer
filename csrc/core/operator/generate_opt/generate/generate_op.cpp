/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_op.cpp
 */

#include "generate_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <algorithm>
#include <limits>
#include <random>

#include "generate_impl_cpu.hpp"
/**
 * @brief Define this macro to constrain max_k to 1024, and forbid max_k == 0
 * (only influences SampleOp).
 */
namespace allspark {
AsStatus GenerateOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      copy_matrix = copy_matrix_cpu;
      sample_init_launcher = gen_sample_init_cpu;
      kernel_launcher = gen_sample_cpu;
      process_logits_launcher = gen_process_logits_cpu;
      beam_init_launcher = gen_beam_init_cpu;
      logprobs_launcher = logprobs_cpu;
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
  topk_value_ =
      std::make_unique<AsTensor>("topk_value", backend, dtype_, DataMode::DENSE,
                                 Shape{max_batch, default_k_});
  topp_value_ =
      std::make_unique<AsTensor>("topp_value", backend, dtype_, DataMode::DENSE,
                                 Shape{max_batch, default_k_});
  topk_indice_ =
      std::make_unique<AsTensor>("topk_indices", backend, DataType::INT64,
                                 DataMode::DENSE, Shape{max_batch, default_k_});
  topk_list_ = std::make_unique<AsTensor>("topk_list", backend, DataType::INT32,
                                          DataMode::DENSE, Shape{max_batch});
  topp_list_ =
      std::make_unique<AsTensor>("topp_list", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch});
  temperature_list_ =
      std::make_unique<AsTensor>("temperature_list", backend, DataType::FLOAT32,
                                 DataMode::DENSE, Shape{max_batch});
  if (out_names_.size() > 3) {
    tensor_map_->at(out_names_[3])->SetDataType(dtype_);
  }
  last_data_ = std::make_unique<AsTensor>("last_data", backend, dtype_,
                                          DataMode::DENSE, Shape{0});
  bad_words_ids_ =
      std::make_unique<AsTensor>("bad_words_ids", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{1024, 1024});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GenerateOp::Reshape(RuntimeContext* runtime_ctx) {
  const int max_batch = ctx_->GetModelMaxBatch();
  switch (runtime_ctx->generate_method) {
    case 0: {
      DeviceType backend = ctx_->GetDeviceType();
      DLOG(INFO) << "SampleOp::Reshape()" << std::endl;
      const Shape& in_shape = tensor_map_->at(in_names_[0])->GetShape();
      batch_size_ = in_shape[0];
      seq_len_ = in_shape[1];
      vocab_size_ = in_shape[2];
      beam_size_ = 1;
      max_k_ = 0;
      std::vector<int> topk_vec(batch_size_);
      std::vector<float> topp_vec(batch_size_);
      std::vector<float> temperatures(batch_size_);
      for (int i = 0; i < batch_size_; i++) {
        GenerateContext* gen_ctx;
        if (runtime_ctx->is_context) {
          gen_ctx = runtime_ctx->GetContextGenCtx();
        } else {
          gen_ctx = runtime_ctx->GetGenCtx(i);
        }
        int real_k =
            gen_ctx->gen_cfg.top_k == 0 ? vocab_size_ : gen_ctx->gen_cfg.top_k;
        max_k_ = std::max(max_k_, real_k);
        topk_vec[i] = real_k;
        topp_vec[i] = gen_ctx->gen_cfg.top_p;
        temperatures[i] = gen_ctx->gen_cfg.temperature;
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
      AS_CHECK_STATUS(
          last_data_->SetShape(Shape{max_batch, beam_size_ * vocab_size_}));
      AS_CHECK_STATUS(
          last_data_->SetShape(Shape{batch_size_, beam_size_ * vocab_size_}));
      /* max_k_ == vocab_size_ means no top-k step, but topk_value_ and
       * topk_indice_ should still reshape to the top-p output shape,
       * i.e., {batch, length} */
      const bool run_topk = max_k_ != vocab_size_;
      const int topp_length = run_topk ? max_k_ : vocab_size_;
      // warm up with max length
      AS_CHECK_STATUS(logprobs_->SetShape(Shape{max_batch, vocab_size_}));
      AS_CHECK_STATUS(topk_indice_->SetShape(Shape{max_batch, vocab_size_}));
      AS_CHECK_STATUS(topk_value_->SetShape(Shape{max_batch, vocab_size_}));
      AS_CHECK_STATUS(topp_value_->SetShape(Shape{max_batch, vocab_size_}));
      AS_CHECK_STATUS(topk_indice_->SetShape(Shape{max_batch, topp_length}));
      AS_CHECK_STATUS(topk_value_->SetShape(Shape{max_batch, topp_length}));
      AS_CHECK_STATUS(topp_value_->SetShape(Shape{max_batch, topp_length}));
      // warm up with max batch
      AS_CHECK_STATUS(
          tensor_map_->at(out_names_[0])->SetShape(Shape{max_batch, 1}));
      AS_CHECK_STATUS(
          tensor_map_->at(out_names_[0])->SetShape(Shape{batch_size_, 1}));
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
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  char* in_ptr = (char*)in_tensor->GetDataPtr() +
                 (seq_len_ - 1) * vocab_size_ * SizeofType(dtype_);
  void* out_ptr = out_tensor->GetDataPtr();
  if (seq_len_ > 1) {
    copy_matrix(dtype_, in_ptr, last_data_->GetDataPtr(), batch_size_,
                vocab_size_, seq_len_ * vocab_size_, vocab_size_, ctx_);
    in_ptr = (char*)last_data_->GetDataPtr();
  }
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  size_t ws_bytes = tensor_map_->at("workspace")->GetSizeInByte();
  void* sample_states =
      runtime_ctx->is_context
          ? runtime_ctx->GetContextGenCtx()->sample_state->GetDataPtr()
          : nullptr;

  bool need_logprobs = false;
  if (runtime_ctx->is_context) {
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    sample_init_launcher(sample_states, gen_ctx->gen_cfg.seed, 1, ctx_);
    int64_t* max_dec_ids =
        static_cast<int64_t*>(tensor_map_->at(in_names_[1])->GetDataPtr()) +
        gen_ctx->current_batch * ctx_->GetModelMaxLength();
    char* batch_in_ptr = in_ptr;
    process_logits_launcher(dtype_, max_dec_ids, batch_in_ptr, 1, vocab_size_,
                            ctx_, gen_ctx, bad_words_ids_, ws_ptr, ws_bytes);
    if (gen_ctx->gen_cfg.logprobs == true) {
      need_logprobs = true;
    }
  } else {
    for (int i = 0; i < batch_size_; i++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
      int64_t* max_dec_ids =
          static_cast<int64_t*>(tensor_map_->at(in_names_[1])->GetDataPtr()) +
          gen_ctx->current_batch * ctx_->GetModelMaxLength();
      char* batch_in_ptr = in_ptr + i * vocab_size_ * SizeofType(dtype_);
      process_logits_launcher(dtype_, max_dec_ids, batch_in_ptr, 1, vocab_size_,
                              ctx_, gen_ctx, bad_words_ids_, ws_ptr, ws_bytes);
      if (gen_ctx->gen_cfg.logprobs == true) {
        need_logprobs = true;
      }
    }
  }
  if (need_logprobs) {
    logprobs_launcher(
        dtype_, in_ptr, logprobs_->GetDataPtr(), topk_value_->GetDataPtr(),
        static_cast<int64_t*>(topk_indice_->GetDataPtr()), batch_size_,
        vocab_size_, runtime_ctx, ws_ptr, ws_bytes, ctx_);
  }
  kernel_launcher(dtype_, static_cast<int64_t*>(out_ptr),
                  topk_value_->GetDataPtr(), topp_value_->GetDataPtr(),
                  static_cast<int64_t*>(topk_indice_->GetDataPtr()), in_ptr,
                  sample_states, batch_size_, max_k_, vocab_size_,
                  static_cast<int*>(topk_list_->GetDataPtr()),
                  static_cast<float*>(topp_list_->GetDataPtr()),
                  static_cast<float*>(temperature_list_->GetDataPtr()), ctx_,
                  runtime_ctx, ws_ptr, ws_bytes);
  if (device == DeviceType::CPU) {
    if (nrank_ > 1) {
      AsStatus status = MpiBcast(tensor_map_->at(out_names_[0]));

      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in sampling " << std::endl;
        return status;
      }
    }
  }

  DLOG(INFO) << "SampleOp::Forward() returns" << std::endl;
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

REGISTER_OP("GenerateOp", CPU, GenerateOp)
}  // namespace allspark
