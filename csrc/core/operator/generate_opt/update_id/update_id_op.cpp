/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    update_id_op.cpp
 */

#include "update_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus cpu_update_id(int64_t* max_dec_ids, const int64_t* dec_ids,
                       const int* beam_idx, int64_t* tmp_id, int batch_size,
                       int beam_size, int max_length, int* step_list,
                       int seq_len, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_update_id" << std::endl;
  cpu::UpdateId(max_dec_ids, dec_ids, beam_idx, tmp_id, batch_size, beam_size,
                step_list, max_length, seq_len);
  return AsStatus::ALLSPARK_SUCCESS;
}

bool check_finish(const int batch_size, const int generated_len,
                  const int max_len, int64_t* out_host, bool* gen_over,
                  const std::vector<std::vector<int64_t>>& stop_words_ids) {
  bool finish = true;
  // only support batch_size = 1
  for (size_t i = 0; i < batch_size; i++) {
    const auto current_data = out_host + i * max_len;
    bool stop_matched = false;
    for (const auto& stop_word : stop_words_ids) {
      if (generated_len > stop_word.size()) {
        const auto word_start = current_data + generated_len - stop_word.size();
        const auto compare_size = stop_word.size() * sizeof(int64_t);

        if (memcmp(word_start, stop_word.data(), compare_size) == 0) {
          stop_matched = true;
          break;
        }
      }
    }
    gen_over[i] |= stop_matched;
    finish &= gen_over[i];
  }
  return finish;
}

AsStatus UpdateIdOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DeviceType backend = ctx.GetDeviceType();
  tmp_id_ = std::make_unique<AsTensor>("tmp_id", backend, INT64,
                                       DataMode::DENSE, Shape{0});
  tmp_step_ =
      std::make_unique<AsTensor>("tmp_step", backend, INT32, DataMode::DENSE,
                                 Shape{ctx.GetModelMaxBatch()});
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_update_id;
      break;
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UpdateIdOp::Reshape(RuntimeContext* runtime_ctx) {
  const Shape& in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  beam_size_ = 1;
  seq_len_ = in_shape[1];
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UpdateIdOp::RunContext(RuntimeContext* runtime_ctx) {
  int* beam_idx = nullptr;
  int64_t* tmp_id = nullptr;
  if (batch_size_ > 1) {
    LOG(ERROR) << " UpdateIdOp conext only support 1 batch" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  if (gen_ctx->finish) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (gen_ctx->step >= gen_ctx->gen_cfg.max_length - 1) {
    gen_ctx->finish = true;
  }
  int64_t* dec_ids =
      static_cast<int64_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
  int64_t* max_dec_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[0])->GetDataPtr()) +
      runtime_ctx->current_batch * gen_ctx->engine_max_length;
  int run_step = gen_ctx->step + gen_ctx->in_length_bias;
  tmp_step_->CopyDataFrom(&run_step, sizeof(int), DeviceType::CPU, ctx_);
  kernel_launcher(max_dec_ids, dec_ids, beam_idx, tmp_id, 1, beam_size_,
                  gen_ctx->engine_max_length, (int*)tmp_step_->GetDataPtr(),
                  seq_len_, ctx_);
  if (gen_ctx->generate_method == 0 &&
      (!gen_ctx->gen_cfg.stop_words_ids.empty())) {
    const auto num_element = gen_ctx->engine_max_length;
    std::vector<int64_t> out_host(num_element);

    switch (ctx_->GetDeviceType()) {
      case DeviceType::CPU: {
        memcpy(out_host.data(), max_dec_ids, num_element * sizeof(int64_t));
        break;
      }
      default:
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    const auto generated_len =
        gen_ctx->step + gen_ctx->in_length_bias + seq_len_;
    auto* gen_over = gen_ctx->gen_over;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UpdateIdOp::RunDecoder(RuntimeContext* runtime_ctx) {
  int* beam_idx = nullptr;
  int64_t* tmp_id = nullptr;
  int64_t* dec_ids =
      static_cast<int64_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
  int64_t* max_dec_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[0])->GetDataPtr());
  std::vector<int> run_step_list(batch_size_);
  for (int i = 0; i < batch_size_; i++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
    run_step_list[i] = gen_ctx->step;
  }
  int engine_max_length = ctx_->GetModelMaxLength();
  tmp_step_->CopyDataFrom(run_step_list.data(), batch_size_ * sizeof(int),
                          DeviceType::CPU, ctx_);
  kernel_launcher(max_dec_ids, dec_ids, beam_idx, tmp_id, batch_size_,
                  beam_size_, engine_max_length, (int*)tmp_step_->GetDataPtr(),
                  seq_len_, ctx_);
  std::vector<int64_t> out_host(batch_size_ * engine_max_length);
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      memcpy(out_host.data(), max_dec_ids,
             batch_size_ * engine_max_length * sizeof(int64_t));
      break;
    }
    default: {
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  for (int i = 0; i < batch_size_; i++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
    if (gen_ctx->finish) {
      continue;
    }
    if (gen_ctx->step >= gen_ctx->gen_cfg.max_length - 1) {
      gen_ctx->finish = true;
    }
    if (gen_ctx->gen_cfg.early_stopping) {
      if (out_host[i * engine_max_length + gen_ctx->step] ==
          (int64_t)gen_ctx->gen_cfg.eos_token_id) {
        gen_ctx->finish = true;
      }
    }
    if (gen_ctx->generate_method == 0 &&
        (!gen_ctx->gen_cfg.stop_words_ids.empty())) {
      const auto generated_len = gen_ctx->step + seq_len_;
      auto* gen_over = gen_ctx->gen_over;
      if (check_finish(1, generated_len, gen_ctx->engine_max_length,
                       (int64_t*)out_host.data() + i * engine_max_length,
                       gen_over, gen_ctx->gen_cfg.stop_words_ids)) {
        gen_ctx->finish = true;
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UpdateIdOp::Forward(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "UpdateIdOp::Forward()" << std::endl;
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP("UpdateId", CPU, UpdateIdOp)
}  // namespace allspark
