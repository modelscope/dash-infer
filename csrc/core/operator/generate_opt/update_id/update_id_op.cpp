/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    update_id_op.cpp
 */

#include "update_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_update_id(int64_t* max_dec_ids, const int64_t* dec_ids,
                       const int* beam_idx, int64_t* tmp_id, int batch_size,
                       int beam_size, int max_length, int* step_list,
                       int seq_len, const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cuda::UpdateId(max_dec_ids, dec_ids, beam_idx, tmp_id, batch_size, beam_size,
                 step_list, max_length, seq_len, gpu_ctx->GetStream());
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif
AsStatus cpu_update_id(int64_t* max_dec_ids, const int64_t* dec_ids,
                       const int* beam_idx, int64_t* tmp_id, int batch_size,
                       int beam_size, int max_length, int* step_list,
                       int seq_len, const DeviceContext* ctx) {
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

UpdateIdOp::~UpdateIdOp() {
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    AS_CHECK_CUDA(cudaFreeHost(tmp_step_host_));
  }
#endif
}

AsStatus UpdateIdOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DeviceType backend = ctx.GetDeviceType();
  tmp_id_ = std::make_unique<AsTensor>("tmp_id", backend, INT64,
                                       DataMode::DENSE, Shape{0});
  tmp_step_device_ =
      std::make_unique<AsTensor>("tmp_step", backend, INT32, DataMode::DENSE,
                                 Shape{ctx.GetModelMaxBatch()});

  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA:
      kernel_launcher = gpu_update_id;
      AS_CHECK_CUDA(cudaMallocHost(&tmp_step_host_,
                                   sizeof(int) * ctx.GetModelMaxBatch(),
                                   cudaHostAllocDefault));
      break;
#endif
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
  dim_t ws_size = ctx_->GetModelMaxLength() * sizeof(int64_t);
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
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

  int64_t* dec_ids =
      static_cast<int64_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
  int64_t* max_dec_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[0])->GetDataPtr()) +
      runtime_ctx->current_batch * gen_ctx->engine_max_length;

  // clang-format off
  /*
  * |-------------------------------------------------------------------------------|
  * |                 |      prefix_len == 0       |      prefix_len > 0            |
  * |-------------------------------------------------------------------------------|
  * | update_id_first | step = 0                   | step = prefix_len              |
  * |                 | in_length_bias = 0         | in_length_bias = 0             |
  * |                 | run_step = 0               | run_step = 0                   |
  * |                 | seq_len = input_len        | seq_len = new_input_len        |
  * |-------------------------------------------------------------------------------|
  * | update_id       | step = 0                   | step = prefix_len              |
  * |                 | in_length_bias = input_len | in_length_bias = new_input_len |
  * |                 | run_step = input_len       | run_step = input_len           |
  * |                 | seq_len = 1                | seq_len = 1                    |
  * |-------------------------------------------------------------------------------|
  */
  // clang-format on

  int run_step = gen_ctx->in_length_bias == 0
                     ? 0
                     : gen_ctx->step + gen_ctx->in_length_bias;
  int engine_max_length = ctx_->GetModelMaxLength();
  std::vector<int> run_step_list(tmp_step_device_->GetShape()[0], 0);
  run_step_list[0] = run_step;

  if (gen_ctx->prefix_len > 0 && gen_ctx->in_length_bias == 0) {
    // the input tensor in the tensor_map_ is incomplete
    // copy the original input tensor from the request
    dec_ids = static_cast<int64_t*>(tensor_map_->at("workspace")->GetDataPtr());
    TensorUtils::DeepCopyWholeTolerantAsync(
        *tensor_map_->at("workspace"), *gen_ctx->request->inputs["input_ids"],
        ctx_);
  }

  tmp_step_device_->CopyDataFrom(run_step_list.data(),
                                 sizeof(int) * run_step_list.size(),
                                 DeviceType::CPU, ctx_);

  if (run_step == 0) {
    kernel_launcher(max_dec_ids, dec_ids, beam_idx, tmp_id, 1, beam_size_,
                    engine_max_length, (int*)tmp_step_device_->GetDataPtr(),
                    gen_ctx->prefix_len + seq_len_, ctx_);
  } else {
    kernel_launcher(max_dec_ids, dec_ids, beam_idx, tmp_id, 1, beam_size_,
                    engine_max_length, (int*)tmp_step_device_->GetDataPtr(),
                    seq_len_, ctx_);
  }

  std::vector<int64_t> out_host(engine_max_length);
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      AS_CHECK_CUDA(cudaMemcpyAsync(
          out_host.data(), max_dec_ids, engine_max_length * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          static_cast<const CUDAContext*>(ctx_)->GetStream()));
      ctx_->Synchronize();
      break;
    }
#endif
    case DeviceType::CPU: {
      memcpy(out_host.data(), max_dec_ids, engine_max_length * sizeof(int64_t));
      break;
    }
    default: {
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }

  if (run_step == 0) {
    ;  // finish flag should not be generated in the "update_id_first" step
  } else {
    if (gen_ctx->step + gen_ctx->in_length_bias >=
        gen_ctx->gen_cfg.max_length - 1) {
      gen_ctx->finish = true;
    }
    if (gen_ctx->gen_cfg.early_stopping) {
      if (out_host[gen_ctx->step + gen_ctx->in_length_bias] ==
          (int64_t)gen_ctx->gen_cfg.eos_token_id) {
        gen_ctx->finish = true;
      }
    }
    if (gen_ctx->generate_method == 0 &&
        (!gen_ctx->gen_cfg.stop_words_ids.empty())) {
      const auto generated_len =
          gen_ctx->step + gen_ctx->in_length_bias + seq_len_;
      auto* gen_over = gen_ctx->gen_over;
      if (check_finish(1, generated_len, gen_ctx->engine_max_length,
                       (int64_t*)out_host.data(), gen_over,
                       gen_ctx->gen_cfg.stop_words_ids)) {
        gen_ctx->finish = true;
      }
    }
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
  int* run_step_list_ptr = run_step_list.data();
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    // Use PINNED memory on cuda.
    run_step_list_ptr = tmp_step_host_;
  }
#endif

  for (int i = 0; i < batch_size_; i++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
    run_step_list_ptr[i] = gen_ctx->step;
  }
  int engine_max_length = ctx_->GetModelMaxLength();
  tmp_step_device_->CopyDataFrom(run_step_list_ptr, batch_size_ * sizeof(int),
                                 DeviceType::CPU, ctx_);
  kernel_launcher(max_dec_ids, dec_ids, beam_idx, tmp_id, batch_size_,
                  beam_size_, engine_max_length,
                  (int*)tmp_step_device_->GetDataPtr(), seq_len_, ctx_);
  std::vector<int64_t> out_host(batch_size_ * engine_max_length);
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      AS_CHECK_CUDA(
          cudaMemcpyAsync(out_host.data(), max_dec_ids,
                          batch_size_ * engine_max_length * sizeof(int64_t),
                          cudaMemcpyDeviceToHost,
                          static_cast<const CUDAContext*>(ctx_)->GetStream()));
      ctx_->Synchronize();
      break;
    }
#endif  // ENABLE_CUDA
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
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP(UpdateId, CUDA, UpdateIdOp)
REGISTER_OP(UpdateId, CPU, UpdateIdOp)
}  // namespace allspark
