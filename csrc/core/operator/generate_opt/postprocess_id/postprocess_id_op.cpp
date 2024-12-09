/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    postprocess_id_op.cpp
 */

#include "postprocess_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

AsStatus PostProcessIdOp::Init(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  output_host_ = std::make_unique<AsTensor>(
      "output_host", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
      Shape{ctx.GetModelMaxBatch(), ctx.GetModelMaxLength()});
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PostProcessIdOp::Reshape(RuntimeContext* runtime_ctx) {
  if (in_names_.size() > 1) {
    switch (runtime_ctx->generate_method) {
      case 0: {
        in_name_ = in_names_[0];
        break;
      }
      case 1: {
        in_name_ = in_names_[1];
        break;
      }
    }
  } else {
    in_name_ = in_names_[0];
  }

  Shape in_shape = tensor_map_->at(in_name_)->GetShape();
  batch_size_ = runtime_ctx->GetGenCtxListSize();
  in_stride_ = 1 * ctx_->GetModelMaxLength();
  int num_return_sequences = 1;
  out_stride_ = num_return_sequences * ctx_->GetModelMaxLength();
  Shape out_shape =
      Shape({batch_size_ * num_return_sequences, ctx_->GetModelMaxLength()});
  // warm up with max batch
  tensor_map_->at(out_names_[0])
      ->SetShape(Shape({ctx_->GetModelMaxBatch() * num_return_sequences,
                        ctx_->GetModelMaxLength()}));
  tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape));
  output_host_->SetShape(
      Shape({batch_size_ * num_return_sequences, ctx_->GetModelMaxLength()}));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus PostProcessIdOp::RunOneBatch(GenerateContext* gen_ctx,
                                      int current_batch) {
  DeviceType backend = ctx_->GetDeviceType();
  Request* request = gen_ctx->request.get();

  if (rank_ == 0) {
    // 直接写入request的outputs,CPU
    std::string name = "generated_ids";
    std::shared_ptr<AsTensor> out_tensor = request->outputs[name];
    out_tensor->SetShape(Shape{1, gen_ctx->step + gen_ctx->in_length_bias + 1});
    memcpy(out_tensor->GetDataPtr(),
           (int64_t*)output_host_->GetDataPtr() + current_batch * out_stride_,
           (gen_ctx->step + gen_ctx->in_length_bias + 1) * sizeof(int64_t));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus PostProcessIdOp::RunContext(RuntimeContext* runtime_ctx) {
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  DLOG(INFO) << "PostProcessIdOp::RunContext [" << gen_ctx->request->request_id
             << "]";
  RunOneBatch(gen_ctx, runtime_ctx->current_batch);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PostProcessIdOp::RunDecoder(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "PostProcessIdOp::RunDecoder: batch size=" << batch_size_;
  for (int batch = 0; batch < batch_size_; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    RunOneBatch(gen_ctx, batch);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
void UpdateProbs(GenerateContext* gen_ctx, RuntimeContext* runtime_ctx,
                 int current_batch, int batch_stride) {
  if (gen_ctx->gen_cfg.logprobs) {
    std::vector<std::pair<int, float>> log_probs;
    for (int index = 0; index < gen_ctx->gen_cfg.top_logprobs; index++) {
      log_probs.push_back(std::make_pair(
          runtime_ctx
              ->logprobs_indice_host[current_batch * batch_stride + index],
          runtime_ctx
              ->logprobs_value_host[current_batch * batch_stride + index]));
      DLOG(INFO)
          << "indice = "
          << runtime_ctx
                 ->logprobs_indice_host[current_batch * batch_stride + index]
          << ",value ="
          << runtime_ctx
                 ->logprobs_value_host[current_batch * batch_stride + index]
          << std::endl;
    }
    gen_ctx->request->log_probs_list.push_back(log_probs);
    gen_ctx->request->token_logprobs_list.push_back(
        runtime_ctx->token_logprobs_host[current_batch]);
  }
}
AsStatus PostProcessIdOp::Forward(RuntimeContext* runtime_ctx) {
  int64_t* in_ids =
      static_cast<int64_t*>(tensor_map_->at(in_name_)->GetDataPtr());
  int64_t* out_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[0])->GetDataPtr());
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* cu_ctx = static_cast<const CUDAContext*>(ctx_);
      rank_ = cu_ctx->GetRank();
      cuda::PostProcessId(out_ids, in_ids, batch_size_, in_stride_, out_stride_,
                          cu_ctx->GetStream());
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      cpu::PostProcessId(out_ids, in_ids, batch_size_, in_stride_, out_stride_);
      rank_ = cpu_ctx->GetRank();
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  TensorUtils::DeepCopyWholeAsync(*output_host_,
                                  *tensor_map_->at(out_names_[0]), ctx_);
  ctx_->Synchronize();
  int batch_stride = ctx_->GetMaxTopLogprobs();
  if (rank_ == 0) {
    if (runtime_ctx->is_context) {
      UpdateProbs(runtime_ctx->GetContextGenCtx(), runtime_ctx, 0,
                  batch_stride);
    } else {
      for (int i = 0; i < batch_size_; i++) {
        GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
        UpdateProbs(gen_ctx, runtime_ctx, i, batch_stride);
      }
    }
  }
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP(PostProcessId, CUDA, PostProcessIdOp)
REGISTER_OP(PostProcessId, CPU, PostProcessIdOp)
}  // namespace allspark
