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
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PostProcessIdOp::Reshape(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

void UpdateProbs(std::shared_ptr<GenerateContext> gen_ctx,
                 RuntimeContext* runtime_ctx, int current_batch,
                 int batch_stride) {
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
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* cu_ctx = static_cast<const CUDAContext*>(ctx_);
      rank_ = cu_ctx->GetRank();
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      rank_ = cpu_ctx->GetRank();
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  int batch_stride = ctx_->GetMaxTopLogprobs();
  int batch_size = runtime_ctx->GetGenCtxListSize();
  if (rank_ == 0) {
    if (runtime_ctx->is_context) {
      UpdateProbs(runtime_ctx->GetContextGenCtx(), runtime_ctx, 0,
                  batch_stride);
    } else {
      for (int i = 0; i < batch_size; i++) {
        std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(i);
        UpdateProbs(gen_ctx, runtime_ctx, i, batch_stride);
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(PostProcessId, CUDA, PostProcessIdOp)
REGISTER_OP(PostProcessId, CPU, PostProcessIdOp)
}  // namespace allspark
