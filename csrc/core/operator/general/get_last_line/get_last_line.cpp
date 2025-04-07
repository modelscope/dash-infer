/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    get_last_line.cpp
 */

#include "get_last_line.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;

namespace allspark {

AsStatus GetLastLineOp::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GetLastLineOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_ = in_shape[0];
  seq_len_ = in_shape[1];
  hidden_size_ = in_shape[2];
  AS_CHECK_STATUS(tensor_map_->at(out_names_[0])
                      ->SetShape(Shape({ctx_->GetModelMaxBatch(), 1,
                                        hidden_size_})));  // WARMUP
  AS_CHECK_STATUS(tensor_map_->at(out_names_[0])
                      ->SetShape(Shape({batch_, 1, hidden_size_})));
  return AsStatus::ALLSPARK_SUCCESS;
}
void GetLastLineOp::UpdateHiddenStates(RuntimeContext* runtime_ctx,
                                       AsTensor* out_tensor) {
  if (rank_info_.rank_id == 0) {
    std::string tensor_name = "hidden_states";
    if (runtime_ctx->is_context) {
      GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
      if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
        std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
            tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
            DataMode::DENSE, Shape({seq_len_, hidden_size_}));
        int data_size = SizeofType(output_tensor->GetDataType());
        CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                 out_tensor->GetDataPtr(), out_tensor->GetDeviceType(),
                 seq_len_ * hidden_size_ * data_size, ctx_);
        gen_ctx->request->tensors_from_model_inference_list[tensor_name]
            .push_back(output_tensor);
      }
    } else {
      int batch_size = runtime_ctx->GetGenCtxListSize();
      for (int i = 0; i < batch_size; i++) {
        GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
        if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
          std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
              tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
              DataMode::DENSE, Shape({seq_len_, hidden_size_}));
          int data_size = SizeofType(output_tensor->GetDataType());
          CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                   out_tensor->GetDataPtr() + i * hidden_size_ * data_size,
                   out_tensor->GetDeviceType(), hidden_size_ * data_size, ctx_);
          gen_ctx->request->tensors_from_model_inference_list[tensor_name]
              .push_back(output_tensor);
        }
      }
    }
  }
}

AsStatus GetLastLineOp::Forward(RuntimeContext* runtime_ctx) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  out_tensor->CopyDataFrom(
      (char*)in_tensor->GetDataPtr() +
          (seq_len_ - 1) * hidden_size_ * SizeofType(in_tensor->GetDataType()),
      batch_ * 1 * hidden_size_ * SizeofType(in_tensor->GetDataType()),
      ctx_->GetDeviceType(), ctx_);
  UpdateHiddenStates(runtime_ctx, in_tensor);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(GetLastLine, CUDA, GetLastLineOp)
REGISTER_OP(GetLastLine, CPU, GetLastLineOp)
}  // namespace allspark
