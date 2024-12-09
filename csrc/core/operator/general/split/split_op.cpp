/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    split_op.cpp
 */

#include "split_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
#include <weight/weight_loader.h>

namespace allspark {
AsStatus SplitOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                       const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("split_type") == attr_map.end()) {
    LOG(ERROR) << "SplitOp : can't find split_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  split_type_ = *(SplitMode*)(attr_map.at("split_type").c_str());
  DLOG(INFO) << "split_type = " << split_type_ << std::endl;
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      rank_info_ = RankInfo(cpu_ctx->GetRank(), cpu_ctx->GetNranks());
      splitter_ =
          WeightSplitterFactory::GetSplitterByMode(split_type_, rank_info_);
      break;
    }
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      rank_info_ = RankInfo(gpu_ctx->GetRank(), gpu_ctx->GetNranks());
      splitter_ =
          WeightSplitterFactory::GetSplitterByMode(split_type_, rank_info_);
      break;
    }
#endif
    default:
      LOG(ERROR) << "Split Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus SplitOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  int batch = in_shape[0];
  int seq = in_shape[1];
  int k = in_shape[2];
  switch (split_type_) {
    case SplitMode::VSPLIT: {
      if (k % rank_info_.rank_size != 0) {
        LOG(ERROR) << "Split Operator k can't div ranks " << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      tensor_map_->at(out_names_[0])
          ->SetShape(Shape{batch, seq, k / rank_info_.rank_size});
      break;
    }
    default:
      LOG(ERROR) << "Split Operator does not support " << split_type_
                 << " split_type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus SplitOp::Forward() {
  ctx_->Synchronize();
  std::shared_ptr<AsTensor> in_tensor = tensor_map_->at(in_names_[0]);
  std::shared_ptr<AsTensor> out_tensor = tensor_map_->at(out_names_[0]);
  Shape in_shape = in_tensor->GetShape();
  Shape out_shape = out_tensor->GetShape();
  int batch = in_shape[0];
  int seq = in_shape[1];
  int k = in_shape[2];
  int out_k = out_shape[2];
  in_tensor->SetShape(Shape{batch * seq, k});
  out_tensor->SetShape(Shape{batch * seq, out_k});
  TensorInfo tensor_info = TensorInfo();
  tensor_info.shape = Shape{batch * seq, k};
  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      splitter_->CopyWeight(tensor_info, out_tensor, in_tensor, nullptr, 0);
      break;
    }
#endif
    case DeviceType::CPU: {
      splitter_->CopyWeight(tensor_info, out_tensor, in_tensor, nullptr, 0);
      break;
    }
    default:
      break;
  }
  in_tensor->SetShape(Shape{batch, seq, k});
  out_tensor->SetShape(Shape{batch, seq, out_k});
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Split, CUDA, SplitOp)
REGISTER_OP(Split, CPU, SplitOp)
}  // namespace allspark
