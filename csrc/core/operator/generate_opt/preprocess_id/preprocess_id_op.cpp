/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    preprocess_id_op.cpp
 */

#include "preprocess_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

AsStatus PreProcessIdOp::Init(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  auto& attr_map = op_proto.attr();
  if (attr_map.find("start_id") != attr_map.end()) {
    start_id_ = *(int64_t*)(attr_map.at("start_id").c_str());
  }
  if (attr_map.find("num_beam") != attr_map.end()) {
    num_beam_ = *(int*)(attr_map.at("num_beam").c_str());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PreProcessIdOp::Reshape() {
  num_beam_ = 1;
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  seq_len_ = start_id_ == -1 ? in_shape[1] : 1;
  max_len_ = ctx_->GetModelMaxLength();
  tensor_map_->at(out_names_[0])
      ->SetShape(Shape{batch_size_ * num_beam_, seq_len_});
  tensor_map_->at(out_names_[1])
      ->SetShape(Shape{batch_size_ * num_beam_, max_len_});
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PreProcessIdOp::Forward() {
  int64_t* dec_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[0])->GetDataPtr());
  int64_t* max_dec_ids =
      static_cast<int64_t*>(tensor_map_->at(out_names_[1])->GetDataPtr());
  const int64_t* in_ids =
      static_cast<const int64_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      cudaStream_t cu_stream =
          static_cast<const CUDAContext*>(ctx_)->GetStream();
      // TensorUtils::Memset(*tensor_map_->at(out_names_[1]), 0);
      cuda::PreProcessForGeneration(dec_ids, max_dec_ids, in_ids, start_id_,
                                    batch_size_, num_beam_, max_len_, seq_len_,
                                    cu_stream);
      break;
    }
#endif
    case DeviceType::CPU:
      // TensorUtils::Memset(*tensor_map_->at(out_names_[1]), 0);
      cpu::PreProcessForGeneration(dec_ids, max_dec_ids, in_ids, start_id_,
                                   batch_size_, num_beam_, max_len_, seq_len_);
      break;
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(PreProcessId, CUDA, PreProcessIdOp)
REGISTER_OP(PreProcessId, CPU, PreProcessIdOp)
}  // namespace allspark
