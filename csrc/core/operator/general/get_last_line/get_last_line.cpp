/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    get_last_line.cpp
 */

#include "get_last_line.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
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
AsStatus GetLastLineOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_ = in_shape[0];
  seq_ = in_shape[1];
  hid_ = in_shape[2];
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])
          ->SetShape(Shape({ctx_->GetModelMaxBatch(), 1, hid_})));  // WARMUP
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(Shape({batch_, 1, hid_})));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GetLastLineOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  out_tensor->CopyDataFrom(
      (char*)in_tensor->GetDataPtr() +
          (seq_ - 1) * hid_ * SizeofType(in_tensor->GetDataType()),
      batch_ * 1 * hid_ * SizeofType(in_tensor->GetDataType()),
      ctx_->GetDeviceType(), ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("GetLastLine", CPU, GetLastLineOp)
}  // namespace allspark
