/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mul_op.cpp
 */

#include "mul_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
using dnnl::memory;

namespace allspark {
AsStatus MulOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                     const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("alpha") == attr_map.end()) {
    LOG(ERROR) << "MulOp : can't find alpha attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  alpha_ = *(float*)(attr_map.at("alpha").c_str());
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus MulOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus MulOp::Forward() {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(out_names_[0]).get();
  switch (ctx_->GetDeviceType()) {
    case DeviceType::CPU: {
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(y_tensor->GetDataPtr());
        const T* typed_in = static_cast<const T*>(x_tensor->GetDataPtr());
        cpu::MulKernelLauncher(typed_out, typed_in,
                               x_tensor->GetShape().Count(), alpha_);
      };
      DispatchCPU(x_tensor->GetDataType(), functor);
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("Mul", CPU, MulOp)
}  // namespace allspark
