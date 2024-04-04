/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    layernorm_op.cpp
 */

#include "layernorm_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
namespace allspark {
AsStatus cpu_layernorm(DataType dtype, void* out, const void* input,
                       const void* bias, const void* gamma, const void* beta,
                       int m, int n, float eps, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_layernorm" << std::endl;
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(input);
    const T* typed_bias = static_cast<const T*>(bias);
    const T* typed_gamma = static_cast<const T*>(gamma);
    const T* typed_beta = static_cast<const T*>(beta);
    cpu::LayerNormKernel(typed_out, typed_input, typed_bias, typed_gamma,
                         typed_beta, m, n, eps);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus LayerNormOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  if (weights_.size() != 2) {
    LOG(ERROR) << "LayerNormOp has 2 weights [gamma], [beta]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (weights_[0]->GetShape() != weights_[1]->GetShape()) {
    LOG(ERROR) << "LayerNormOp : Invalid weight shape." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  hidden_size_ = weights_[0]->GetShape()[0];
  // type inference
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("eps") == attr_map.end()) {
    LOG(ERROR) << "LayerNormOp : can't find eps attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  eps_ = *(float*)(attr_map.at("eps").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_layernorm;
      break;
    default:
      LOG(ERROR) << "LayerNorm Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus LayerNormOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus LayerNormOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* bias = in_names_.size() == 2
                   ? tensor_map_->at(in_names_[1])->GetDataPtr()
                   : nullptr;
  int64_t m = in_tensor->GetShape().Count() / hidden_size_;
  kernel_launcher(in_tensor->GetDataType(), out, in, bias,
                  weights_[0]->GetDataPtr(), weights_[1]->GetDataPtr(), m,
                  hidden_size_, eps_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("LayerNorm", CPU, LayerNormOp)
}  // namespace allspark
