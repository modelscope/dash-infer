/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    calc_expert_op.cpp
 */

#include "calc_expert_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {
AsStatus CalcExpertOp::Init(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_experts") == attr_map.end()) {
    LOG(ERROR) << "CalcExpertOp : can't find num_expert attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_expert_ = *(float*)(attr_map.at("num_experts").c_str());
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus CalcExpertOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  total_token_ = out_shape[0] * out_shape[1];
  hidden_size_ = out_shape[2];
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus CalcExpertOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* expert_weight_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(out_tensor->GetDataPtr());
        T* typed_in = static_cast<T*>(in_tensor->GetDataPtr());
        T* typed_expert_weight =
            static_cast<T*>(expert_weight_tensor->GetDataPtr());
        cuda::CalcExpertKernelLauncher(typed_out, typed_in, typed_expert_weight,
                                       total_token_, hidden_size_, num_expert_,
                                       gpu_ctx->GetStream());
      };
      DispatchCUDA(in_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      // auto functor = [&]<typename T>() {
      //   T* typed_out = static_cast<T*>(y_tensor->GetDataPtr());
      //   const T* typed_in = static_cast<const T*>(x_tensor->GetDataPtr());
      //   cpu::MulKernelLauncher(typed_out, typed_in,
      //                          x_tensor->GetShape().Count(), alpha_);
      // };
      // DispatchCPU(x_tensor->GetDataType(), functor);
    }
    default:
      break;
  }
  // PrintInformation();
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(CalcExpert, CUDA, CalcExpertOp)
REGISTER_OP(CalcExpert, CPU, CalcExpertOp)
}  // namespace allspark
