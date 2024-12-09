/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_glu_op.cpp
 */

#include "unary_glu_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;

namespace allspark {
AsStatus UnaryGLUOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("unary_type") == attr_map.end()) {
    LOG(ERROR) << "UnaryGLUOp : can't find unary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  unary_type_ = *(UnaryType*)(attr_map.at("unary_type").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      dnnl_op_ctx_ = std::make_unique<DNNLOpContext>();
      auto& algo_map = DNNLOpContext::unary_algo_map_;
      if (algo_map.find(unary_type_) == algo_map.end()) {
        LOG(ERROR) << "Unsupported unary type:" << UnaryType_Name(unary_type_)
                   << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      dnnl_op_ctx_->algo_ = algo_map[unary_type_];
      dnnl_op_ctx_->pr_fwd_.resize(1);
      dnnl_op_ctx_->ins_.resize(1);
      dnnl_op_ctx_->outs_.resize(1);
      break;
    }
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      break;
    }
#endif
    default:
      LOG(ERROR) << "UnaryGLU Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UnaryGLUOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  size_t double_n = in_shape[-1];
  if (double_n % 2 != 0) {
    LOG(ERROR) << "UnaryGLU Operator input_size can't % 2 " << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  outer_size_ = in_shape.Count() / double_n;
  inner_size_ = double_n / 2;
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  out_shape[-1] = inner_size_;
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UnaryGLUOp::Forward() {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(out_names_[0]).get();
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(y_tensor->GetDataPtr());
        const T* typed_in = static_cast<const T*>(x_tensor->GetDataPtr());
        cuda::UnaryGLUKernelLauncher(typed_out, typed_in, outer_size_,
                                     inner_size_, unary_type_,
                                     gpu_ctx->GetStream());
      };
      DispatchCUDA(x_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      // TODO
      LOG(ERROR) << "UnaryGLU Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    default:
      LOG(ERROR) << "UnaryGLU Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(UnaryGLU, CUDA, UnaryGLUOp)
REGISTER_OP(UnaryGLU, CPU, UnaryGLUOp)
}  // namespace allspark
