/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_op.cpp
 */

#include "unary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;

namespace allspark {
AsStatus UnaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                       const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("unary_type") == attr_map.end()) {
    LOG(ERROR) << "UnaryOp : can't find unary_type attribute." << std::endl;
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
      LOG(ERROR) << "Unary Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UnaryOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  if (ctx_->GetDeviceType() == DeviceType::CPU) {
    auto eng = DNNLEngine::GetInstance().GetEngine();
    memory::desc data_desc({out_shape.Count()}, memory::data_type::f32,
                           memory::format_tag::x);
    dnnl_op_ctx_->ins_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->outs_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->pr_fwd_[0] = std::make_unique<dnnl::eltwise_forward>(
        dnnl::eltwise_forward::primitive_desc{
            eng, dnnl::prop_kind::forward_inference, dnnl_op_ctx_->algo_,
            data_desc, data_desc, 0.f, 0.f});
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UnaryOp::Forward() {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(out_names_[0]).get();
  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(y_tensor->GetDataPtr());
        const T* typed_in = static_cast<const T*>(x_tensor->GetDataPtr());
        cuda::UnaryKernelLauncher(typed_out, typed_in,
                                  x_tensor->GetShape().Count(), unary_type_,
                                  gpu_ctx->GetStream());
      };
      DispatchCUDA(x_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      dnnl::memory& in_mem = *(dnnl_op_ctx_->ins_[0]);
      dnnl::memory& out_mem = *(dnnl_op_ctx_->outs_[0]);
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      in_mem.set_data_handle(x_tensor->GetDataPtr());
      out_mem.set_data_handle(y_tensor->GetDataPtr());
      std::unordered_map<int, memory> args{{DNNL_ARG_SRC, in_mem},
                                           {DNNL_ARG_DST, out_mem}};
      dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args);
      break;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Unary, CUDA, UnaryOp)
REGISTER_OP(Unary, CPU, UnaryOp)
}  // namespace allspark
