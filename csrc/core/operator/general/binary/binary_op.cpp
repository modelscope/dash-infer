/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    binary_op.cpp
 */

#include "binary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
using dnnl::memory;

namespace allspark {
AsStatus BinaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                        const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("binary_type") == attr_map.end()) {
    LOG(ERROR) << "BinaryOp : can't find binary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  binary_type_ = *(BinaryType*)(attr_map.at("binary_type").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      dnnl_op_ctx_ = std::make_unique<DNNLOpContext>();
      auto& algo_map = DNNLOpContext::binary_algo_map_;
      if (algo_map.find(binary_type_) == algo_map.end()) {
        LOG(ERROR) << "Unsupported binary type:"
                   << BinaryType_Name(binary_type_) << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      dnnl_op_ctx_->algo_ = algo_map[binary_type_];
      if ((binary_type_ == BinaryType::GEGLU) ||
          (binary_type_ == BinaryType::SWIGLU)) {
        dnnl_op_ctx_->pr_fwd_.resize(2);
      } else {
        dnnl_op_ctx_->pr_fwd_.resize(1);
      }
      dnnl_op_ctx_->ins_.resize(2);
      dnnl_op_ctx_->outs_.resize(1);
      break;
    }
    default:
      LOG(ERROR) << "Binary Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BinaryOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  if (ctx_->GetDeviceType() == DeviceType::CPU) {
    auto eng = DNNLEngine::GetInstance().GetEngine();
    memory::desc data_desc({out_shape.Count()}, memory::data_type::f32,
                           memory::format_tag::x);
    dnnl_op_ctx_->ins_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->ins_[1] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->outs_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->pr_fwd_[0] =
        std::make_unique<dnnl::binary>(dnnl::binary::primitive_desc{
            eng, dnnl_op_ctx_->algo_, data_desc, data_desc, data_desc});

    if (binary_type_ == BinaryType::GEGLU) {
      // do gelu first
      dnnl_op_ctx_->pr_fwd_[1] = std::make_unique<dnnl::eltwise_forward>(
          dnnl::eltwise_forward::primitive_desc{
              eng, dnnl::prop_kind::forward_inference,
              dnnl::algorithm::eltwise_gelu_tanh, data_desc, data_desc, 0.f,
              0.f});
    }
    if (binary_type_ == BinaryType::SWIGLU) {
      // do gelu first
      dnnl_op_ctx_->pr_fwd_[1] = std::make_unique<dnnl::eltwise_forward>(
          dnnl::eltwise_forward::primitive_desc{
              eng, dnnl::prop_kind::forward_inference,
              dnnl::algorithm::eltwise_swish, data_desc, data_desc, 1.f, 0.f});
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BinaryOp::Forward() {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* z_tensor = tensor_map_->at(out_names_[0]).get();
  int64_t count = x_tensor->GetShape().Count();

  switch (ctx_->GetDeviceType()) {
    case DeviceType::CPU: {
      dnnl::memory& in0_mem = *(dnnl_op_ctx_->ins_[0]);
      dnnl::memory& in1_mem = *(dnnl_op_ctx_->ins_[1]);
      dnnl::memory& out_mem = *(dnnl_op_ctx_->outs_[0]);

      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      in0_mem.set_data_handle(x_tensor->GetDataPtr());
      in1_mem.set_data_handle(y_tensor->GetDataPtr());
      out_mem.set_data_handle(z_tensor->GetDataPtr());
      if ((binary_type_ == BinaryType::GEGLU ||
           binary_type_ == BinaryType::SWIGLU) &&
          dnnl_op_ctx_->pr_fwd_.size() > 1) {
        std::unordered_map<int, memory> args1{{DNNL_ARG_SRC_0, in1_mem},
                                              {DNNL_ARG_DST, in1_mem}};
        dnnl_op_ctx_->pr_fwd_[1]->execute(cpu_ctx->GetStream(), args1);

        std::unordered_map<int, memory> args2{{DNNL_ARG_SRC_0, in0_mem},
                                              {DNNL_ARG_SRC_1, in1_mem},
                                              {DNNL_ARG_DST, out_mem}};
        dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args2);
      } else {
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC_0, in0_mem},
                                             {DNNL_ARG_SRC_1, in1_mem},
                                             {DNNL_ARG_DST, out_mem}};
        dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args);
      }
      break;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
REGISTER_OP("Binary", CPU, BinaryOp)
}  // namespace allspark
