/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    relativePE_op.cpp
 */

#include "relativePE_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus cpu_relativePE(DataType dtype, void* out, const void* attention_bias,
                        int batch, int seq_len, int k, int step,
                        bool is_decoder, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_relativePE" << std::endl;
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* attention_bias_typed = static_cast<const T*>(attention_bias);
    cpu::RelativePEKernel(typed_out, attention_bias_typed, batch, seq_len, k,
                          step, is_decoder);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus RelativePEOp::Init(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  if (weights_.size() != 1) {
    LOG(ERROR) << "RelativePEOp has 1 weights [attention_bias]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  auto& attr_map = op_proto.attr();
  if (attr_map.find("max_seq") != attr_map.end()) {
    max_seq_ = *(int*)(attr_map.at("max_seq").c_str());
  }
  if (attr_map.find("is_decoder") != attr_map.end()) {
    is_decoder_ = *(bool*)(attr_map.at("is_decoder").c_str());
  }
  k_ = weights_[0]->GetShape()[1];
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_relativePE;
      break;
    default:
      LOG(ERROR) << "RelativePE Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus RelativePEOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  if (!is_decoder_) {
    seq_length_ = std::max((int)in_shape[1], max_seq_);

  } else {
    seq_length_ = gen_ctx_->max_length;
  }
  Shape out_shape = Shape{batch_size_, seq_length_, k_, seq_length_};
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus RelativePEOp::Forward() {
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  const int64_t* in =
      static_cast<const int64_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
  kernel_launcher(out_tensor->GetDataType(), out_tensor->GetDataPtr(),
                  weights_[0]->GetDataPtr(), batch_size_, seq_length_, k_,
                  (gen_ctx_->step + 1), is_decoder_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("RelativePE", CPU, RelativePEOp)
}  // namespace allspark
