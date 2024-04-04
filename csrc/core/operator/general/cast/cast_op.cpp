/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast_op.cpp
 */

#include "cast_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus cpu_cast(DataType src_dtype, DataType dst_dtype, const void* in,
                  void* out, int size, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_cast" << std::endl;
  auto functor = [&]<typename SrcType, typename DstType>() {
    const SrcType* typed_in = static_cast<const SrcType*>(in);
    DstType* typed_out = static_cast<DstType*>(out);
    cpu::CastKernelLauncher(typed_in, typed_out, size);
  };
  DispatchCast(src_dtype, dst_dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus CastOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                      const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  auto& attr_map = op_proto.attr();
  src_datatype_ = tensor_map_->at(in_names_[0])->GetDataType();
  dst_datatype_ = *(DataType*)(attr_map.at("dst_type").c_str());
  tensor_map_->at(out_names_[0])->SetDataType(dst_datatype_);
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_cast;
      break;
    default:
      LOG(ERROR) << "Cast Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus CastOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(in_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus CastOp::Forward() {
  void* in = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  int size = tensor_map_->at(in_names_[0])->GetShape().Count();
  kernel_launcher(src_datatype_, dst_datatype_, in, out, size, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("Cast", CPU, CastOp)
}  // namespace allspark
