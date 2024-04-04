/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chunk_op.cpp
 */

#include "chunk_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
namespace allspark {

void chunk_launcher(DataType dtype, void* out, void* in, int batch_size,
                    int seq_len, int hidden_size, int chunk_split,
                    const DeviceContext* ctx) {
  DeviceType backend = ctx->GetDeviceType();
  if (backend == DeviceType::CPU) {
    auto functor = [&]<typename T>() {
      cpu::ChunkKernelLauncher<T>((T*)out, (T*)in, batch_size, seq_len,
                                  hidden_size, chunk_split);
    };
    DispatchCPU(dtype, functor);
  }
}
AsStatus ChunkOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                       const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("fixed_training_split") != attr_map.end()) {
    chunk_split_ = *(int*)(attr_map.at("fixed_training_split").c_str());
  } else {
    chunk_split_ = 1;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus ChunkOp::Reshape() {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  hidden_size_ = y_shape[2];
  tensor_map_->at(out_names_[0])
      ->SetShape(Shape{batch_size_, seq_len_, hidden_size_ / 2});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus ChunkOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  chunk_launcher(dtype_, out_tensor->GetDataPtr(), in_tensor->GetDataPtr(),
                 batch_size_, seq_len_, hidden_size_, chunk_split_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("Chunk", CPU, ChunkOp)
}  // namespace allspark
