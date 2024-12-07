/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chunk_binary_op.cpp
 */

#include "chunk_binary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {

void chunk_binary_launcher(DataType dtype, void* out, void* in, int batch_size,
                           int seq_len, int hidden_size, int chunk_split,
                           BinaryType type, const DeviceContext* ctx) {
  DeviceType backend = ctx->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      auto functor = [&]<typename T>() {
        cuda::ChunkBinary<T>((T*)out, (T*)in, batch_size, seq_len, hidden_size,
                             chunk_split, type, cu_stream);
      };
      DispatchCUDA(dtype, functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      auto functor = [&]<typename T>() {
        cpu::ChunkBinary<T>((T*)out, (T*)in, batch_size, seq_len, hidden_size,
                            chunk_split, type);
      };
      DispatchCPU(dtype, functor);
      break;
    }
    default:
      LOG(ERROR) << "ChunkBinary Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return;
  }
}
AsStatus ChunkBinaryOp::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("fixed_training_split") != attr_map.end()) {
    chunk_split_ = *(int*)(attr_map.at("fixed_training_split").c_str());
  } else {
    chunk_split_ = 1;  // 实际训练用的卡数，如果不指定则为1。
  }
  if (attr_map.find("binary_type") == attr_map.end()) {
    LOG(ERROR) << "BinaryOp : can't find binary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  binary_type_ = *(BinaryType*)(attr_map.at("binary_type").c_str());
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus ChunkBinaryOp::Reshape() {
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
AsStatus ChunkBinaryOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  chunk_binary_launcher(dtype_, out_tensor->GetDataPtr(),
                        in_tensor->GetDataPtr(), batch_size_, seq_len_,
                        hidden_size_, chunk_split_, binary_type_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(ChunkBinary, CUDA, ChunkBinaryOp)
REGISTER_OP(ChunkBinary, CPU, ChunkBinaryOp)
}  // namespace allspark
