/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embeddingT5_op.cpp
 */

#include "embeddingT5_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <string>
#include <utility>
namespace allspark {
AsStatus cpu_embedding(DataType dtype, void* out, void* in_ids,
                       const void* embedding_table, int batch_size, int seq_len,
                       int hidden_size, int vocab_size,
                       const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_embedding" << std::endl;
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const int64_t* typed_in_ids = static_cast<const int64_t*>(in_ids);
    const T* typed_embedding_table = static_cast<const T*>(embedding_table);
    cpu::EmbeddingT5KernelLauncher(typed_out, typed_in_ids,
                                   typed_embedding_table, batch_size, seq_len,
                                   hidden_size, vocab_size, false);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus EmbeddingT5Op::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  if (weights_.size() != 1) {
    LOG(ERROR) << "EmbeddingT5Op has 1 weights [word_embedding_table], "
                  "[pos_embedding_table], [token_embedding_table](optional)"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  hidden_size_ = weights_[0]->GetShape()[1];
  for (int i = 1; i < weights_.size(); ++i) {
    if (weights_[i]->GetShape()[1] != hidden_size_) {
      LOG(ERROR) << "EmbeddingT5Op : Invalid weight shape." << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  vocab_size_ = weights_[0]->GetShape()[0];
  // type inference
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // kernel choose
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_embedding;
      break;
    default:
      LOG(ERROR) << "Embedding Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus EmbeddingT5Op::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  seq_len_ = in_shape[1];
  Shape out_shape = Shape({batch_size_, seq_len_, hidden_size_});
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus EmbeddingT5Op::Forward() {
  void* in_ids = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  kernel_launcher(weights_[0]->GetDataType(), out, in_ids,
                  weights_[0]->GetDataPtr(), batch_size_, seq_len_,
                  hidden_size_, vocab_size_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("EmbeddingT5", CPU, EmbeddingT5Op)
}  // namespace allspark
