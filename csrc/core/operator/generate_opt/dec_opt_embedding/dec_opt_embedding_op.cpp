/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_embedding_op.cpp
 */

#include "dec_opt_embedding_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <utility>

namespace allspark {
AsStatus cpu_dec_opt_embedding(DataType dtype, void* out, void* in_ids,
                               void* token_type_ids,
                               const void* embedding_table,
                               const void* pos_table,
                               const void* token_type_table, int batch_size,
                               int step, int seq_len, int hidden_size,
                               int vocab_size, int* offset, int force_offset,
                               const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_dec_opt_embedding dtype: " << (int)dtype
             << " step: " << step << " seq_len: " << seq_len << std::endl;
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const int64_t* typed_in_ids = static_cast<const int64_t*>(in_ids);
    const int64_t* typed_token_ids =
        static_cast<const int64_t*>(token_type_ids);
    const T* typed_embedding_table = static_cast<const T*>(embedding_table);
    const T* typed_pos_table = static_cast<const T*>(pos_table);
    const T* typed_token_type_table = static_cast<const T*>(token_type_table);
    if (step == 0 && seq_len != 1) {
      cpu::EmbeddingKernelLauncher(
          typed_out, typed_in_ids, typed_token_ids, typed_embedding_table,
          typed_pos_table, typed_token_type_table, batch_size, seq_len,
          hidden_size, vocab_size, offset, force_offset, false);
    } else {
      cpu::EmbeddingKernelLauncher(
          typed_out, typed_in_ids, typed_token_ids, typed_embedding_table,
          typed_pos_table, typed_token_type_table, batch_size, step,
          hidden_size, vocab_size, offset, force_offset, true);
    }
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus DecOptEmbeddingOp::Init(const OperatorProto& op_proto,
                                 const DeviceContext& ctx,
                                 const TensorMap& weights_map,
                                 TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  auto& attr_map = op_proto.attr();
  if (attr_map.find("offset") != attr_map.end()) {
    offset_ = *(int*)(attr_map.at("offset").c_str());
  }
  if (weights_.size() != 2 && weights_.size() != 3) {
    LOG(ERROR) << "DecOptEmbeddingOp has 2-3 weights [word_embedding_table], "
                  "[pos_embedding_table], [token_embedding_table](optional)"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  dtype_ = weights_[0]->GetDataType();
  hidden_size_ = weights_[0]->GetShape()[1];
  vocab_size_ = weights_[0]->GetShape()[0];
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // kernel choose
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU:
      kernel_launcher = cpu_dec_opt_embedding;
      break;
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptEmbeddingOp::Reshape(RuntimeContext* runtime_ctx) {
  const Shape& in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  seq_len_ = in_shape[1];
  Shape out_shape = Shape({batch_size_, seq_len_, hidden_size_});
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptEmbeddingOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR)
        << "DecOptEmbeddingOp only support multibatch in decoder pharse, "
           "not context pharse."
        << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  RunOneBatch(gen_ctx, 0);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptEmbeddingOp::RunOneBatch(GenerateContext* gen_ctx,
                                        int current_batch) {
  void* in_ids = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* token_type_ids = nullptr;
  void* token_embedding_weights = nullptr;
  if (weights_.size() == 3) {
    LOG(ERROR) << "Not support token_type_ids";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  in_ids = (char*)in_ids + current_batch * 1 * SizeofType(DataType::INT64);
  out = (char*)out + current_batch * hidden_size_ * SizeofType(dtype_);
  int* offset = nullptr;
  kernel_launcher(weights_[0]->GetDataType(), out, in_ids, token_type_ids,
                  weights_[0]->GetDataPtr(), weights_[1]->GetDataPtr(),
                  token_embedding_weights, 1, gen_ctx->step, seq_len_,
                  hidden_size_, vocab_size_, (int*)offset, offset_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptEmbeddingOp::RunDecoder(RuntimeContext* runtime_ctx) {
  for (int batch = 0; batch < batch_size_; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    RunOneBatch(gen_ctx, batch);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptEmbeddingOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP("DecOptEmbedding", CPU, DecOptEmbeddingOp)
}  // namespace allspark
