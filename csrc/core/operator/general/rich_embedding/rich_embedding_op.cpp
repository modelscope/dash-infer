/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rich_embedding_op.cpp
 */

#include "rich_embedding_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <string>
#include <utility>
namespace allspark {

AsStatus RichEmbeddingOp::Init(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);
  input_ids_host_ = std::make_unique<AsTensor>(
      "input_ids_host_", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
      Shape{1 * ctx_->GetModelMaxLength()}, flags);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RichEmbeddingOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[1])->GetShape();
  batch_size_ = in_shape[0];
  seq_len_ = in_shape[1];
  hidden_size_ = in_shape[2];
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RichEmbeddingOp::Forward(RuntimeContext* runtime_ctx) {
  if (!runtime_ctx->is_context) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  TensorListMap extra_embedding = gen_ctx->request->extra_embedding;
  if (extra_embedding.empty()) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (batch_size_ != 1) {
    LOG(ERROR) << "RichEmbeddingOp only support single batch in context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  void* in_ids = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  DataType dtype = tensor_map_->at(in_names_[1])->GetDataType();
  int word_size = SizeofType(dtype);
  int64_t* input_ids_host = (int64_t*)input_ids_host_->GetDataPtr();
  DeviceType backend = ctx_->GetDeviceType();
  input_ids_host_->SetShape(Shape{seq_len_});
  input_ids_host_->CopyDataFrom(in_ids, seq_len_ * sizeof(int64_t),
                                ctx_->GetDeviceType(), ctx_);
  ctx_->Synchronize();
  std::map<int64_t, int> offset_map;
  offset_map.clear();
  int pos = 0;
  for (; pos < seq_len_; pos++) {
    std::string token_id_str = std::to_string(input_ids_host[pos]);
    if (extra_embedding.count(token_id_str) > 0) {
      int length = 1;
      int start_pos = pos;
      int64_t now_token = input_ids_host[pos];
      while (pos + 1 < seq_len_ && input_ids_host[pos + 1] == now_token) {
        length++;
        pos++;
      }
      if (offset_map[now_token]) {
        offset_map[now_token] = offset_map[now_token] + 1;
      } else {
        offset_map[now_token] = 1;
      }
      if (offset_map[now_token] > extra_embedding[token_id_str].size()) {
        LOG(ERROR) << "Embedding relpace num error,input_num= "
                   << offset_map[now_token] << ",embedding_num = "
                   << extra_embedding[token_id_str].size() << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }

      AsTensor* embedding =
          (extra_embedding[token_id_str])[offset_map[now_token] - 1].get();
      if (length != embedding->GetShape()[0]) {
        LOG(ERROR) << "Embedding relpace length error input_len=" << length
                   << ",embedding_len = " << embedding->GetShape()[0]
                   << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      switch (backend) {
        case DeviceType::CPU: {
          if (dtype != DataType::FLOAT32) {
            LOG(ERROR) << "CPU only support FLOAT32";
            return AsStatus::ALLSPARK_RUNTIME_ERROR;
          }
          memcpy((char*)out + start_pos * hidden_size_ * word_size,
                 (char*)embedding->GetDataPtr(),
                 length * hidden_size_ * word_size);
          break;
        }
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("RichEmbedding", CPU, RichEmbeddingOp)
}  // namespace allspark
