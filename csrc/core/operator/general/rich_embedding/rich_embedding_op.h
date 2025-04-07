/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rich_embedding_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {
// inputs[0]:input_ids[batch,seq_len]
// inputs[1]:embedding_out[batch,seq_len,hidden_size]
// output[0]:embedding_out[batch,seq_len,hidden_size]
class RichEmbeddingOp : public AsOperator {
 public:
  using AsOperator::AsOperator;
  explicit RichEmbeddingOp(const std::string& op_type = "")
      : AsOperator(op_type), hidden_size_(768), batch_size_(1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  void UpdateInputsEmbedding(RuntimeContext* runtime_ctx, AsTensor* out_tensor);

 private:
  int hidden_size_;
  int batch_size_;
  int seq_len_;
  std::unique_ptr<AsTensor> embedding_device_;
  std::unique_ptr<AsTensor> reply_part_;
};
}  // namespace allspark
