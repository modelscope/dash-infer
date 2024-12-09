/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embeddingT5_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class EmbeddingT5Op : public AsOperator {
 public:
  using AsOperator::AsOperator;
  explicit EmbeddingT5Op(const std::string& op_type = "")
      : AsOperator(op_type), hidden_size_(768), batch_size_(1), seq_len_(1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  int hidden_size_;
  int batch_size_;
  int seq_len_;
  int vocab_size_;
  AsStatus (*kernel_launcher)(DataType dtype, void* out, void* in_ids,
                              const void* embedding_table, int batch_size,
                              int seq_len, int hidden_size, int vocab_size,
                              const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
