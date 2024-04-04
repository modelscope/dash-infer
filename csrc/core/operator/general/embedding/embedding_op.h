/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embedding_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class EmbeddingOp : public AsOperator {
 public:
  using AsOperator::AsOperator;
  explicit EmbeddingOp(const std::string& op_type = "")
      : AsOperator(op_type),
        hidden_size_(768),
        batch_size_(1),
        seq_len_(1),
        offset_(0) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  int hidden_size_;
  int batch_size_;
  int seq_len_;
  int offset_;
  int vocab_size_;
  AsStatus (*kernel_launcher)(DataType dtype, void* out, void* in_ids,
                              void* token_type_ids, const void* embedding_table,
                              const void* pos_table,
                              const void* token_type_table, int batch_size,
                              int seq_len, int hidden_size, int vocab_size,
                              int offset, const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
