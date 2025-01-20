/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_embedding_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class DecOptEmbeddingOp : public AsOperator {
 public:
  using AsOperator::AsOperator;
  explicit DecOptEmbeddingOp(const std::string& op_type = "")
      : AsOperator(op_type),
        hidden_size_(768),
        batch_size_(1),
        seq_len_(1),
        offset_(0) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  // AsStatus Reshape() override;
  // AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunOneBatch(std::shared_ptr<GenerateContext> gen_ctx,
                       int current_batch);

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int hidden_size_;
  int batch_size_;
  int seq_len_;
  int offset_;
  int vocab_size_;
  AsStatus (*kernel_launcher)(DataType dtype, void* out, void* in_ids,
                              void* token_type_ids, const void* embedding_table,
                              const void* pos_table,
                              const void* token_type_table, int batch_size,
                              int step, int seq_len, int hidden_size,
                              int vocab_size, int* offset, int force_offset,
                              const DeviceContext* ctx) = nullptr;
};

class DecOptLearnedPositionalEmbeddingOp : public DecOptEmbeddingOp {
 public:
  explicit DecOptLearnedPositionalEmbeddingOp(const std::string& op_type = "")
      : DecOptEmbeddingOp(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  int offset_;
};

}  // namespace allspark
