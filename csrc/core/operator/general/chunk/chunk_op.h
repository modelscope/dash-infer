/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chunk_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {
class ChunkOp : public AsOperator {
 public:
  explicit ChunkOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(768),
        num_heads_(16),
        size_per_head_(64),
        gemm_batch_(1),
        score_size_(0),
        alpha_(-1.0f),
        pos_embedding_(false),
        first_beam_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int batch_size_;
  int seq_len_;
  int hidden_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t score_size_;
  float alpha_;
  bool pos_embedding_;
  bool first_beam_;
  int chunk_split_ = 1;
};

}  // namespace allspark
