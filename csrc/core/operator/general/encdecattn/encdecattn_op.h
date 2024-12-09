/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    encdecattn_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/** @brief Cross Attention，设计时需要考虑到beam search引起的broadcast的问题
 *    input :
 *       > query: [batch_size * beam_size, 1, hidden_size]
 *       > enc_kv: [batch_size, enc_seq_len, hidden_size * 2]
 *       > enc_mask: [batch_size, enc_seq_len, enc_seq_len]
 *    output:
 *       > cross_attn_out: [batch_size * beam_size, 1, hidden_size]
 */
class EncdecAttentionOp : public AsOperator {
 public:
  explicit EncdecAttentionOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        hidden_size_(768),
        enc_seq_len_(0),
        beam_size_(1),
        num_heads_(16),
        size_per_head_(64),
        gemm_batch_(1),
        score_size_(0),
        alpha_(-1.0f) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType dtype_;
  int batch_size_;
  int hidden_size_;
  int enc_seq_len_;
  int beam_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t score_size_;
  float alpha_;
};
}  // namespace allspark
