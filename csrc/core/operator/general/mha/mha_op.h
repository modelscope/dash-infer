/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mha_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/*
 * inputs:
     > qkv_fuse: (batch, seq_len, hidden_size * 3)
     > mask: (batch, seq_len, step)
 * outputs:
     > attn_out: (batch, seq_len, hidden_size)
 */

class MultiHeadAttentionOp : public AsOperator {
 public:
  explicit MultiHeadAttentionOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        hidden_size_(768),
        seq_len_(128),
        num_heads_(16),
        size_per_head_(64),
        gemm_batch_(1),
        score_size_(0),
        alpha_(-1.0f),
        pos_embedding_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  int batch_size_;
  int hidden_size_;
  int seq_len_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t score_size_;
  float alpha_;
  bool pos_embedding_;
  AsStatus (*kernel_launcher)(DataType dtype, void* out, void* score,
                              const void* query, const void* key,
                              const void* value, const float* mask,
                              const void* position_embedding, void** q_array,
                              void** k_array, void** v_array,
                              void** score_array, void** out_array,
                              int batch_size, int seq_len, int hidden_size,
                              int num_heads, int size_per_head, int gemm_batch,
                              float alpha, const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
