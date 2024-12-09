/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    ALiBiPE_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class ALiBiPEOp : public AsOperator {
 public:
  explicit ALiBiPEOp(const std::string& op_type = "")
      : AsOperator(op_type), batch_size_(1), seq_length_(1), num_heads_(1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  AsStatus (*kernel_launcher)(DataType dtype, void* out, int* batch_offset,
                              int batch, int seq_len, int num_heads,
                              int ori_num_heads, int step,
                              const DeviceContext* ctx) = nullptr;
  int batch_size_;
  int seq_length_;
  int num_heads_;
  int ori_num_heads_;
  int max_seq_;
};

}  // namespace allspark
