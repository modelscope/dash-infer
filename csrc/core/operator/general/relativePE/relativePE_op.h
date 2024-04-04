/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    relativePE_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class RelativePEOp : public AsOperator {
 public:
  explicit RelativePEOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_length_(1),
        k_(1),
        max_seq_(0),
        is_decoder_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  AsStatus (*kernel_launcher)(DataType dtype, void* out,
                              const void* attention_bias, int batch,
                              int seq_len, int k, int step, bool is_decoder,
                              const DeviceContext* ctx) = nullptr;
  int batch_size_;
  int seq_length_;
  int k_;
  int max_seq_;
  int is_decoder_;
};

}  // namespace allspark
