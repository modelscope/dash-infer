/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8.h
 */

#pragma once
#include <core/operator/operator.h>

#include <string>

namespace allspark {
class GemmA16W8Base : public AsOperator {
 public:
  explicit GemmA16W8Base(const std::string& op_type = "")
      : AsOperator(op_type),
        m_(0),
        n_(0),
        k_(0),
        batch_(1),
        transA_(false),
        transB_(false),
        lda_(0),
        ldb_(0),
        ldc_(0),
        alpha_(1.0f),
        beta_(1.0f),
        is_pooler_(false),
        activation_(UnaryType::UNARYTYPE_UNDEFINED) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(int yn);

 protected:
  DataType ftype_ = DATATYPE_UNDEFINED;
  DataType qtype_ = DATATYPE_UNDEFINED;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t batch_;
  int transA_;
  int transB_;
  int lda_;
  int ldb_;
  int ldc_;
  float alpha_;
  float beta_;
  bool is_pooler_;  // for bert pooler
  UnaryType activation_;

  int64_t group_size_ = -1;  // -1 mean per_channel
};

}  // namespace allspark
