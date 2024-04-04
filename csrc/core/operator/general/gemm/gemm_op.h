/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op.h
 */

#pragma once
#include <core/operator/operator.h>

namespace allspark {
/*!
 * @brief 支持gemm和batch_gemm
   batch_gemm仅支持x和不同的weight相乘
   x不支持转置, w支持转置
   不支持beta不为0的情形
 */
class GemmOpBase : public AsOperator {
 public:
  explicit GemmOpBase(const std::string& op_type = "")
      : AsOperator(op_type),
        m_(0),
        n_(0),
        k_(0),
        batch_(1),
        transB_(false),
        lda_(0),
        ldb_(0),
        ldc_(0),
        alpha_(1.0f),
        is_pooler_(false),
        is_split_k_(false),
        nranks_(1),
        rank_id_(0),
        activation_(UnaryType::UNARYTYPE_UNDEFINED) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape(int yn);

 protected:
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t batch_;
  bool transB_;
  int lda_;
  int ldb_;
  int ldc_;
  float alpha_;
  bool is_pooler_;
  bool is_split_k_;
  UnaryType activation_;
  BinaryType binary_type_ = BINARYTYPE_UNDEFINED;
  int nranks_;
  int rank_id_;
  DataType dtype_ = DATATYPE_UNDEFINED;
};
}  // namespace allspark
