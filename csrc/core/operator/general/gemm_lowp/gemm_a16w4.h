/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4.h
 */

#pragma once
#include <core/operator/operator.h>

#include <string>
namespace allspark {

/**
 * @brief
 *
 * Input [4-5]:
 *      input[0]    fdata   [Batch, M, K]       float16 or bfloat16
 *      input[1]    weight  [K, (N+1)/2]        int8 or uint8.          Two 4bit
 * pack in one 8bit. int8 mean int4, uint8 mean uint4. input[2]    scale   [1,
 * N] or [*, N]    float16 or bfloat16 input[3]    zero    [1, N] or [*, N]
 * float16 or bfloat16 input[4]    bias    [N]                 float16 or
 * bfloat16 Output [1]: output[0]   res     [Batch, M, N]       float16 or
 * bfloat16
 *
 *
 * Two 4bit pack into One 8bit
 * Python-Demo:
 *      data shape is [K, N]
 *      data_pack = (data[:, 1::2] << 4) | (data[:, 0::2] & 0xf)
 *
 * Example:
 *      uint8: [250] <-> [10, 15]
 *      uint8: [67]  <-> [3, 4]
 *
 */
class GemmA16W4Base : public AsOperator {
 public:
  explicit GemmA16W4Base(const std::string& op_type = "")
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
                  TensorMap* tensor_map) override;
  virtual AsStatus Reshape() = 0;
  virtual AsStatus Forward() = 0;

 protected:
  DataType ftype_ = DATATYPE_UNDEFINED;
  DataType qtype_ = DATATYPE_UNDEFINED;
  int64_t m_;
  int64_t n_;
  int64_t n_pack_;
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