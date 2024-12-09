/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dynamic_quantize_matmul_fused.h
 */

#pragma once

#include <core/operator/operator.h>
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/cuda_common.h>

#include "core/kernel/cuda/cuda_fused_matmul/fused_matmul_interface.hpp"
#endif
namespace allspark {
/*!

 */
class DynamicQuantizeMatmulFusedOp : public AsOperator {
 public:
  explicit DynamicQuantizeMatmulFusedOp(const std::string& op_type = "")
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
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 public:
  // fused kernel.
  hie::DataType exec_type_ = hie::DataType::FLOAT;
  hie::dqmm_fused_kernel ker_ = hie::dqmm_fused_kernel::DQMM_FUSED_NONE;

  // device info
  int sm_cnt_ = 0;
  int sm_ver_ = 0;
  int sm_share_ = 0;
  bool force_exec_typical_kernel_ = false;

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t batch_;

  float alpha_;
  float beta_;
  int transA_;
  int transB_;
  int lda_;
  int ldb_;
  int ldc_;
  bool is_pooler_;  // for bert pooler
  int per_channel_ = 3;
  std::string active_ = "NONE";
  UnaryType activation_;
  size_t lhs_cnt_;
  size_t lhs_reduce_cnt_;
  size_t rhs_cnt_;
  size_t rhs_reduce_cnt_;
  int8_t* lhs_qdata_ = nullptr;
  int8_t* rhs_qdata_ = nullptr;
  float* lhs_scale_ = nullptr;
  float* rhs_scale_ = nullptr;
  int8_t* lhs_zero_ = nullptr;
  int8_t* rhs_zero_ = nullptr;
  int* lhs_redsum_ = nullptr;
  int* rhs_redsum_ = nullptr;
  bool with_bias_ = false;
  bool with_elemwiseB_ = false;
  int elemwiseA_ndims_ = 0;
  int elemwiseB_ndims_ = 0;
  hie::Array<uint32_t, 6> lhs_reduce_dims_;
  hie::Array<uint32_t, 6> rhs_reduce_dims_;
  hie::Array<uint32_t, 6> output_dims_;
  hie::Array<uint32_t, 6> bias_dims_;
  hie::Array<uint32_t, 6> elemwiseA_dims_;  // int8-gemm`s int32 data shape.
  hie::Array<uint32_t, 6> elemwiseB_dims_;  // residual data shape.
  int* elemwiseA_ = nullptr;  // Int8-gemm int32 result. Store in workspace.
  int* elemwiseB_ = nullptr;  // residual data shape.
  std::unique_ptr<AsTensor> elemwiseA;
  template <typename FType, typename QType>
  AsStatus Preprocess();
  template <typename FType, typename QType>
  AsStatus Postprocess();
  size_t aligne_size(size_t n, size_t aligne = 256) {
    return (n + aligne - 1) / aligne * aligne;
  }
};
}  // namespace allspark
