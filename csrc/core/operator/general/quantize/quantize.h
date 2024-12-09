/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    quantize.h
 */

#pragma once

#include <core/operator/operator.h>
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/cuda_common.h>
#endif
namespace allspark {
/*!

 */
class QuantizeOp : public AsOperator {
 public:
  explicit QuantizeOp(const std::string& op_type = "")
      : AsOperator(op_type), m_(0), n_(0), k_(0), batch_(1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t batch_;
  int per_channel_ = 3;
  UnaryType activation_;
  size_t lhs_cnt_;
  size_t lhs_reduce_cnt_;
  int8_t* lhs_qdata_ = nullptr;
  float* lhs_scale_ = nullptr;
  int8_t* lhs_zero_ = nullptr;
  int* lhs_redsum_ = nullptr;
  template <typename FType, typename QType>
  AsStatus Process();

  //  template <typename FType, typename QType>
  //  AsStatus Postprocess();
  //  AsStatus matmul(void* workspace);
  //  AsStatus (*kernel_launcher)(DataType dtype, void* out, const void* in,
  //                              const void* bias, const AsTensor* weight, int
  //                              m, int n, int k, int lda, int ldb, int ldc,
  //                              bool transA, bool transB, int batch, float
  //                              alpha, UnaryType activation, const
  //                              DeviceContext* ctx) = nullptr;
  size_t aligne_size(size_t n, size_t aligne = 256) {
    return (n + aligne - 1) / aligne * aligne;
  }
};
}  // namespace allspark
