/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mul_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>

namespace allspark {

class MulOp : public AsOperator {
 public:
  explicit MulOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  float alpha_;
};

}  // namespace allspark
