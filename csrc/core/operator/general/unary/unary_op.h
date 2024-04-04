/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>

namespace allspark {

class UnaryOp : public AsOperator {
 public:
  explicit UnaryOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  UnaryType unary_type_ = UNARYTYPE_UNDEFINED;
};

}  // namespace allspark
