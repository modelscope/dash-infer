/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    calc_expert_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>

namespace allspark {

class CalcExpertOp : public AsOperator {
 public:
  explicit CalcExpertOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  int num_expert_;
  int total_token_;
  int hidden_size_;
};

}  // namespace allspark
