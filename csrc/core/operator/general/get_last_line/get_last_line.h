/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    get_last_line.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>

namespace allspark {

class GetLastLineOp : public AsOperator {
 public:
  explicit GetLastLineOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  int batch_;
  int seq_;
  int hid_;
};

}  // namespace allspark
