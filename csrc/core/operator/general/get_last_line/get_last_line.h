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
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  void UpdateHiddenStates(RuntimeContext* runtime_ctx, AsTensor* out_tensor);

 private:
  int batch_;
  int seq_len_;
  int hidden_size_;
};

}  // namespace allspark
