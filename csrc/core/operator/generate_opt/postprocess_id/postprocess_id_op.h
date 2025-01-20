/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    postprocess_id_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class PostProcessIdOp : public AsOperator {
 public:
  explicit PostProcessIdOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;

 private:
  int rank_ = 0;
};

}  // namespace allspark
