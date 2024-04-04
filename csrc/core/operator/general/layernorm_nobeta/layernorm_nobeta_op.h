/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    layernorm_nobeta_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class LayerNormNoBetaOp : public AsOperator {
 public:
  explicit LayerNormNoBetaOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  int hidden_size_ = 768;
  float eps_ = 1e-12f;
  AsStatus (*kernel_launcher)(DataType dtype, void* out, const void* input,
                              const void* bias, const void* gamma, int m, int n,
                              float eps, const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
