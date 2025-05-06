/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qk_layernorm_nobeta_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class QKLayerNormNoBetaOp : public AsOperator {
 public:
  explicit QKLayerNormNoBetaOp(const std::string& op_type = "")
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
  int head_dim_ = 128;
  int batch_size_ = 1;
  int num_heads_ = 1;
  int multi_query_group_num_ = 1;
  float eps_ = 1e-12f;
  std::unique_ptr<AsTensor> input_points_, output_points_;
  std::unique_ptr<AsTensor> q_host_input_points_, q_host_output_points_;
  std::unique_ptr<AsTensor> k_host_input_points_, k_host_output_points_;
};
}  // namespace allspark
