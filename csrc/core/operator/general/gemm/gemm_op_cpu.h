/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_cpu.h
 */

#pragma once
#include <core/operator/operator.h>

#include "gemm_op.h"

namespace allspark {
class GemmOpCPU : public GemmOpBase {
 public:
  GemmOpCPU(const std::string& op_type = "") : GemmOpBase(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 protected:
  DataType weight_data_type_ = DataType::FLOAT32;
  int reshape_cnt = 0;
};
}  // namespace allspark
