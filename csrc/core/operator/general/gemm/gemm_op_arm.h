/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_arm.h
 */

#pragma once
#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>
#include <core/operator/operator.h>

#include <common/hie_bfloat16.hpp>

#include "gemm_op_cpu.h"

namespace allspark {
class GemmOpARM : public GemmOpCPU {
 public:
  GemmOpARM(const std::string& op_type = "") : GemmOpCPU(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;

 private:
  int K_pack_;
  void PackWeightBf16(const uint32_t N, const uint32_t K, const uint32_t K_pack,
                      const float* b_fp32, hie::bfloat16* b_bf16);
};
}  // namespace allspark
#endif
