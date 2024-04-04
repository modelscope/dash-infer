/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_arm.h
 */

#pragma once
#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>
#include <core/operator/operator.h>

#include <common/hie_bfloat16.hpp>
#include <string>

#include "gemm_a16w8.h"

namespace allspark {
class GemmA16W8ARM : public GemmA16W8Base {
 public:
  GemmA16W8ARM(const std::string& op_type = "") : GemmA16W8Base(op_type) {}

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

 private:
  void PackWeight(const uint32_t N, const uint32_t K, const uint32_t K_pack,
                  const uint8_t* b_u8_unpack, uint8_t* b_u8);
  void PackWeightBf16(const uint32_t N, const uint32_t K, const uint32_t K_pack,
                      const uint8_t* b_u8_unpack, hie::bfloat16* b_bf16,
                      const float* scale, const float* zero, int group_size);
  void ProcessQuantParam(float* scale, float* zero, Shape scale_shape,
                         float* scale_new, float* zero_new);
  void ProcessQuantParamFp16(float* scale, float* zero, Shape scale_shape,
                             float16_t* scale_fp16, float16_t* scaleXzp_fp16);

  std::shared_ptr<AsTensor> weight_bf16_packed_;
  int K_pack_;
};

}  // namespace allspark
#endif