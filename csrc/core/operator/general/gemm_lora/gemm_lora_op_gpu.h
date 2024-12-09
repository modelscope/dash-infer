/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_lora_op_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/operator/general/gemm/gemm_op_gpu.h>

namespace allspark {
class GemmLoraOpGPU : public GemmOpGPU {
 public:
  GemmLoraOpGPU(const std::string& op_type = "") : GemmOpGPU(op_type) {
    is_lora_op_ = true;
  }
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;

 private:
  std::shared_ptr<AsTensor> batch_lora_weights_;
  std::shared_ptr<AsTensor> batch_lora_bias_;
  float lora_scaling_ = 1.0;  // after optimizing scaling into aslora file,
                              // lora_scaling_ always equals to 1.0
  bool has_lora_bias_ = false;
  OperatorProto op_proto_;
  std::map<std::pair<std::string, int>, std::shared_ptr<AsTensor>>
      qkv_weight_cache_;  // store splitted Q K V from lora weight
  std::map<std::pair<std::string, int>, std::shared_ptr<AsTensor>>
      qkv_bias_cache_;  // store splitted Q K V from lora bias

  // quant BEGIN
  std::string inner_gemm_type_ = "Gemm";
  bool use_quant_ = false;
  OperatorProto quant_op_proto_;
  // quant END

  /*
  int64_t lora_m_;
  int64_t lora_n_;
  int64_t lora_k_;
  int64_t lora_batch_;
  int lora_lda_;
  int lora_ldb_;
  int lora_ldc_;
  */
};
}  // namespace allspark
#endif
