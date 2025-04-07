/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_capsule_op_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/operator/general/gemm/gemm_op_gpu.h>

namespace allspark {
class GemmLoraCapsuleOpGPU : public AsOperator {
 public:
  GemmLoraCapsuleOpGPU(const std::string& op_type = "GemmLoraCapsule")
      : AsOperator(op_type) {
    is_lora_op_ = true;
  }
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;

 private:
  void SwitchLoraGraph(bool use_std_gemm_graph);
  bool has_lora_in_batch_ = false;
  std::unique_ptr<AsOperator> std_gemm_op_;
  std::vector<std::unique_ptr<AsOperator>> lora_op_list_;
  std::string inner_gemm_type_ = "Gemm";  // for quant
  std::string capsule_out_name_, base_out_name_, bin_add_out_name_;
  UnaryType activation_ = UnaryType::UNARYTYPE_UNDEFINED;
};
}  // namespace allspark
#endif
