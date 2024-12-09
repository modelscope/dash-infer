/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/moe/moe_kernel.h>
#endif
namespace allspark {

class MoeOp : public AsOperator {
 public:
  explicit MoeOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType dtype_;
  int num_expert_;
  int num_expert_pertoken_;
  int total_token_;
  int hidden_size_;
  int64_t ws_size_;
  int proj_size_;
  int64_t expert_size_;
  bool first_moe_ = true;
  void* hWs;
  size_t hWsSize, dWsSize;
  std::unique_ptr<AsTensor> experts_score_;
  std::unique_ptr<AsTensor> float_gate_score_;
  std::unique_ptr<AsTensor> topk_value_;
  std::unique_ptr<AsTensor> topk_indice_;
  std::unique_ptr<AsTensor> mid_row_indices_;
  std::unique_ptr<AsTensor> mid_expert_indices_;
  std::unique_ptr<AsTensor> final_row_indices_;
  //
  void* reorder_data;
  void* gate_up_proj_out;
  void* mid_result;
  void* final_result;
};

}  // namespace allspark
