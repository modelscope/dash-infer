/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_op_a8w8_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/cuda/moe_lowp/moe_a8w8_kernel.h>
#include <core/operator/operator.h>

namespace allspark {

class MoeA8W8Gpu : public AsOperator {
 public:
  explicit MoeA8W8Gpu(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  template <typename QT>
  void DispatchReshapeKernel(void* ws_ptr);
  template <typename FT, typename QT>
  void DispatchKernel();
  void B_I8_Tranpose_Dim12_Gpu();
  DataType ftype_ = DATATYPE_UNDEFINED;
  int sm_count_;
  int sm_version_;
  DataType dtype_;
  int num_expert_;
  int num_expert_pertoken_;
  int total_token_;
  int hidden_size_;
  int64_t ws_size_;
  int block_size_;
  int proj_size_;
  int64_t expert_size_;
  bool first_moe_ = true;
  std::unique_ptr<AsTensor> experts_score_;
  std::unique_ptr<AsTensor> float_gate_score_;
  std::unique_ptr<AsTensor> topk_value_;
  std::unique_ptr<AsTensor> topk_indice_;
  std::unique_ptr<AsTensor> experts_idx_;
  std::unique_ptr<AsTensor> experts_seq_;
  std::unique_ptr<AsTensor> indice_source_;
  std::unique_ptr<AsTensor> total_tokens_post_pad_;

  std::unique_ptr<AsTensor> gate_up_proj_array_ptr;
  std::unique_ptr<AsTensor> down_proj_array_ptr;
  std::unique_ptr<AsTensor> reorder_data_array_ptr;
  std::unique_ptr<AsTensor> gate_up_proj_out_array_ptr;
  std::unique_ptr<AsTensor> mid_result_array_ptr;
  std::unique_ptr<AsTensor> final_result_array_ptr;

  // FT type
  void* gate_up_proj_out;
  void* mid_result;
  void* final_result;
  void** gate_up_proj_array;
  void** down_proj_array;
  void** reorder_data_array;
  void** gate_up_proj_out_array;
  void** mid_result_array;
  void** final_result_array;

  // reorder quantization data
  int8_t* in_qdata;
  int8_t* in_reorder_qdata;
  float* in_scale;
  float* in_reorder_scale;
  float* in_red_max;
  uint32_t* in_red_count;
  int32_t* in_red_sum;
  int32_t* in_reorder_red_sum;

  // reorder mid data
  int8_t* mid_qdata;
  float* mid_scale;
  float* mid_red_max;
  uint32_t* mid_red_count;
  int32_t* mid_red_sum;

  // reorder weight array
  void** gate_up_proj_scale_array;
  void** gate_up_proj_zero_array;
  void** down_proj_scale_array;
  void** down_proj_zero_array;

  // batch gemm tmp data
  int32_t* gate_up_proj_out_i32;
  int32_t* final_result_i32;
};

}  // namespace allspark
#endif