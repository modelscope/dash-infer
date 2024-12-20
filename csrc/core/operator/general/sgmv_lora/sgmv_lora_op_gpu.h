/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sgmv_lora_op_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/cuda_common.h>
#include <core/operator/operator.h>
#include <utility/check.h>

namespace allspark {
class SgmvLoraOpGPU : public AsOperator {
 public:
  SgmvLoraOpGPU(const std::string& op_type = "") : AsOperator(op_type) {
    is_lora_op_ = true;
    temp_qkv_.resize(3);
    lora_B_weight_parts_data_ptrs_.resize(3);
  }
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;

 private:
  AsStatus (*kernel_launcher)(DataType dtype, void* out, const void* in,
                              const AsTensor* weight_ptrs,
                              const AsTensor* segments, const AsTensor* ranks,
                              void* buf, int d_in, int d_out, bool is_k_tensor,
                              bool is_n_tensor, int num_problems, bool unsplit,
                              int unsplit_n, int max_rank, int CC,
                              const DeviceContext* ctx) = nullptr;
  // type: int64, shape: {>=#lora_weights}(may have duplicate ptr)
  std::shared_ptr<AsTensor> lora_A_weight_ptrs_;
  std::shared_ptr<AsTensor> lora_B_weight_ptrs_;

  // type: int32, shape: {lora_A/B_weight_ptrs_.shape[0]}
  std::shared_ptr<AsTensor> lora_ranks_;

  // for attention.self
  // type: int32, shape: {lora_B_weight_ptrs_.shape[0]}
  std::shared_ptr<AsTensor> lora_B_ranks_;

  // type: int32, shape: {lora_A/B_weight_ptrs_.shape[0] * 2}
  std::shared_ptr<AsTensor> segments_;
  // std::vector<int32_t> segments_;
  // segments_[weight_idx * 2] == starting batch idx
  // segments_[weight_idx * 2 + 1] == ending batch idx + 1

  // type: dtype, shape: {bs, seq_len, max_rank}
  // temp_ = input @ lora_A
  void* temp_;

  // type: dtype, shape: {bs, seq_len, max_rank / 3}
  std::shared_ptr<AsTensor> temp_qkv_ptrs_;
  std::vector<void*> temp_qkv_;

  // type: uint8, shape: {sgmv_tmp_size}
  void* buf_;

  // store 3 weight pointer tensors, each:
  // type: int64_t, shape: {lora_B_weight_ptrs_.shape[0]}
  std::vector<std::shared_ptr<AsTensor>> lora_B_weight_parts_vec_;

  int num_problems_ = 0;
  int max_lora_r_ = 0;
  float lora_scaling_ = 1.0;
  bool has_lora_bias_ = false;
  bool is_attention_self_ = false;

  // quant BEGIN
  // std::string inner_gemm_type_ = "Gemm";
  bool use_quant_ = false;
  // OperatorProto quant_op_proto_;
  //  quant END

  int lora_A_d_in_;
  int lora_A_d_out_;

  int lora_B_d_in_;
  int lora_B_d_out_;

  std::vector<AsTensor*> lora_A_weight_ptrs_vec_;
  std::vector<AsTensor*> lora_B_weight_ptrs_vec_;
  std::vector<std::vector<int64_t>> lora_B_weight_parts_data_ptrs_;
  std::vector<int32_t> lora_A_ranks_vec_;
  std::vector<int32_t> lora_B_ranks_vec_;
  std::vector<int32_t> segmented_batch_idx_;

  bool use_cublas_ = false;

  // batch中有不带lora_name的请求，output tensor相应位置需赋0
  bool need_set_zero_ = false;

  int CC_;  // compute capability

  DataType dtype_ = DATATYPE_UNDEFINED;

  // col-wise split lora_B.weight to 3 parts for qkv
  dim_t q_outdim_size_{0}, k_outdim_size_{0}, v_outdim_size_{0};
  std::vector<dim_t> qkv_weight_dims_;
  int qkv_sum_ = 0;

  int64_t ws_size_ = 0;
  size_t buf_size_ = 0;
};
}  // namespace allspark
#endif
