/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/operator/operator.h>

#include <string>

#include "gemm_a16w8.h"

namespace allspark {
class GemmA16W8GPU : public GemmA16W8Base {
 public:
  GemmA16W8GPU(const std::string& op_type = "") : GemmA16W8Base(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override { return Reshape(); }
  AsStatus Forward(RuntimeContext* runtime_ctx) override { return Forward(); }
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  template <typename FType, typename QType>
  void get_weight_padded_k_align(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void get_weight_padded_n_align(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void B_I8_Reorder_Hmma16816_N32_K16_Cpu(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void B_I8_Reorder_Hmma16816_N32_K16_Gpu(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void trans_TN(TensorMap& weights_buffer);
  template <typename FT, typename QT>
  void DispatchKernel();
  void GetWeightPaddedDispatch(const DataType ftype, const DataType qtype,
                               TensorMap& weights_buffer);

 private:
  int sm_count_;
  int sm_version_;
  bool is_kpad_ = false;
  bool is_npad_ = false;
  int64_t n_padded_before_;
  SplitKParams splitk_params_;
  cuda::GemmA16W8Launcher::KernelType ktype;
};

}  // namespace allspark
#endif
