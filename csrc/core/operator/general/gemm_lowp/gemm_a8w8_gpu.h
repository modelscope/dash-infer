/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a8w8_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/operator/operator.h>

#include <string>

#include "gemm_a16w8.h"
#include "gemm_a16w8_gpu.h"

/**
 * @brief
 * GemmA8W8GPU reuse GemmA16W8Base.
 * TBD: moreover only GemmA16WxBase to be reused.
 **/
namespace allspark {
class GemmA8W8GPU : public GemmA16W8Base {
 public:
  GemmA8W8GPU(const std::string& op_type = "") : GemmA16W8Base(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  template <typename FType, typename QType>
  void get_weight_padded_k_align(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void get_weight_padded_n_align(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void B_I8_Reorder_Hmma16816_N32_K16_Gpu(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void trans_TN(TensorMap& weights_buffer);
  template <typename FT, typename QT>
  void DispatchKernel();
  void GetWeightPaddedDispatch(const DataType ftype, const DataType qtype,
                               TensorMap& weights_buffer);

  void set_padding_flag(RuntimeContext* runtime_ctx) {
    bool do_padding = false;
    if (runtime_ctx == nullptr) {
      do_padding = true;
    } else if (runtime_ctx->is_context == false) {
      do_padding = true;
    }

    do_padding_ = do_padding;
  }

 private:
  int sm_count_;
  int sm_version_;
  bool is_kpad_ = false;
  bool is_npad_ = false;
  int64_t n_padded_before_;
  std::unique_ptr<GemmA16W8GPU> a16w8_op_ = nullptr;

  // only do padding in decoder worker, prefill worker will reuse the padding
  // result
  bool do_padding_ = false;
};

}  // namespace allspark
#endif
