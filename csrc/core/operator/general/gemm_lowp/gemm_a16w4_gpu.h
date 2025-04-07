/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w4_kernel.h>
#include <core/operator/operator.h>

#include <string>

#include "gemm_a16w4.h"

namespace allspark {

class GemmA16W4GPU : public GemmA16W4Base {
 public:
  explicit GemmA16W4GPU(const std::string& op_type = "")
      : GemmA16W4Base(op_type) {}
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  template <typename FType, typename QType>
  void get_weight_padded_k_align(TensorMap& weights_buffer);
  template <typename FType, typename QType>
  void get_weight_padded_n_align(TensorMap& weights_buffer);
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
  SplitKParams splitk_params_;

  // only do padding in decoder worker, prefill worker will reuse the padding
  // result
  bool do_padding_ = false;
};

}  // namespace allspark
#endif