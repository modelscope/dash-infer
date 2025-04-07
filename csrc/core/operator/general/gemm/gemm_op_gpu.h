/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/cuda_common.h>
#include <core/operator/operator.h>

#include <core/kernel/cuda/hie/cuda_activation.hpp>

#include "gemm_op.h"

namespace allspark {
class GemmOpGPU : public GemmOpBase {
 public:
  GemmOpGPU(const std::string& op_type = "") : GemmOpBase(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
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

 protected:
  AsStatus (*kernel_launcher)(DataType dtype, void* out, const void* in,
                              const void* bias, const AsTensor* weight, int m,
                              int n, int k, int lda, int ldb, int ldc,
                              bool transA, bool transB, int batch, float alpha,
                              const void* bin_res, UnaryType activation,
                              const DeviceContext* ctx) = nullptr;

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
  template <typename FType, typename QType>
  AsStatus DeQuantize();
  template <typename FType>
  void get_weight_padded_k_align8(TensorMap& weights_buffer);
  template <typename FType>
  void get_weight_padded_n_align8(TensorMap& weights_buffer);
  int sm_count_;

  bool is_kpad_ = false;
  bool is_npad_ = false;
  int64_t n_padded_before_;

  // only do padding in decoder worker, prefill worker will reuse the padding
  // result
  bool do_padding_ = false;
};
}  // namespace allspark
#endif
