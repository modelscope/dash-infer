/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_fp8_a8w8_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/operator/operator.h>

#include <string>

#include "gemm_fp8.h"

namespace allspark {

float median(std::vector<float>& times) {
  const size_t size = times.size();
  if (size == 0) {
    return 0;
  }

  std::sort(times.begin(), times.end());

  const size_t mid = size / 2;
  if (size % 2 == 0) {
    return (times[mid] + times[mid - 1]) / 2;
  } else {
    return times[mid];
  }
}

class GemmFP8A8W8GPU : public GemmFP8Base {
 public:
  GemmFP8A8W8GPU(const std::string& op_type = "") : GemmFP8Base(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override { return Reshape(); }
  AsStatus Forward(RuntimeContext* runtime_ctx) override { return Forward(); }
  AsStatus Reshape() override;
  AsStatus Forward() override;
  //   ~GemmFP8A8W8GPU();
 private:
  template <typename AType, typename WType>
  AsStatus DispatchKernel();
  void TuneUseHeuristic();
  template <typename WType>
  void trans_TN(TensorMap& weights_buffer);
  template <typename WType>
  void weight_scaled_fp8_quant(TensorMap& weights_buffer);

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
  std::string weight_name_pattern_;
  // float* weight_scale_ptr = nullptr;

  // only do padding in decoder worker, prefill worker will reuse the padding
  // result
  bool do_padding_ = false;

  cudaDataType_t scaleType_ = CUDA_R_32F;
  cublasComputeType_t computeType_ = CUBLAS_COMPUTE_32F;
  cudaDataType_t fp8_cuda_type_;
  cudaDataType_t in_cuda_type_;
  cublasOperation_t OpA_ = CUBLAS_OP_N;
  cublasOperation_t OpB_ = CUBLAS_OP_N;
  cublasLtMatmulDesc_t operationDesc_ = NULL;
  cublasLtMatmulPreference_t preference_ = NULL;
  cublasLtMatrixLayout_t Adesc_ = NULL, Bdesc_ = NULL, Cdesc_ = NULL,
                         Ddesc_ = NULL;
  cublasLtMatmulHeuristicResult_t heuristicResult_ = {};
  size_t workspaceLimit_;
};

}  // namespace allspark
#endif
