/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_sparse_a8w8_gpu.h
 */

#pragma once
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/operator/operator.h>

#include <string>

#include "gemm_a16w8.h"
// #include "gemm_a16w8_gpu.h"
#include "gemm_a8w8_gpu.h"

/**
 * @brief
 * GemmSparseA8W8GPU reuse GemmA16W8Base.
 * TBD: moreover only GemmA16WxBase to be reused.
 **/
namespace allspark {
struct cusparseLtAlgCfg {
  int algo_config_id;
  int split_k;
  int split_k_mode;
  int split_k_buffers;
};
class GemmSparseA8W8GPU : public GemmA16W8Base {
 public:
  GemmSparseA8W8GPU(const std::string& op_type = "") : GemmA16W8Base(op_type) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map, RuntimeContext* runtime_ctx) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;
  virtual ~GemmSparseA8W8GPU() override;
  bool UseSparseOpt() { return is_sparse; }

 private:
  template <typename FType, typename QType>
  void trans_TN(TensorMap& weights_buffer);
  template <typename FT, typename QT>
  void DispatchKernel();
  void GetWeightPaddedDispatch(const DataType ftype, const DataType qtype,
                               TensorMap& weights_buffer);
#ifdef ENABLE_CUSPARSELT
  bool IsSparseWeight(cusparseLtMatDescriptor_t& matA,
                      cusparseLtMatDescriptor_t& matB,
                      cusparseLtMatDescriptor_t& matC,
                      cusparseLtMatmulDescriptor_t& matmul,
                      cusparseLtMatmulAlgSelection_t& alg_sel,
                      cusparseLtMatmulPlan_t& plan);
  void CompressWeightAndSearch(cusparseLtMatDescriptor_t& matA,
                               cusparseLtMatDescriptor_t& matB,
                               cusparseLtMatDescriptor_t& matC,
                               cusparseLtMatmulDescriptor_t& matmul,
                               cusparseLtMatmulAlgSelection_t& alg_sel,
                               cusparseLtMatmulPlan_t& plan);
  void InitCuSparseDescAlgPlan(cusparseLtHandle_t& cslt_handle,
                               cusparseLtMatDescriptor_t& matA,
                               cusparseLtMatDescriptor_t& matB,
                               cusparseLtMatDescriptor_t& matC,
                               cusparseLtMatmulDescriptor_t& matmul,
                               cusparseLtMatmulAlgSelection_t& alg_sel,
                               cusparseLtMatmulPlan_t& plan, const int M,
                               const int N, const int K);
#endif
 private:
  int sm_count_;
  int sm_version_;
  int64_t n_padded_before_;
  int64_t m_padded_after_;
  bool is_sparse = false;
  cusparseLtAlgCfg best_cfg;
  size_t cslt_ws_size = 0;
  int* d_valid;
  bool is_prompt;
};

}  // namespace allspark
#endif
