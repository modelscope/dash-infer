/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batch_mqa_op.h
 */

#pragma once

#include <core/operator/operator.h>
#ifdef ENABLE_CUDA
#include <core/kernel/cuda/flashv2/flashv2.h>
#include <core/kernel/cuda/trivial_mha/trivial_mha.h>
#include <core/kernel/cuda/xformer_mha/xformer_mha.h>
#endif  // ENABLE_CUDA

#include "env_config.h"

namespace allspark {

/* @brief: assumpt seq_len = 1
 * inputs:
      qkv_fuse: (batch * beam, 1, hidden_size)
      mask: (batch * beam, 1, step) (只有第一次解码用得到这东西)
      beam_idx : [batch * beam]
 * outputs:
      attn_out: (batch * beam, 1, hidden_size)
 */

class BatchMQAOp : public AsOperator {
 public:
  explicit BatchMQAOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(768),
        num_heads_(16),
        size_per_head_(64),
        gemm_batch_(1),
        score_size_(0),
        src_blk_(0),
        tgt_blk_(0),
        alpha_(-1.0f),
        pos_embedding_(false),
        first_beam_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
  bool UseFlashAttn() const {
    return ctx_->GetPrefillMode() == AsMHAPrefill::AsPrefillFlashV2 &&
           seq_len_ > AttentionEnvConfig::GetFlashThresh();
  }
  AsStatus RunFlash(std::shared_ptr<GenerateContext> gen_ctx);
#endif
  AsStatus RunOneBatch(std::shared_ptr<GenerateContext> gen_ctx,
                       int current_batch);
  AsStatus ResetCache() override;
  AsStatus setWorkspace(const RuntimeContext* runtime_ctx);

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int batch_size_;
  int seq_len_;
  int hidden_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t cache_size_;
  int64_t score_size_;
  int src_blk_;
  int tgt_blk_;
  float alpha_;
  bool pos_embedding_;
  bool first_beam_;
  int group_num_ = 0;
  int qkv_stride_ = 0;
  int kv_stride_ = 0;

  // multi_gpu && multi_numa shared the same variable
  bool multi_nodes_;
  int layer_num_ = 0;
  size_t other_workspace_size_ = 0;

#ifdef ENABLE_CUDA
#ifdef FLASH_ATTN_V2
  cuda::flashv2_t flash_v2_params_;
#endif  // FLASH_ATTN_V2
  cudaDeviceProp dprop_;
  cuda::trivial_t trivial_params_;
#ifdef XFORMER_FMHA
  cuda::xformer_t xformer_params_;
#endif  // XFORMER_FMHA
#endif

  bool causal_mask_ = true;
  void (*kernel_launcher)(DataType dtype, void* out, void* score,
                          const void* query, const void* key, const void* value,
                          const float* mask, const void* position_embedding,
                          void* k_cache, void* v_cache, void** q_array,
                          void** k_array, void** v_array, void** score_array,
                          void** out_array, int batch_size, int beam_size,
                          int seq_len, int step, int cache_max_len,
                          int hidden_size, int num_heads, int size_per_head,
                          int group_num, int gemm_batch, float alpha,
                          void* other_workspace, size_t other_workspace_size,
                          const DeviceContext* ctx) = nullptr;
  //   void (*repeat_kv_kerenl_launcher)(DataType dtype,void* kptr,void*
  //   vptr,void* k_buf,void* v_buf,int seq_len,int num_heads,int group_num, int
  //   size_per_head,const DeviceContext* ctx) = nullptr;
  // void (*reorder_kv_cache_launcher)(DataType dtype, void* k_cache,
  //                                   void* v_cache, void* old_k_cache,
  //                                   void* old_v_cache, int* beam_idx,
  //                                   int batch_size, int beam_size, int
  //                                   inner_dim, const DeviceContext* ctx) =
  //                                   nullptr;
  // std::unique_ptr<AsTensor> k_cache_;
  // std::unique_ptr<AsTensor> v_cache_;
  // std::unique_ptr<AsTensor> tmp_k_cache_;
  // std::unique_ptr<AsTensor> tmp_v_cache_;
  // void* k_cache_buf_ = nullptr;
  // void* v_cache_buf_ = nullptr;
  // void* old_k_cache_buf_ = nullptr;
  // void* old_v_cache_buf_ = nullptr;

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
  void (*ctx_kernel_launcher)(DataType dtype, void* out, const void* query,
                              const void* key, const void* value,
                              const float* mask, const void* position_embedding,
                              void* k_cache, void* v_cache, int batch_size,
                              int beam_size, int seq_len, int step,
                              int cache_max_len, int hidden_size, int num_heads,
                              int size_per_head, int group_num, void* workspace,
                              int src_blk, int tgt_blk, float alpha,
                              const DeviceContext* ctx) = nullptr;
#endif
};

}  // namespace allspark
