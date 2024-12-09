/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_mha_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/* @brief: assumpt seq_len = 1
 * inputs:
      qkv_fuse: (batch * beam, 1, hidden_size)
      mask: (batch * beam, 1, step) (只有第一次解码用得到这东西)
      beam_idx : [batch * beam]
 * outputs:
      attn_out: (batch * beam, 1, hidden_size)
 */

class DecOptMHAOp : public AsOperator {
 public:
  explicit DecOptMHAOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(768),
        num_heads_(16),
        size_per_head_(64),
        gemm_batch_(1),
        score_size_(0),
        alpha_(-1.0f),
        pos_embedding_(false),
        first_beam_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus ResetCache() override;

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
  float alpha_;
  bool pos_embedding_;
  bool first_beam_;
  int input_len_ = 1;
  bool multi_nodes_;
  void (*kernel_launcher)(DataType dtype, void* out, void* score,
                          const void* query, const void* key, const void* value,
                          const float* mask, const void* position_embedding,
                          void* k_cache, void* v_cache, void** q_array,
                          void** k_array, void** v_array, void** score_array,
                          void** out_array, int batch_size, int beam_size,
                          int seq_len, int step, int cache_max_len,
                          int hidden_size, int num_heads, int size_per_head,
                          int gemm_batch, float alpha, int input_len,
                          bool xlogn_enable,
                          int xlogn_len, /* logn attention support */
                          const DeviceContext* ctx) = nullptr;
  void (*reorder_kv_cache_launcher)(DataType dtype, void* k_cache,
                                    void* v_cache, void* old_k_cache,
                                    void* old_v_cache, int* beam_idx,
                                    int batch_size, int beam_size,
                                    int inner_dim,
                                    const DeviceContext* ctx) = nullptr;
  std::unique_ptr<AsTensor> k_cache_;
  std::unique_ptr<AsTensor> v_cache_;
  std::unique_ptr<AsTensor> tmp_k_cache_;
  std::unique_ptr<AsTensor> tmp_v_cache_;
  void* k_cache_buf_ = nullptr;
  void* v_cache_buf_ = nullptr;
  void* old_k_cache_buf_ = nullptr;
  void* old_v_cache_buf_ = nullptr;

 private:
  // logn support
  int xlogn_ = -1;  // model base sequence embedding length.
  bool enable_logn_ = false;
  AsStatus logn_from_attributes_(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("logn_model_embedding") != attr.end()) {
      enable_logn_ = true;
      xlogn_ = *(int*)(attr.at("logn_model_embedding").c_str());
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
