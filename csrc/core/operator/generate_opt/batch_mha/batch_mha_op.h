/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batch_mha_op.h
 */

#pragma once
#include <core/operator/operator.h>

#include "env_config.h"

namespace allspark {

/* @brief: assumpt seq_len = 1
 * inputs:
      qkv_fuse: (batch * beam, 1, hidden_size)
      mask: (batch * beam, 1, step)
      beam_idx : [batch * beam]
 * outputs:
      attn_out: (batch * beam, 1, hidden_size)
 */

class BatchMHAOp : public AsOperator {
 public:
  explicit BatchMHAOp(const std::string& op_type = "")
      : AsOperator(op_type),
        score_size_(0),
        src_blk_(0),
        tgt_blk_(0),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(1),
        num_heads_(1),
        size_per_head_(1),
        gemm_batch_(1),
        layer_num_(0),
        dtype_(DATATYPE_UNDEFINED),
        alpha_(-1.0f),
        pos_embedding_(false),
        first_beam_(false),
        multi_nodes_(false),
        causal_mask_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus Alloc(RuntimeContext* runtime_ctx) override;

 protected:
  virtual AsStatus runContext(RuntimeContext* runtime_ctx);
  virtual AsStatus runDecoder(RuntimeContext* runtime_ctx);
#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
  bool useFlashAttn() const {
    return ctx_->GetPrefillMode() == AsMHAPrefill::AsPrefillFlashV2 &&
           seq_len_ > AttentionEnvConfig::GetFlashThresh();
  }
  AsStatus runFlash(GenerateContext* gen_ctx);
#endif
  AsStatus runOneBatch(GenerateContext* gen_ctx, int current_batch);

  AsStatus lognFromAttributes(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("logn_model_embedding") != attr.end()) {
      enable_logn_ = true;
      xlogn_ = *(int*)(attr.at("logn_model_embedding").c_str());
      if (xlogn_ == 0) {
        LOG(ERROR) << "logn_model_embedding can't be 0!" << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
  void (*ctx_kernel_launcher)(DataType dtype, void* out, const void* query,
                              const void* key, const void* value,
                              const float* mask, const void* position_embedding,
                              void* k_cache, void* v_cache, int batch_size,
                              int beam_size, int seq_len, int step,
                              int cache_max_len, int hidden_size, int num_heads,
                              int size_per_head, void* workspace, int src_blk,
                              int tgt_blk, float alpha,
                              const DeviceContext* ctx) = nullptr;
#endif

  void (*dec_kernel_launcher)(
      DataType dtype, void* out, void* score, const void* query,
      const void* key, const void* value, const float* mask,
      const void* position_embedding, void* k_cache, void* v_cache,
      void** q_array, void** k_array, void** v_array, void** score_array,
      void** out_array, int batch_size, int beam_size, int seq_len, int step,
      int cache_max_len, int hidden_size, int num_heads, int size_per_head,
      int gemm_batch, float alpha, bool xlogn_enable, int xlogn_len,
      const DeviceContext* ctx) = nullptr;

  std::pair<bool, AsMHAPrefill> GetPrefillMode() {
    AsMHAPrefill prefill_mode = AsMHAPrefill(ctx_->GetPrefillMode());

    std::string mha_dtype_indicator =
        "decoder.layer.0.attention.output.dense.out";
    auto tensor_map_iter = tensor_map_->find(mha_dtype_indicator);
    bool dtype_indicator_exist = false;      // if false, cannot use flash.
    DataType mha_dtype = DataType::FLOAT32;  // flash only support bf16 / half
    if (tensor_map_iter != tensor_map_->end()) {
      dtype_indicator_exist = true;
      mha_dtype = tensor_map_->at(mha_dtype_indicator).get()->GetDataType();
    }

    return std::make_pair(true, prefill_mode);
  }

  int64_t score_size_;
  int src_blk_;
  int tgt_blk_;
  int batch_size_;
  int seq_len_;
  int hidden_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int layer_num_;
  int xlogn_ = -1;  // model base sequence embedding length.
  DataType dtype_ = DATATYPE_UNDEFINED;
  float alpha_;
  bool pos_embedding_;
  bool first_beam_;
  bool multi_nodes_;
  bool causal_mask_ = false;
  bool enable_logn_ = false;
};

}  // namespace allspark
