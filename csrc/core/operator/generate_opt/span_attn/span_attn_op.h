/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_op.h
 */
#pragma once

#include <memory>

#include "core/operator/operator.h"
#include "head/head_gqa.h"
#include "head/head_mha.h"

namespace allspark {

/**
 * @brief
 *
 * For decoder:
 *
 *  inputs:
 *      qkv_fuse: (batch * beam, 1, q_size + 2 * cache_size)
 *
 *  outputs:
 *      attn_out: (batch * beam, 1, q_size)
 */
class SpanAttnOp : public AsOperator {
 public:
  explicit SpanAttnOp(const std::string& op_type = "")
      : AsOperator(op_type),
        dtype_(DATATYPE_UNDEFINED),
        layer_num_(0),
        batch_size_(1),
        seq_len_(1),
        alpha_(-1.0f),
        multi_nodes_(false),
        causal_mask_(false) {}

  virtual ~SpanAttnOp() {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map,
                TensorMap* tensor_map) override final;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override final;
  AsStatus Forward(RuntimeContext* runtime_ctx) override final;
  AsStatus Alloc(RuntimeContext* runtime_ctx) override final;

 protected:
  /* for Init */
  /// @brief this should never set shape-related variables
  virtual AsStatus setAttributes(const OperatorProto& op_proto);
  virtual AsStatus deviceInit() = 0;

  /* for Reshape */
  /// @brief this must ensure all shape-related variables are set
  virtual AsStatus deviceReshape(const RuntimeContext* runtime_ctx) = 0;
  virtual AsStatus setWorkspace(const RuntimeContext* runtime_ctx);

  /* for Forward (prefill / context) */
  virtual void contextAttnLauncher(void* k_cache_buf, void* v_cache_buf,
                                   int beam_size) = 0;
  virtual void contextCopySpanLauncher(const AsTensor& k_cache_ptr_tensor,
                                       const AsTensor& v_cache_ptr_tensor,
                                       const void* k_contiguous_cache,
                                       const void* v_contiguous_cache) = 0;
  virtual void copyPrefixSpanToCtxMemLauncher(
      const AsTensor& k_cache_ptr_tensor, const AsTensor& v_cache_ptr_tensor,
      const void* k_contiguous_cache, const void* v_contiguous_cache) = 0;

  /* for Forward (decoder) */
  virtual void decoderAppendCacheLauncher() = 0;
  virtual void decoderAttnLauncher(const RuntimeContext* runtime_ctx) = 0;

 private:
  AsStatus runContext(RuntimeContext* runtime_ctx);
  AsStatus runDecoder(RuntimeContext* runtime_ctx);
  AsStatus decoderAppendCache(const RuntimeContext* runtime_ctx);

 protected:
  std::unique_ptr<AttentionHead> attn_head_;

  DataType dtype_;

  int layer_num_;

  // deduced from input shape
  int batch_size_;
  int seq_len_;

  /// @brief alpha for QK
  float alpha_;

  bool multi_nodes_;
  /// @brief this should always be true
  bool causal_mask_;

  // [batch, 1, num_heads, size_per_head]
  std::unique_ptr<AsTensor> decoder_q_tensor_;

  // [batch]
  std::unique_ptr<AsTensor> decoder_seq_len_tensor_host_;

  // [batch, num_spans], type T**, array of pointers
  std::unique_ptr<AsTensor> k_span_array_tensor_host_;
  std::unique_ptr<AsTensor> v_span_array_tensor_host_;
};

}  // namespace allspark
