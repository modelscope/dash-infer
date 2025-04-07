/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_op_cuda.h
 */
#pragma once

#include <core/kernel/cuda/flashv2/flashv2.h>
#include <core/kernel/cuda/trivial_mha/trivial_mha.h>
#include <spanattn/span_attn.h>

#include "core/kernel/kernel.h"
#include "cuda/cuda_kernel_span_cache.h"
#include "span_attn_op.h"       // NOLINT
#include <core/kernel/cuda/xformer_mha/xformer_mha.h>

namespace allspark {

class SpanAttnOpCUDA : public SpanAttnOp {
 public:
  explicit SpanAttnOpCUDA(const std::string& op_type = "")
      : SpanAttnOp(op_type) {}

  virtual ~SpanAttnOpCUDA() = default;

 protected:
  /* for Init */
  AsStatus deviceInit() override;

  /* for Reshape */
  AsStatus deviceReshape(const RuntimeContext* runtime_ctx) override;
  AsStatus setWorkspace(const RuntimeContext* runtime_ctx) override;

  /* for Forward (prefill / context) */
  void contextAttnLauncher(void* k_cache_buf, void* v_cache_buf,
                           GenerateContext* gen_ctx) override;
  void contextCopySpanLauncher(const AsTensor& k_cache_ptr_tensor,
                               const AsTensor& v_cache_ptr_tensor,
                               const void* k_contiguous_cache,
                               const void* v_contiguous_cache,
                               int prefix_len) override;
  void copyPrefixSpanToCtxMemLauncher(const AsTensor& k_cache_ptr_tensor,
                                      const AsTensor& v_cache_ptr_tensor,
                                      const void* k_contiguous_cache,
                                      const void* v_contiguous_cache,
                                      int prefix_len) override;

  /* for Forward (decoder) */
  void decoderAppendCacheLauncher() override;
  void decoderAttnLauncher(const RuntimeContext* runtime_ctx) override;

 private:
  AsStatus setDecoderWorkspaceSize();

#ifdef FLASH_ATTN_V2
  cuda::flashv2_t flash_v2_params_;
#endif  // FLASH_ATTN_V2
  cudaDeviceProp dprop_;
  // trivial_attention is deprecated
  // cuda::trivial_t trivial_params_;
#ifdef XFORMER_FMHA
  cuda::xformer_t xformer_params_;
#endif  // XFORMER_FMHA

  // [batch]
  std::unique_ptr<AsTensor> decoder_seq_len_tensor_device_;

  // [batch, num_spans], type T**, array of pointers
  std::unique_ptr<AsTensor> k_span_array_tensor_device_;
  std::unique_ptr<AsTensor> v_span_array_tensor_device_;

  std::unique_ptr<AsTensor> host_workspace_;
};

}  // namespace allspark
