/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_mha.h
 */

#pragma once
#include "core/kernel/kernel.h"
#ifdef ENABLE_CUDA

#ifndef XFORMER_FMHA
#define XFORMER_FMHA
#endif  // XFORMER_FMHA
namespace allspark {
namespace cuda {
enum class XformerQKVFormat {
  INTERLEAVED = 0,  // [batch, seq, qkv, nhead, phead] qkv, qkv, qkv ...
  CONTINUOUS = 1,   // q: [batch, seqq, nhead, phead] qqq ...
                    // k: [batch, seqk, nhead, phead] kkk ...
                    // v: [batch, seqk, nhead, phead] vvv ...
  MIX = 2,          // q: [batch, seqq, qkv, nhead, phead] qxx, qxx, qxx
                    // k: [batch, seqk, nhead, phead] kkk ...
                    // v: [batch, seqk, nhead, phead] vvv ...
  UNKNOWN = 99
};

class XformerMHAParam {
 public:
  bool causal;
  DataType dtype;
  size_t batch, nhead, nhead_kv, phead, seqlen_q, seqlen_k;
  int sm_version;
  XformerQKVFormat qkv_format;
};
using xformer_t = XformerMHAParam;

AsStatus xformer_prefill_attention(const xformer_t& param, void* qptr,
                                   void* kptr, void* vptr, void* output,
                                   void* workspace, float alpha,
                                   const cudaStream_t& stream);

size_t xformer_prefill_attention_workspace_inbytes(const xformer_t& param);

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
