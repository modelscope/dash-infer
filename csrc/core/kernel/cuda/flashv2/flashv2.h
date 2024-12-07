/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    flashv2.h
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <vector>

#include "cuda/cuda_common.h"

#if CUDA_VERSION >= 11080
#include "flash_attn/src/flash.h"

#ifndef FLASH_ATTN_V2
#define FLASH_ATTN_V2
#endif  // FLASH_ATTN_V2

namespace allspark {

namespace cuda {

enum class FlashQKVFormat {
  INTERLEAVED = 0,  // [batch, seq, qkv, nhead, phead] qkv, qkv, qkv ...
  CONTINUOUS = 1,   // q: [batch, seqq, nhead, phead] qqq ...
                    // k: [batch, seqk, nhead, phead] kkk ...
                    // v: [batch, seqk, nhead, phead] vvv ...
  MIX = 2,          // q: [batch, seqq, qkv, nhead, phead] qxx, qxx, qxx
                    // k: [batch, seqk, nhead, phead] kkk ...
                    // v: [batch, seqk, nhead, phead] vvv ...
  UNKNOWN = 99
};

using flashv2_t = Flash_fwd_params;

size_t flashv2_wss(flashv2_t& params);

void flashv2_clear_param(flashv2_t& params);

void flashv2_set_static_param(flashv2_t& params, cudaDeviceProp& dprop,
                              cudaDataType_t dtype, const size_t batch,
                              const size_t qseql, const size_t kseql,
                              const size_t nhead, const size_t nhead_k,
                              const size_t phead, FlashQKVFormat qkv_format,
                              bool is_causal);

void flashv2_set_runtime_param(flashv2_t& params, void* q_ptr, void* k_ptr,
                               void* v_ptr, void* o_ptr, void* workspace,
                               float softmax_scale);

void flashv2_dispatch(flashv2_t& params, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif  // CUDA_VERSION >= 11080
