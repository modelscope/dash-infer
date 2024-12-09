/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    flash_attention_operator.cu
 */

#ifdef ENABLE_CUDA
#include "cuda.h"
#include "flashv2.h"

#if CUDA_VERSION >= 11080
#include <cutlass/numeric_types.h>

#include <vector>

#include "flash_attn/src/static_switch.h"

namespace allspark {
namespace cuda {
size_t flashv2_wss(flashv2_t& params) {
  auto calculate_wss_round256 = [&](int batch, int nhead, int qseql) -> size_t {
    size_t wss_raw = size_t(batch) * nhead * qseql * sizeof(float) * 10;
    return (wss_raw + 256 - 1) / 256 * 256;
  };
  return calculate_wss_round256(params.b, params.h, params.seqlen_k);
}

void flashv2_clear_param(flashv2_t& params) {
  memset(&params, 0, sizeof(params));
}

void flashv2_set_static_param(flashv2_t& params, cudaDeviceProp& dprop,
                              cudaDataType_t dtype, const size_t batch,
                              const size_t qseql, const size_t kseql,
                              const size_t nhead, const size_t nhead_k,
                              const size_t phead, FlashQKVFormat qkv_format,
                              bool is_causal) {
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  switch (dtype) {
    case cudaDataType_t::CUDA_R_16BF:
      params.is_bf16 = true;
      break;
    case cudaDataType_t::CUDA_R_16F:
      params.is_bf16 = false;
      break;
    default:
      return;
  }
  params.is_causal = is_causal;
  params.SetCudaConfig(&dprop);

  params.b = batch;
  params.h = nhead;
  params.h_k = nhead_k;
  params.h_h_k_ratio = nhead / nhead_k;
  params.seqlen_q = qseql;
  params.seqlen_k = kseql;
  params.seqlen_q_rounded = round_multiple(qseql, 128);  // seqlen_q_rounded;
  params.seqlen_k_rounded = round_multiple(kseql, 128);  // seqlen_k_rounded;
  params.d = phead;
  params.d_rounded = round_multiple(phead, 32);  //  d_rounded;

  params.cu_seqlens_q = nullptr;  // static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = nullptr;  // static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = nullptr;     // static_cast<int *>(seqused_k);
  params.window_size_left = -1;
  params.window_size_right = -1;
  if (is_causal) {
    params.window_size_right = 0;
  }
  // params.is_seqlens_k_cumulative = true;

  switch (qkv_format) {
    case FlashQKVFormat::INTERLEAVED:
      params.q_batch_stride =
          qseql * (nhead + 2 * nhead_k) * phead;  // q->dim().count(1);
      params.k_batch_stride =
          kseql * (nhead + 2 * nhead_k) * phead;  // k->dim().count(1);
      params.v_batch_stride =
          kseql * (nhead + 2 * nhead_k) * phead;      // v->dim().count(1);
      params.o_batch_stride = qseql * nhead * phead;  // out->dim().count(1);
      params.q_row_stride = (nhead + 2 * nhead_k) * phead;
      params.k_row_stride = (nhead + 2 * nhead_k) * phead;
      params.v_row_stride = (nhead + 2 * nhead_k) * phead;
      params.o_row_stride = nhead * phead;  // out->dim().shape(2);
      params.q_head_stride = phead;         // head_dim
      params.k_head_stride = phead;
      params.v_head_stride = phead;
      params.o_head_stride = phead;
      break;
    case FlashQKVFormat::CONTINUOUS:
      params.q_batch_stride = qseql * nhead * phead;    // q->dim().count(1);
      params.k_batch_stride = kseql * nhead_k * phead;  // k->dim().count(1);
      params.v_batch_stride = kseql * nhead_k * phead;  // v->dim().count(1);
      params.o_batch_stride = qseql * nhead * phead;    // out->dim().count(1);
      params.q_row_stride = nhead * phead;  // num_head * head_dim = hidden_size
      params.k_row_stride = nhead_k * phead;  // k->dim().shape(2);
      params.v_row_stride = nhead_k * phead;  // v->dim().shape(2);
      params.o_row_stride = nhead * phead;    // out->dim().shape(2);
      params.q_head_stride = phead;           // head_dim
      params.k_head_stride = phead;
      params.v_head_stride = phead;
      params.o_head_stride = phead;
      break;
    case FlashQKVFormat::MIX:
      params.q_batch_stride =
          qseql * (nhead + 2 * nhead_k) * phead;        // q->dim().count(1);
      params.k_batch_stride = kseql * nhead_k * phead;  // k->dim().count(1);
      params.v_batch_stride = kseql * nhead_k * phead;  // v->dim().count(1);
      params.o_batch_stride = qseql * nhead * phead;    // out->dim().count(1);
      params.q_row_stride = (nhead + 2 * nhead_k) * phead;
      params.k_row_stride = nhead_k * phead;  // k->dim().shape(2);
      params.v_row_stride = nhead_k * phead;  // v->dim().shape(2);
      params.o_row_stride = nhead * phead;    // out->dim().shape(2);
      params.q_head_stride = phead;           // head_dim
      params.k_head_stride = phead;
      params.v_head_stride = phead;
      params.o_head_stride = phead;
      break;
    default:
      throw std::runtime_error("Unknown QKV format");
      break;
  }
}

void flashv2_set_runtime_param(flashv2_t& params, void* q_ptr, void* k_ptr,
                               void* v_ptr, void* o_ptr, void* workspace,
                               float softmax_scale) {
  // Set the pointers and strides.
  params.q_ptr = q_ptr;           // q->data();
  params.k_ptr = k_ptr;           // k->data();
  params.v_ptr = v_ptr;           // v->data();
  params.o_ptr = o_ptr;           // out->data();
  params.cu_seqlens_q = nullptr;  // static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = nullptr;  // static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = nullptr;     // static_cast<int *>(seqused_k);

  // Softmax sum
  params.softmax_lse_ptr = workspace;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  float p_dropout = 0.0;
  params.p_dropout = 1.f - p_dropout;
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
}

void flashv2_dispatch(flashv2_t& params, cudaStream_t stream) {
  run_mha_fwd(params, stream);
}

}  // namespace cuda
}  // namespace allspark
#endif  // CUDA_VERSION >= 11080
#endif  // ENABLE_CUDA
