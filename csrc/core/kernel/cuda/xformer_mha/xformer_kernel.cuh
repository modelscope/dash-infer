/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_kernel.cuh
 */

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "kernel_forward.h"
#include "xformer_mha.h"

namespace allspark {
namespace cuda {

namespace xformer_utils {
template <typename CType>
struct CutlassType2DataType {
  constexpr static DataType dt = DataType::DATATYPE_UNDEFINED;
};
template <>
struct CutlassType2DataType<float> {
  constexpr static DataType dt = DataType::FLOAT32;
};
#ifdef ENABLE_FP16
template <>
struct CutlassType2DataType<cutlass::half_t> {
  constexpr static DataType dt = DataType::FLOAT16;
};
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
template <>
struct CutlassType2DataType<cutlass::bfloat16_t> {
  constexpr static DataType dt = DataType::BFLOAT16;
};
#endif  // ENABLE_BF16

template <typename CArch>
struct CutlassArch2Int {
  constexpr static int version = 0;
};
template <>
struct CutlassArch2Int<cutlass::arch::Sm80> {
  constexpr static int version = 0x0800;
};
template <>
struct CutlassArch2Int<cutlass::arch::Sm75> {
  constexpr static int version = 0x0705;
};
template <>
struct CutlassArch2Int<cutlass::arch::Sm70> {
  constexpr static int version = 0x0700;
};

}  // namespace xformer_utils

template <typename CType, typename FMHA>
void xformer_prefill_attention_param_cast(const xformer_t& pfrom,
                                          typename FMHA::Params& pto) {
  pto.logsumexp_ptr = nullptr;
  pto.output_accum_ptr = nullptr;
  pto.num_batches = pfrom.batch;
  pto.num_heads = pfrom.nhead;
  pto.num_heads_kv = pfrom.nhead_kv;
  pto.head_dim = pfrom.phead;
  pto.head_dim_value = pfrom.phead;

  switch (pfrom.qkv_format) {
    case XformerQKVFormat::INTERLEAVED:
      pto.num_queries = pfrom.seqlen_q;
      pto.num_keys = pfrom.seqlen_k;
      pto.q_strideH = pfrom.phead;
      pto.k_strideH = pfrom.phead;
      pto.v_strideH = pfrom.phead;
      pto.q_strideM = (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.k_strideM = (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideM = (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.o_strideM = pfrom.nhead * pfrom.phead;
      pto.q_strideB =
          pfrom.seqlen_q * (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.k_strideB =
          pfrom.seqlen_k * (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideB =
          pfrom.seqlen_k * (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.custom_mask_type =
          pfrom.causal ? FMHA::CausalFromTopLeft : FMHA::NoCustomMask;
      break;
    case XformerQKVFormat::CONTINUOUS:
      pto.num_queries = pfrom.seqlen_q;
      pto.num_keys = pfrom.seqlen_k;
      pto.q_strideH = pfrom.phead;
      pto.k_strideH = pfrom.phead;
      pto.v_strideH = pfrom.phead;
      pto.q_strideM = (pfrom.nhead) * pfrom.phead;
      pto.k_strideM = (pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideM = (pfrom.nhead_kv) * pfrom.phead;
      pto.o_strideM = pfrom.nhead * pfrom.phead;
      pto.q_strideB = pfrom.seqlen_q * (pfrom.nhead) * pfrom.phead;
      pto.k_strideB = pfrom.seqlen_k * (pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideB = pfrom.seqlen_k * (pfrom.nhead_kv) * pfrom.phead;
      pto.custom_mask_type =
          pfrom.causal ? FMHA::CausalFromTopLeft : FMHA::NoCustomMask;
      break;
    case XformerQKVFormat::MIX:
      pto.num_queries = pfrom.seqlen_q;
      pto.num_keys = pfrom.seqlen_k;
      pto.q_strideH = pfrom.phead;
      pto.k_strideH = pfrom.phead;
      pto.v_strideH = pfrom.phead;
      pto.q_strideM = (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.k_strideM = (pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideM = (pfrom.nhead_kv) * pfrom.phead;
      pto.o_strideM = pfrom.nhead * pfrom.phead;
      pto.q_strideB =
          pfrom.seqlen_q * (pfrom.nhead + 2 * pfrom.nhead_kv) * pfrom.phead;
      pto.k_strideB = pfrom.seqlen_k * (pfrom.nhead_kv) * pfrom.phead;
      pto.v_strideB = pfrom.seqlen_k * (pfrom.nhead_kv) * pfrom.phead;
      pto.custom_mask_type =
          pfrom.causal ? FMHA::CausalFromBottomRight : FMHA::NoCustomMask;
      break;
    default:
      throw std::runtime_error("Unknown QKV format");
      break;
  }
}

#define XFORMER_PREFILL_ATTENTION_IMPL_A(CT, CA)                              \
  template <>                                                                 \
  AsStatus xformer_prefill_attention_dispatch<CT, CA, 64, 64, true>(          \
      const xformer_t&, CT*, CT*, CT*, CT*, void*, float,                     \
      const cudaStream_t&);                                                   \
  template <>                                                                 \
  size_t                                                                      \
  xformer_prefill_attention_workspace_inbytes_dispatch<CT, CA, 64, 64, true>( \
      const xformer_t&);
#define XFORMER_PREFILL_ATTENTION_IMPL_B(CT, CA)                               \
  template <>                                                                  \
  AsStatus xformer_prefill_attention_dispatch<CT, CA, 32, 128, true>(          \
      const xformer_t&, CT*, CT*, CT*, CT*, void*, float,                      \
      const cudaStream_t&);                                                    \
  template <>                                                                  \
  size_t                                                                       \
  xformer_prefill_attention_workspace_inbytes_dispatch<CT, CA, 32, 128, true>( \
      const xformer_t&);
#define XFORMER_PREFILL_ATTENTION_IMPL_C(CT, CA)                               \
  template <>                                                                  \
  AsStatus xformer_prefill_attention_dispatch<CT, CA, 32, 128, false>(         \
      const xformer_t&, CT*, CT*, CT*, CT*, void*, float,                      \
      const cudaStream_t&);                                                    \
  template <>                                                                  \
  size_t xformer_prefill_attention_workspace_inbytes_dispatch<CT, CA, 32, 128, \
                                                              false>(          \
      const xformer_t&);

template <typename CType, typename CArch, int kBlockQuery, int kBlockKey,
          int kMaxK>
size_t xformer_prefill_attention_workspace_inbytes_dispatch(
    const xformer_t& param) {
  using FMHA = AttentionKernel<CType, CArch, true, kBlockQuery, kBlockKey,
                               kMaxK, false, false>;
  if (FMHA::kNeedsOutputAccumulatorBuffer) {
    return param.batch * param.seqlen_k * param.nhead * param.phead *
           sizeof(typename FMHA::output_accum_t);
  }
  return 0;
}

template <typename CType, typename CArch, int kBlockQuery, int kBlockKey,
          int kMaxK>
AsStatus xformer_prefill_attention_dispatch(const xformer_t& param, CType* qptr,
                                            CType* kptr, CType* vptr,
                                            CType* output, void* workspace,
                                            float alpha,
                                            const cudaStream_t& stream) {
  using FMHA = AttentionKernel<CType, CArch, true, kBlockQuery, kBlockKey,
                               kMaxK, false, false>;
  constexpr auto kernel_fn = attention_kernel_batched_impl<FMHA>;
  size_t smem_inbytes = sizeof(typename FMHA::SharedStorage);
  if (smem_inbytes > 0xc000) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_inbytes);
  }

  typename FMHA::Params fmhap;
  xformer_prefill_attention_param_cast<CType, FMHA>(param, fmhap);
  fmhap.scale = alpha;
  fmhap.query_ptr = qptr;
  fmhap.key_ptr = kptr;
  fmhap.value_ptr = vptr;
  fmhap.output_ptr = output;

  if (FMHA::kNeedsOutputAccumulatorBuffer) {
    fmhap.output_accum_ptr =
        reinterpret_cast<typename FMHA::output_accum_t*>(workspace);
  }
  if (!FMHA::check_supported(fmhap)) {
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  kernel_fn<<<fmhap.getBlocksGrid(), fmhap.getThreadsGrid(), smem_inbytes,
              stream>>>(fmhap);
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace cuda
}  // namespace allspark
