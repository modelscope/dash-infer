/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_score_gemv.cu
 */

#include "attn_kv_cache.hpp"
#include "kernel_utils.cuh"

// TODO(zhangyufei): optimize later
// per warp process single phead.
// a16w8/a16w4 gemv for first gemv
// qkv              [batch,     1, nhead, phead] x
//      stride                  2
//      offset                  0,
// kcache           [batch, nhead, xseql, phead] +
//      stride                     cache,
//      offset                         0,
// kparam           [batch, nhead, xseql]
//      stride                     cache,
//      offset                         0,
// mask-default     [batch,     1,     1, xseql] ->
// pos-embedding    [batch,     1,     1, xseql]
//      stride                     inlen, inlen
//      offset                   xseql-1,     0
// score            [batch, nhead,     1, xseql]

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
namespace score_gemv {
constexpr int32_t warp_num = 4;
constexpr int32_t coorp_tid = 4;
constexpr int32_t block_size = warp_num * utils::warp_size;

enum mask_mode { BATCH_XSEQL, BATCH_INLEN_INLEN, BATCH_NHEAD_XSEQL };

template <typename TYPE, mask_mode MASK = BATCH_XSEQL>
struct mask_indexing {
  int32_t nhead, xseql, inlen;
  __device__ __forceinline__ TYPE operator()(const TYPE* ptr, int32_t bidx,
                                             int32_t nidx, int32_t xidx) {
    int32_t index = bidx * xseql + xidx;
    if (ptr)
      return ptr[index];
    else
      return 0.;
  }
};

template <typename TYPE>
struct mask_indexing<TYPE, BATCH_INLEN_INLEN> {
  int32_t nhead, xseql, inlen;
  __device__ __forceinline__ TYPE operator()(const TYPE* ptr, int32_t bidx,
                                             int32_t nidx, int32_t xidx) {
    int32_t index = bidx * inlen * inlen + (inlen - 1) * inlen +
                    xidx;  // see mha.cu decoder_softmax_with_mask, i have no
                           // idea with this.
    if (ptr && xidx < inlen)
      return (1 - ptr[index]) * -1e5;
    else
      return 0.;
  }
};

template <typename TYPE>
struct mask_indexing<TYPE, BATCH_NHEAD_XSEQL> {
  int32_t nhead, xseql, inlen;
  __device__ __forceinline__ TYPE operator()(const TYPE* ptr, int32_t bidx,
                                             int32_t nidx, int32_t xidx) {
    int32_t index = bidx * nhead * xseql + nidx * xseql + xidx;
    if (ptr)
      return ptr[index];
    else
      return 0.;
  }
};

template <typename FT, typename QT, int32_t QTPack, int32_t PACK,
          typename MT = float, typename ZT = float, typename CPT = float>
__global__ void batch_gemv_nt_AF_BQ_kernel(
    const FT* query_in_qkv, const MT* mask0, const MT* mask1, const QT* kcache,
    const ZT* kzero, const CPT* kscale, FT* output, float alpha, int32_t batch,
    int32_t nhead, int32_t phead, int32_t cache, int32_t xseql, int32_t inlen,
    u32div_t div_nhead, u32div_t div_xseql) {
  // load qkv
  constexpr utils::LdgHint FPHint = utils::LdgHint::NONE;
  constexpr utils::LdgHint I8Hint =
      utils::LdgHint::CACHE_NON_COHERENT;  // CS / NC-CS is better than CG, then
                                           // better than CA.
  using qkv_t = utils::packed_data<PACK, FT>;
  using w8k_t = utils::packed_data<PACK / QTPack, QT>;
  using cpk_t = utils::packed_data<QTPack, CPT>;
  mask_indexing<MT, BATCH_INLEN_INLEN> mask0_handle;
  mask0_handle.nhead = nhead;
  mask0_handle.xseql = xseql;
  mask0_handle.inlen = inlen;
  mask_indexing<MT, BATCH_NHEAD_XSEQL> mask1_handle;
  mask1_handle.nhead = nhead;
  mask1_handle.xseql = xseql;
  mask1_handle.inlen = inlen;

  int32_t sidx = (blockIdx.x * blockDim.x + threadIdx.x) / coorp_tid;
  auto xseql_divmod = div_xseql.divmod(sidx);
  int32_t xidx = xseql_divmod.mod;
  auto nhead_divmod = div_nhead.divmod(xseql_divmod.div);
  int32_t nidx = nhead_divmod.mod;
  int32_t bidx = nhead_divmod.div;

  if (bidx < batch && nidx < nhead && xidx < xseql) {
    // load bias or mask
    CPT score = static_cast<CPT>(mask0_handle(mask0, bidx, nidx, xidx)) +
                static_cast<CPT>(mask1_handle(mask1, bidx, nidx, xidx));

    // load k-param
    int32_t qidx = bidx * nhead * cache + nidx * cache + xidx;
    CPT kz = static_cast<CPT>(utils::LdgFPCache<ZT, FPHint>()(kzero + qidx));
    CPT ks = utils::LdgFPCache<CPT, FPHint>()(kscale + qidx);

    CPT sum[PACK] = {0.f};
    int32_t part = phead / (coorp_tid * PACK);
    for (int32_t loop = 0; loop < part; loop++) {
      int32_t pidx = loop * coorp_tid * PACK + (threadIdx.x % coorp_tid) * PACK;
      // load qkv
      int32_t qkv_index =
          bidx * 3 * nhead * phead + 0 * nhead * phead + nidx * phead + pidx;
      qkv_t qkv_pack =
          reinterpret_cast<const qkv_t*>(query_in_qkv + qkv_index)[0];
      CPT fq[PACK];
#pragma unroll
      for (int32_t pack = 0; pack < PACK; pack++) {
        fq[pack] = static_cast<CPT>(qkv_pack.pack[pack]);
      }

      // load kcache
      int32_t w8k_index = (bidx * nhead * cache * phead + nidx * cache * phead +
                           xidx * phead + pidx) /
                          QTPack;
      utils::packed_data<PACK / QTPack, int8_t> i8k_t =
          utils::LdgI8Cache<PACK / QTPack, I8Hint>()(kcache + w8k_index);
      w8k_t w8 = reinterpret_cast<w8k_t&>(i8k_t);
      CPT fk[PACK];
#pragma unroll
      for (int32_t pack = 0; pack < PACK / QTPack; pack++) {
        cpk_t fk_slice =
            utils::UnpackW8AndDequant<CPT, QT, QTPack>(w8.pack[pack], ks, kz);
        reinterpret_cast<cpk_t*>(fk)[pack] = fk_slice;
      }

// sum
#pragma unroll
      for (int32_t pack = 0; pack < PACK; pack++) {
        sum[pack] += fq[pack] * fk[pack];
      }
    }

// final sum.
// float debug = 0.f;
#pragma unroll
    for (int32_t pack = 0; pack < PACK; pack++) {
      // reduce thread group.
      sum[pack] = utils::ReduceThread<utils::SumOp, CPT, coorp_tid>(sum[pack]);
      score += alpha * sum[pack];
      // debug += sum[pack];
    }

    // store
    if (threadIdx.x % coorp_tid == 0) {
      output[sidx] = static_cast<FT>(score);
    }
  }
}

template <typename FT, typename QT>
struct gemv_with_position_embedding_impl {
  // qkv          [batch, 3, nhead, phead]
  // mask_bii     [batch, inlen, inlen]
  // mask_bnx     [batch, nhead, xseql]
  // kcache       [batch, nhead, cache, phead]
  // output       [batch, nhead, xseql]
  void operator()(cudaStream_t stream, const void* qkv, const float* mask_bii,
                  const float* mask_bnx, const QT* kcache, const float* zero,
                  const float* scale, void* output, float alpha, int32_t batch,
                  int32_t nhead, int32_t phead, int32_t cache, int32_t xseql,
                  int32_t inlen, QuantType quant_type) {
    constexpr int32_t kPack = utils::pack_infer<FT>::PACK;
    constexpr int32_t kBlock = block_size;
    int32_t kGrid = utils::cal_ceil(batch * nhead * xseql, kBlock / coorp_tid);
    u32div_t div_nhead(nhead);
    u32div_t div_xseql(xseql);

    const FT* tqkv = reinterpret_cast<const FT*>(qkv);
    FT* toutput = reinterpret_cast<FT*>(output);

    if (phead % (kPack * coorp_tid) != 0) {
      if (quant_type == QuantType::INT8) {
        constexpr int32_t QTPack = 1;
        constexpr int32_t nPack = 1;
        // TODO(zhangyufei): print warning for lower performance
        // printf("batch_gemv_nt_AF_BQ_kernel<kPack=%d, BATCH_INLEN_INLEN,
        // BATCH_NHEAD_XSEQL><<<%3d, %3d>>>(batch=%d, nhead=%d, phead=%d,
        // cache=%d, xseeql=%d)\n",
        //         nPack, kGrid, kBlock, batch, nhead, phead, cache, xseql);
        batch_gemv_nt_AF_BQ_kernel<FT, QT, QTPack, nPack, float, float, float>
            <<<kGrid, kBlock, 0, stream>>>(
                tqkv, mask_bii, mask_bnx, kcache, zero, scale, toutput, alpha,
                batch, nhead, phead, cache, xseql, inlen, div_nhead, div_xseql);
        if (phead % (nPack * coorp_tid) != 0) {
          LOG(ERROR) << "not support per-head size (" << phead
                     << ") for i8-cache attention.";
          return;
        }
      } else if (quant_type == QuantType::UINT4) {
        constexpr int32_t QTPack = 2;
        int32_t nPack = kPack / 2;
        while (phead % (kPack * coorp_tid) != 0 && nPack >= 2) {
          nPack /= 2;
        }
        switch (nPack) {
          case 4:
            batch_gemv_nt_AF_BQ_kernel<FT, QT, QTPack, 4, float, float, float>
                <<<kGrid, kBlock, 0, stream>>>(
                    tqkv, mask_bii, mask_bnx, kcache, zero, scale, toutput,
                    alpha, batch, nhead, phead, cache, xseql, inlen, div_nhead,
                    div_xseql);
            break;
          case 2:
            batch_gemv_nt_AF_BQ_kernel<FT, QT, QTPack, 2, float, float, float>
                <<<kGrid, kBlock, 0, stream>>>(
                    tqkv, mask_bii, mask_bnx, kcache, zero, scale, toutput,
                    alpha, batch, nhead, phead, cache, xseql, inlen, div_nhead,
                    div_xseql);
            break;
          default:
            LOG(ERROR) << "not support per-head size (" << phead
                       << ") for u4-cache attention.";
            return;
        }
      }
    } else {
      // printf("batch_gemv_nt_AF_BQ_kernel<kPack=%d, BATCH_INLEN_INLEN,
      // BATCH_NHEAD_XSEQL><<<%3d, %3d>>>(batch=%d, nhead=%d, phead=%d,
      // cache=%d, xseeql=%d)\n",
      //         kPack, kGrid, kBlock, batch, nhead, phead, cache, xseql);
      if (quant_type == QuantType::INT8) {
        constexpr int32_t QTPack = 1;
        batch_gemv_nt_AF_BQ_kernel<FT, QT, QTPack, kPack, float, float, float>
            <<<kGrid, kBlock, 0, stream>>>(
                tqkv, mask_bii, mask_bnx, kcache, zero, scale, toutput, alpha,
                batch, nhead, phead, cache, xseql, inlen, div_nhead, div_xseql);
      } else if (quant_type == QuantType::UINT4) {
        constexpr int32_t QTPack = 2;
        batch_gemv_nt_AF_BQ_kernel<FT, QT, QTPack, kPack, float, float, float>
            <<<kGrid, kBlock, 0, stream>>>(
                tqkv, mask_bii, mask_bnx, kcache, zero, scale, toutput, alpha,
                batch, nhead, phead, cache, xseql, inlen, div_nhead, div_xseql);
      }
    }
  }
};

}  // namespace score_gemv

#define SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(FT, QT)                         \
  template <>                                                                 \
  void score_gemv_w8_position_embedding<FT, QT>(                              \
      cudaStream_t stream, const FT* qkv, const float* mask,                  \
      const float* position_embedding, const QT* kc, const float* kz,         \
      const float* ks, FT* output, float alpha, int32_t batch, int32_t nhead, \
      int32_t phead, int32_t cache, int32_t xseql, int32_t inlen,             \
      QuantType quant_type) {                                                 \
    score_gemv::gemv_with_position_embedding_impl<FT, QT>()(                  \
        stream, qkv, mask, position_embedding, kc, kz, ks, output, alpha,     \
        batch, nhead, phead, cache, xseql, inlen, quant_type);                \
  }

SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(float, int8_t);
SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(float, uint8_t);
#if ENABLE_FP16
SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(half_t, int8_t);
SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(half_t, uint8_t);
#endif  // ENABLE_FP16
#if ENABLE_BF16
SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(__hie_buildin::bfloat16, int8_t);
SCORE_GEMV_W8_POSITION_EMBEDDING_IMPL(__hie_buildin::bfloat16, uint8_t);
#endif  // ENABLE_BF16

}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
