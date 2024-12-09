/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_context_gemv.cu
 */

#include "attn_kv_cache.hpp"
#include "kernel_utils.cuh"

// TODO(zhangyufei): optimize later
// score            [batch, nhead,     1, xseql] - cache
// vcache           [batch, nhead, xseql, phead] - l2
//      stride                     cache,
//      offset                         0,
// vparam           [batch, nhead, xseql]
//      stride                     cache,
//      offset                         0,
// context          [batch,     1, nhead, phead]

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
namespace context_gemv {
// constexpr int32_t warp_num = 4;
// constexpr int32_t block_size = warp_num * utils::warp_size;

template <typename FT, typename QT, int32_t QTPack, int32_t WARP = 8,
          int32_t PACK = 8, int32_t UNROLL = 4, typename ZT = float,
          typename CPT = float>
__global__ void batch_gemv_AF_BQ_block_reduce_kernel(
    const FT* score, const QT* vcache, const ZT* vzero, const CPT* vscale,
    FT* context, int32_t batch, int32_t nhead, int32_t phead, int32_t cache,
    int32_t xseql, int32_t loop_xseql, u32div_t div_nhead, u32div_t div_phead) {
  using w8k_t = utils::packed_data<PACK / QTPack, QT>;
  using cpk_t = utils::packed_data<QTPack, CPT>;
  __shared__ CPT cross_warp_sum[WARP * utils::warp_size];

  auto batch_nhead_divmod = div_phead.divmod(blockIdx.x);
  int32_t pidg = batch_nhead_divmod.mod;
  auto batch_divmod = div_nhead.divmod(batch_nhead_divmod.div);
  int32_t bidx = batch_divmod.div;
  int32_t nidx = batch_divmod.mod;

  CPT regs[PACK] = {0.};
  // if grid valid.
  if (bidx < batch && nidx < nhead && (pidg + PACK) < phead) {
    // loop for xseql - reduce.
    for (int32_t lp = 0; lp < loop_xseql; lp++) {
#pragma unroll
      for (int32_t ur = 0; ur < UNROLL; ur++) {
        int32_t xidx = lp * UNROLL * WARP * utils::warp_size +
                       ur * WARP * utils::warp_size + threadIdx.x;
        if (xidx < xseql) {
          // zero and scale
          int32_t qidx = bidx * nhead * cache + nidx * cache + xidx;
          CPT fz = static_cast<CPT>(
              utils::LdgFPCache<ZT, utils::LdgHint::NONE>()(vzero + qidx));
          CPT fs =
              utils::LdgFPCache<CPT, utils::LdgHint::NONE>()(vscale + qidx);

          // score
          int32_t sidx = bidx * nhead * xseql + nidx * xseql + xidx;
          CPT sc = static_cast<CPT>(
              utils::LdgFPCache<FT, utils::LdgHint::NONE>()(score + sidx));

          // cache i8
          int32_t vidx = (bidx * nhead * cache * phead + nidx * cache * phead +
                          xidx * phead + pidg * PACK) /
                         QTPack;
          utils::packed_data<PACK / QTPack, int8_t> i8k_t =
              utils::LdgI8Cache<PACK / QTPack,
                                utils::LdgHint::CACHE_NON_COHERENT>()(vcache +
                                                                      vidx);
          w8k_t w8 = reinterpret_cast<w8k_t&>(i8k_t);

#pragma unroll
          for (int32_t pk = 0; pk < PACK / QTPack; pk++) {
            // unpack 8-bit and dequant
            cpk_t regs_tmp =
                utils::UnpackW8AndDequant<CPT, QT, QTPack>(w8.pack[pk], fs, fz);
#pragma unroll
            for (int32_t qt_pack = 0; qt_pack < QTPack; qt_pack++) {
              regs[pk * QTPack + qt_pack] += regs_tmp.pack[qt_pack] * sc;
            }
          }
        }
      }
    }

// reduce per-warp.
#pragma unroll
    for (int32_t pk = 0; pk < PACK; pk++) {
      regs[pk] =
          utils::ReduceThread<utils::SumOp, CPT, utils::warp_size>(regs[pk]);
    }

    // block level sum and store
    int32_t lane = threadIdx.x % utils::warp_size;
    int32_t warp = threadIdx.x / utils::warp_size;
// if (lane < PACK) {
//     int32_t pack = lane;
//     cross_warp_sum[warp * PACK + pack] = regs[pack];  // regs in every tid in
//     same warp should be same.
// }
#pragma unroll
    for (int32_t pk = 0; pk < PACK; pk++) {
      if (lane == 0) {
        cross_warp_sum[warp * PACK + pk] = regs[pk];
      }
    }
    __syncthreads();
    if (threadIdx.x < PACK) {
      int32_t pack = threadIdx.x;
      CPT sum = 0.f;
#pragma unroll
      for (int32_t wp = 0; wp < WARP; wp++) {
        sum += cross_warp_sum[wp * PACK + pack];
      }
      int32_t cidx = bidx * nhead * phead + nidx * phead + pidg * PACK + pack;
      context[cidx] = static_cast<FT>(sum);
    }
  }
}

template <typename FT, typename QT>
struct gemv_nt {
  void operator()(cudaStream_t stream, const void* score, const QT* vcache,
                  const float* vzero, const float* vscale, void* context,
                  int32_t batch, int32_t nhead, int32_t phead, int32_t cache,
                  int32_t xseql, QuantType quant_type) {
    // find valid warp, pack, unroll, loop
    std::vector<int32_t> legal_warp = {14, 10, 8, 6};
    constexpr int32_t kpack = utils::pack_infer<QT>::PACK;
    int32_t kunroll = 4;
    if (xseql % 3 == 0) kunroll = 3;
    if (xseql % 5 == 0) kunroll = 5;
    if (xseql % 6 == 0) kunroll = 6;
    int32_t kloop = 1;
    int32_t kwarp = legal_warp[0];
    while (kloop * kunroll * kwarp * utils::warp_size < xseql) {
      kloop *= 2;
    }
    for (auto warp : legal_warp) {
      if (kloop * kunroll * warp * utils::warp_size >= xseql) {
        kwarp = warp;
      }
    }
    int32_t kgrid = batch * nhead * utils::cal_ceil(phead, kpack);
    u32div_t div_nhead(nhead);
    u32div_t div_phead(utils::cal_ceil(phead, kpack));
    // printf("[mha-a16w8] infer G[%d] W[%d] P[%d] U[%d] L[%d]\n", kgrid, kwarp,
    // kpack, kunroll, kloop);

    // dispatch
#define BATCH_GEMV_A16W8_BLOCK_REDUCE(WP, PK, UR)                              \
  if (kwarp == (WP) && kpack == (PK) && kunroll == (UR)) {                     \
    /*printf("batch_gemv_AF_BQ_block_reduce_kernel <FT, WP=%d, PK=%d, UR=%d>", \
    kwarp, kpack, kunroll); printf("<<<%3d, %3d>>> (LP=%d, batch=%d, nhead=%d, \
    phead=%d, cache=%d, xseql=%d)\n", kgrid, kwarp * utils::warp_size, kloop,  \
    batch, nhead, phead, cache, xseql);*/                                      \
    if (quant_type == QuantType::INT8) {                                       \
      batch_gemv_AF_BQ_block_reduce_kernel<FT, QT, 1, WP, PK, UR, float,       \
                                           float>                              \
          <<<kgrid, kwarp * utils::warp_size, 0, stream>>>(                    \
              reinterpret_cast<const FT*>(score), vcache, vzero, vscale,       \
              reinterpret_cast<FT*>(context), batch, nhead, phead, cache,      \
              xseql, kloop, div_nhead, div_phead);                             \
      return;                                                                  \
    }                                                                          \
    if (quant_type == QuantType::UINT4) {                                      \
      batch_gemv_AF_BQ_block_reduce_kernel<FT, QT, 2, WP, PK, UR, float,       \
                                           float>                              \
          <<<kgrid, kwarp * utils::warp_size, 0, stream>>>(                    \
              reinterpret_cast<const FT*>(score), vcache, vzero, vscale,       \
              reinterpret_cast<FT*>(context), batch, nhead, phead, cache,      \
              xseql, kloop, div_nhead, div_phead);                             \
      return;                                                                  \
    }                                                                          \
  }
#define BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_UNROLL(WP, PK) \
  BATCH_GEMV_A16W8_BLOCK_REDUCE(WP, PK, 3);               \
  BATCH_GEMV_A16W8_BLOCK_REDUCE(WP, PK, 4);               \
  BATCH_GEMV_A16W8_BLOCK_REDUCE(WP, PK, 5);               \
  BATCH_GEMV_A16W8_BLOCK_REDUCE(WP, PK, 6);
#define BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL(WP) \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_UNROLL(WP, 4);            \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_UNROLL(WP, 8);            \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_UNROLL(WP, 16);
#define BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_WARP_PACK_AND_UNROLL() \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL(14);         \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL(10);         \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL(8);          \
  BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL(6);

    BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_WARP_PACK_AND_UNROLL();
#undef BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_WARP_PACK_AND_UNROLL
#undef BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK_AND_UNROLL
#undef BATCH_GEMV_A16W8_BLOCK_REDUCE_WITH_PACK
#undef BATCH_GEMV_A16W8_BLOCK_REDUCE

    // printf("[mha-a16w8] valid kernel not found.\n");
    return;
  }
};

}  // namespace context_gemv

#define CONTEXT_GEMV_W8_IMPL(FT, QT)                                       \
  template <>                                                              \
  void context_gemv_w8<FT, QT>(                                            \
      cudaStream_t stream, const FT* score, const QT* vcache,              \
      const float* vzero, const float* vscale, FT* context, int32_t batch, \
      int32_t nhead, int32_t phead, int32_t cache, int32_t xseql,          \
      QuantType quant_type) {                                              \
    context_gemv::gemv_nt<FT, QT>()(stream, score, vcache, vzero, vscale,  \
                                    context, batch, nhead, phead, cache,   \
                                    xseql, quant_type);                    \
  }

CONTEXT_GEMV_W8_IMPL(float, int8_t);
CONTEXT_GEMV_W8_IMPL(float, uint8_t);
#if ENABLE_FP16
CONTEXT_GEMV_W8_IMPL(half_t, int8_t);
CONTEXT_GEMV_W8_IMPL(half_t, uint8_t);
#endif  // ENABLE_FP16
#if ENABLE_BF16
CONTEXT_GEMV_W8_IMPL(__hie_buildin::bfloat16, int8_t);
CONTEXT_GEMV_W8_IMPL(__hie_buildin::bfloat16, uint8_t);
#endif

}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
