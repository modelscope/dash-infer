/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_fallback.cu
 */

/*
 * basic fallback-softmax kernel
 * please keep this softmax independent
 */
#include "attention_utils.cuh"

namespace allspark {
namespace cuda {
namespace attention {
namespace softmax {

/*
 * naive kernel for kernel selection fallback and test.
 * not optimized, and use different impl for all situations.
 * assume we have four dimensions: [hi, md, lo, rd],
 * where we only do softmax at last dimension (rd).
 * and other 3 dims (hi, md, lo) are all batched dimensions.
 */
template <typename FT, typename MT = float, int32_t WARPN = 4, int32_t UR = 4>
__global__ void softmax_4d_fallback_warp_kernel(
    const FT* input, FT* output,  //  [datah, datam, datal, align]
    const MT* masks,              //  [maskh, maskm, maskl, maska]
    float alpha,                  //  scaling factor, a.k.a 1/sqrt(d)
    int32_t datah, u32div_t udivm, u32div_t udivl, int32_t align,
    int32_t aloop,  // loop on align dim
    int32_t maskh, int32_t maskm, int32_t maskl, int32_t maska,
    /* features flag and params below this line */
    bool mask_decoder,  // mask as [beach / beams, 1, inlen], where dim[-2]
                        // stride = inlen, use xseql - 1 value.
    bool mask_with_10,  // mask with 1 or 0
    bool log_softmax,   // fuse log after softmax
    bool attn_logn, int32_t lognm,
    int32_t lognl  // query logn scale before softmax, logn only on single dim,
                   // and the other dim should be 0
) {
  constexpr float invalid = -1e15;
  int32_t warp = threadIdx.x / utils::warp_size;
  int32_t lane = threadIdx.x % utils::warp_size;
  auto divl = udivl.divmod(blockIdx.x * WARPN + warp);
  auto divm = udivm.divmod(divl.div);
  int32_t hidx = divm.div;  // high
  int32_t midx = divm.mod;  // middle
  int32_t lidx = divl.mod;  // low
  int32_t datam = udivm.d_;
  int32_t datal = udivl.d_;
  int32_t hidm =
      hidx / (datah / maskh);  // if beam search, datah / maskh = beams
  int32_t midm = midx / (datam / maskm);
  int32_t lidm = lidx / (datal / maskl);
  if (mask_decoder) {
    hidm = hidm;  // decoder h dim is batch with beam.
    midm = maskm -
           1;  // decoder m dim is xseql, decoder style mask using last value.
    lidm = 0;  // decoder l dim is nhead
  }

  // check valid warp at h / m / l dims.
  bool valid_mask =
      masks != nullptr && hidm < maskh && midm < maskm && lidm < maskl;
  if (hidx >= datah || midx >= datam || lidx >= datal) return;
  // attn-logn. scale query-dim with log(index, base_model_length) if index >
  // base_model_length
  if (attn_logn && lognm != 0 && midx > lognm) {
    alpha = alpha * logf(midx) / logf(lognm);
  }
  if (attn_logn && lognl != 0 && lidx > lognl) {
    alpha = alpha * logf(lidx) / logf(lognl);
  }

  // find max
  float tidx_data[UR];  // store data
  float tidx_mask[UR];  // store mask as fp32
  float tidx_max = -INFINITY;
  // if aloop == 1, store data in tidx_data/mask, otherwise, reload them.
  for (int32_t lp = 0; lp < aloop; lp++) {
#pragma unroll
    for (int32_t ur = 0; ur < UR; ur++) {
      int32_t aidx = lp * UR * utils::warp_size + ur * utils::warp_size + lane;
      int32_t data_index = hidx * datam * datal * align + midx * datal * align +
                           lidx * align + aidx;
      int32_t mask_index = hidm * maskm * maskl * maska + midm * maskl * maska +
                           lidm * maska + aidx;
      tidx_data[ur] =
          aidx < align ? static_cast<float>(input[data_index]) : invalid;
      tidx_mask[ur] = aidx < maska && valid_mask ? masks[mask_index]
                      : mask_with_10
                          ? 1.f
                          : 0.f;  // default load as direct-add mask style. if
                                  // mask not given, keep valid.
      if (mask_with_10) tidx_mask[ur] = (1.f - tidx_mask[ur]) * invalid;
      tidx_data[ur] = alpha * tidx_data[ur] + tidx_mask[ur];
      tidx_max = tidx_max > tidx_data[ur] ? tidx_max : tidx_data[ur];
    }
  }
  tidx_max =
      utils::ReduceThread<utils::MaxOp, float, utils::warp_size>(tidx_max);
  // if (lane == 0) printf("h[%3d]m[%3d]l[%3d] warp[%d] tidx_max=%f\n", hidx,
  // midx, lidx, warp, tidx_max);

  // find expf sum.
  float expf_sum = 0.f;
  for (int32_t lp = 0; lp < aloop; lp++) {
#pragma unroll
    for (int32_t ur = 0; ur < UR; ur++) {
      // if loop > 1 reload data and mask.
      int32_t aidx = lp * UR * utils::warp_size + ur * utils::warp_size + lane;
      if (aloop > 1) {
        int32_t data_index = hidx * datam * datal * align +
                             midx * datal * align + lidx * align + aidx;
        int32_t mask_index = hidm * maskm * maskl * maska +
                             midm * maskl * maska + lidm * maska + aidx;
        tidx_data[ur] =
            aidx < align ? static_cast<float>(input[data_index]) : invalid;
        tidx_mask[ur] = aidx < maska && valid_mask ? masks[mask_index]
                        : mask_with_10
                            ? 1.f
                            : 0.f;  // default load as direct-add mask style. if
                                    // mask not given, keep valid.
        if (mask_with_10) tidx_mask[ur] = (1.f - tidx_mask[ur]) * invalid;
        tidx_data[ur] = alpha * tidx_data[ur] + tidx_mask[ur];
      }
      // calculate expf(value - max) for higher acc.
      float exp_diff = expf(tidx_data[ur] - tidx_max);
      // if (aidx < align) printf("h[%3d]m[%3d]l[%3d]a[%3d] %2.3f = expf(%2.3f)
      // = expf(%2.3f - %2.3f)\n",
      //     hidx, midx, lidx, aidx, exp_diff, tidx_data[ur] - tidx_max,
      //     tidx_data[ur], tidx_max);
      expf_sum += exp_diff;
      // if possible, store expf result back to tidx_data.
      // log-softmax require keep laod data no expf result.
      if (!log_softmax) tidx_data[ur] = exp_diff;
    }
  }
  expf_sum =
      utils::ReduceThread<utils::SumOp, float, utils::warp_size>(expf_sum);
  // if (lane == 0) printf("h[%3d]m[%3d]l[%3d] warp[%d] expf_sum=%f\n", hidx,
  // midx, lidx, warp, expf_sum);
  expf_sum = log_softmax ? logf(expf_sum + 1e-15) : 1.f / expf_sum;
  // if (lane == 0) printf("h[%3d]m[%3d]l[%3d] warp[%d] expf_div=%f\n", hidx,
  // midx, lidx, warp, expf_sum);

  // store softmax result
  for (int32_t lp = 0; lp < aloop; lp++) {
#pragma unroll
    for (int32_t ur = 0; ur < UR; ur++) {
      int32_t aidx = lp * UR * utils::warp_size + ur * utils::warp_size + lane;
      int32_t data_index = hidx * datam * datal * align + midx * datal * align +
                           lidx * align + aidx;
      int32_t mask_index = hidm * maskm * maskl * maska + midm * maskl * maska +
                           lidm * maska + aidx;
      // if loop > 1 reload data and mask.
      if (aloop > 1) {
        tidx_data[ur] =
            aidx < align ? static_cast<float>(input[data_index]) : invalid;
        tidx_mask[ur] = aidx < maska && valid_mask ? masks[mask_index]
                        : mask_with_10
                            ? 1.f
                            : 0.f;  // default load as direct-add mask style. if
                                    // mask not given, keep valid.
        if (mask_with_10) tidx_mask[ur] = (1.f - tidx_mask[ur]) * invalid;
        tidx_data[ur] = alpha * tidx_data[ur] + tidx_mask[ur];
        if (!log_softmax) tidx_data[ur] = expf(tidx_data[ur] - tidx_max);
      }
      tidx_data[ur] = log_softmax ? tidx_data[ur] - tidx_max - expf_sum
                                  : tidx_data[ur] * expf_sum;
      if (aidx < align) {
        output[data_index] = static_cast<FT>(tidx_data[ur]);
      }
    }
  }
}

template <typename FT, typename MT = float, bool Test = false>
struct softmax_4d_warp_dispatch {
  void operator()(cudaStream_t stream, const FT* input,
                  FT* output,       //  [datah, datam, datal, align]
                  const MT* masks,  //  [maskh, maskm, maskl, maska]
                  float alpha,      //  scaling factor, a.k.a 1/sqrt(d)
                  int32_t datah, int32_t datam, int32_t datal,
                  int32_t align,  // loop on align dim
                  int32_t maskh, int32_t maskm, int32_t maskl, int32_t maska,
                  /* features flag and params below this line */
                  bool mask_decoder,  // decoder style mask, maskl dim keep use
                                      // with maskl - 1 index.
                  bool mask_with_10,  // mask with 1 or 0
                  bool log_softmax,   // fuse log after softmax
                  bool attn_logn, int32_t lognm,
                  int32_t lognl  // query logn scale before softmax, logn only
                                 // on single dim, and the other dim should be 0
  ) {
    constexpr int32_t warpn = Test ? 1 : 4;
    constexpr int32_t unroll = Test ? 1 : 4;
    int32_t kblock = warpn * utils::warp_size;
    int32_t kgrid = utils::cal_ceil(datah * datam * datal, warpn);
    int32_t kloop = utils::cal_ceil(align, unroll * utils::warp_size);
    /*printf("softmax_4d_fallback_warp_kernel<WarpN=%d, UR=%d><<<kblock=%d,
    kgrid=%d>>>\n", warpn, unroll, kblock, kgrid); printf("\talpha=%f, kloop=%d.
    \n\tdatah=%d, datam=%d, datal=%d, align=%d, \n\tmaskh=%d, maskm=%d,
    maskl=%d, maska=%d,\n\t", alpha, kloop, datah, datam, datal, align, maskh,
    maskm, maskl, maska); if (mask_decoder)    printf("mask_decoder, "); if
    (mask_with_10)    printf("with_10, "); if (log_softmax) printf("log_softmax,
    "); if (attn_logn)       printf("lognm=%d, lognl=%d, ", lognm, lognl);
    printf(")\n");*/
    softmax_4d_fallback_warp_kernel<FT, MT, warpn, unroll>
        <<<kgrid, kblock, 0, stream>>>(
            input, output, masks, alpha, datah, u32div_t(datam),
            u32div_t(datal), align, kloop, maskh, maskm, maskl, maska,
            mask_decoder, mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
};

}  // namespace softmax
}  // namespace attention

void softmax_4d_fallback(
    cudaStream_t stream, cudaDataType_t dt, const void* input,
    void* output,       //  [datah, datam, datal, align]
    const void* masks,  //  [maskh, maskm, maskl, maska]
    float alpha,        //  scaling factor, a.k.a 1/sqrt(d)
    int32_t datah, int32_t datam, int32_t datal,
    int32_t align,  // loop on align dim
    int32_t maskh, int32_t maskm, int32_t maskl, int32_t maska,
    /* features flag and params below this line */
    bool mask_decoder,  // decoder style mask, maskl dim keep use with maskl - 1
                        // index.
    bool mask_with_10,  // mask with 1 or 0
    bool log_softmax,   // fuse log after softmax
    bool attn_logn, int32_t lognm,
    int32_t lognl  // query logn scale before softmax, logn only on single dim,
                   // and the other dim should be 0
) {
  if (dt == cudaDataType_t::CUDA_R_32F) {
    attention::softmax::softmax_4d_warp_dispatch<float, float>()(
        stream, (const float*)input, (float*)output, (const float*)masks, alpha,
        datah, datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    attention::softmax::softmax_4d_warp_dispatch<half, float>()(
        stream, (const half*)input, (half*)output, (const float*)masks, alpha,
        datah, datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    attention::softmax::softmax_4d_warp_dispatch<__hie_buildin::bfloat16,
                                                 float>()(
        stream, (const __hie_buildin::bfloat16*)input,
        (__hie_buildin::bfloat16*)output, (const float*)masks, alpha, datah,
        datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#endif  // ENABLE_BF16
}

void softmax_4d_test_only(
    cudaStream_t stream, cudaDataType_t dt, const void* input,
    void* output,       //  [datah, datam, datal, align]
    const void* masks,  //  [maskh, maskm, maskl, maska]
    float alpha,        //  scaling factor, a.k.a 1/sqrt(d)
    int32_t datah, int32_t datam, int32_t datal,
    int32_t align,  // loop on align dim
    int32_t maskh, int32_t maskm, int32_t maskl, int32_t maska,
    /* features flag and params below this line */
    bool mask_decoder,  // decoder style mask, maskl dim keep use with maskl - 1
                        // index.
    bool mask_with_10,  // mask with 1 or 0
    bool log_softmax,   // fuse log after softmax
    bool attn_logn, int32_t lognm,
    int32_t lognl  // query logn scale before softmax, logn only on single dim,
                   // and the other dim should be 0
) {
  if (dt == cudaDataType_t::CUDA_R_32F) {
    attention::softmax::softmax_4d_warp_dispatch<float, float, true>()(
        stream, (const float*)input, (float*)output, (const float*)masks, alpha,
        datah, datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    attention::softmax::softmax_4d_warp_dispatch<half, float, true>()(
        stream, (const half*)input, (half*)output, (const float*)masks, alpha,
        datah, datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    attention::softmax::softmax_4d_warp_dispatch<__hie_buildin::bfloat16, float,
                                                 true>()(
        stream, (const __hie_buildin::bfloat16*)input,
        (__hie_buildin::bfloat16*)output, (const float*)masks, alpha, datah,
        datam, datal, align, maskh, maskm, maskl, maska, mask_decoder,
        mask_with_10, log_softmax, attn_logn, lognm, lognl);
  }
#endif  // ENABLE_BF16
}

}  // namespace cuda
}  // namespace allspark
