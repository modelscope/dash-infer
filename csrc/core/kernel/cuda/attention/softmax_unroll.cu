/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_unroll.cu
 */

/*
 * medium level kernel for both performance
 * and balance last-dim alignment.
 */
#include "attention_utils.cuh"
#include "softmax_utils.cuh"

namespace allspark {
namespace cuda {
namespace attention {
namespace softmax {

/*
 * kernel last-dim is not alignemn and not support packing io.
 */
template <class MaskType, class SoftmaxType,
          int32_t RegPT = 8,    // Regs per thread
          int32_t Block = 128,  // Block 128 / 512
          typename CPT = float>
__global__ void softmax_non_aligned_kernel(
    typename MaskType::Params mask_param,
    typename SoftmaxType::Params softmax_param) {
  constexpr CPT invalid_value = static_cast<CPT>(invalid_const);
  constexpr IDX nwarp = Block / utils::warp_size;

  IDX bidx = blockIdx.x;
  SoftmaxType softmax(softmax_param, bidx);
  MaskType mask(mask_param, bidx);

  CPT regs[RegPT];
  CPT tidx_max = invalid_value;
  CPT tidx_sum = static_cast<CPT>(0.f);
  CPT tidx_div = static_cast<CPT>(0.f);

  if (softmax.is_valid_block()) {
// load and find max
#pragma unroll
    for (IDX ur = 0; ur < RegPT; ur++) {
      IDX aidx = ur * Block + threadIdx.x;
      regs[ur] = softmax.is_valid_align(aidx)
                     ? softmax.scale *
                           static_cast<CPT>(softmax.in[softmax.index(aidx)])
                     : invalid_value;
      regs[ur] = mask.update_with_mask(regs[ur], aidx);
      // if (softmax.is_valid_align(aidx)) printf("G[%3d]T[%3d] a[%3d] load data
      // = %2.3f\n",
      //     blockIdx.x, threadIdx.x, aidx, regs[ur]);
      tidx_max = max(tidx_max, regs[ur]);
    }
    tidx_max = utils::ReduceBlock<utils::MaxOp, CPT, nwarp>(tidx_max);
// if (threadIdx.x == 0)   printf("[%3d] tidx_max=%f\n", blockIdx.x, tidx_max);

// calculate sum and div
#pragma unroll
    for (IDX ur = 0; ur < RegPT; ur++) {
      IDX aidx = ur * Block + threadIdx.x;
      float exp_diff =
          softmax.is_valid_align(aidx) ? expf(float(regs[ur] - tidx_max)) : 0.f;
      // if (softmax.is_valid_align(aidx)) printf("G[%3d]T[%3d] a[%3d] %2.3f =
      // expf(%2.3f) = expf(%2.3f - %2.3f)\n",
      //     blockIdx.x, threadIdx.x, aidx, exp_diff, regs[ur] - tidx_max,
      //     regs[ur], tidx_max);
      if (!softmax.fuse_log) regs[ur] = exp_diff;
      tidx_sum += exp_diff;
    }
    tidx_sum = utils::ReduceBlock<utils::SumOp, CPT, nwarp>(tidx_sum);
    if (softmax.fuse_log)
      tidx_div = static_cast<CPT>(logf(static_cast<float>(tidx_sum)));
    else
      tidx_div =
          static_cast<CPT>(__fdividef(1.f, static_cast<float>(tidx_sum)));
// if (threadIdx.x == 0)   printf("[%3d] tidx_sum=%f, tidx_div=%f\n",
// blockIdx.x, tidx_sum, tidx_div);

// store
#pragma unroll
    for (IDX ur = 0; ur < RegPT; ur++) {
      if (softmax.fuse_log)
        regs[ur] = regs[ur] - tidx_max - tidx_div;
      else
        regs[ur] = regs[ur] * tidx_div;
      IDX aidx = ur * Block + threadIdx.x;
      if (softmax.is_valid_align(aidx)) {
        softmax.out[softmax.index(aidx)] =
            static_cast<typename SoftmaxType::Ptr_t>(regs[ur]);
      }
    }
  }
}

template <typename FT, bool FuseLog>
struct softmax_unroll_nomask_dispatch {
  static constexpr int32_t kregs = 8;
  static constexpr int32_t kblock = 512;
  bool is_param_valid(float alpha, int data_batch, int data_xseql,
                      int data_align, bool enable_attn_logn, int base_logn) {
    return data_align <= kregs * kblock;
  }
  void operator()(cudaStream_t stream, const FT* input,
                  FT* output,  //  [batch, xseql, align]
                  float alpha, int data_batch, int data_xseql, int data_align,
                  bool enable_attn_logn, int base_logn) {
    int32_t kgrid = data_batch * data_xseql;
    using CPT = float;
    using MaskType = NoMask<float, CPT, kregs>;
    typename MaskType::Params mask_param;
    if (enable_attn_logn) {
      typename AttnLognSoftmax<FT, FuseLog>::Params softmax_param = {
          input, output,     alpha,      data_batch,
          1,     data_xseql, data_align, base_logn};
      // printf("softmax_non_aligned_kernel<\n\tMask=%s,\n\tSoftmax=%s,\n\tkreg=%d,
      // kblock=%d> <<<%#d, %3d>>>\n",
      //         mask_param.toString().c_str(),
      //         softmax_param.toString().c_str(), kregs, kblock, kgrid,
      //         kblock);
      softmax_non_aligned_kernel<MaskType, AttnLognSoftmax<FT, FuseLog>, kregs,
                                 kblock, CPT>
          <<<kgrid, kblock, 0, stream>>>(mask_param, softmax_param);
    } else {
      typename DefaultSoftmax<FT, FuseLog>::Params softmax_param = {
          input, output, alpha, data_batch * data_xseql, data_align};
      // printf("softmax_non_aligned_kernel<\n\tMask=%s,\n\tSoftmax=%s,\n\tkreg=%d,
      // kblock=%d> <<<%#d, %3d>>>\n",
      //         mask_param.toString().c_str(),
      //         softmax_param.toString().c_str(), kregs, kblock, kgrid,
      //         kblock);
      softmax_non_aligned_kernel<MaskType, DefaultSoftmax<FT, FuseLog>, kregs,
                                 kblock, CPT>
          <<<kgrid, kblock, 0, stream>>>(mask_param, softmax_param);
    }
  }
};  // softmax_unroll_nomask_dispatch

// no fuse log. support logn.
template <typename FT, typename MT = float,
          MaskMode Mode = MaskMode::MaskDirectAdd>
struct softmax_unroll_3dmask_dispatch {
  static constexpr int32_t kregs = 8;
  static constexpr int32_t kblock = 512;
  bool is_param_valid(float alpha, int data_batch, int data_xseql,
                      int data_nhead, int data_align, int mask_batch,
                      int mask_xsql, int mask_align, bool mask_decoder,
                      bool attn_logn, int base_logn) {
    return data_align <= kregs * kblock &&  // enough regs for compute.
           data_batch % mask_batch == 0;    // make sure batch beam is valid.
  }
  void operator()(cudaStream_t stream, const FT* input, FT* output,
                  const MT* masks, float alpha, int data_batch, int data_xseql,
                  int data_nhead, int data_align, int mask_batch,
                  int mask_xseql, int mask_align, bool mask_decoder,
                  bool attn_logn, int base_logn) {
    int32_t kgrid = data_batch * data_xseql * data_nhead;
    using CPT = float;
    using MaskType = DefaultMask<MT, CPT, kregs, Mode>;
    typename MaskType::Params mask_param = {
        masks,      mask_decoder, mask_batch,
        mask_xseql, mask_align,   data_batch / mask_batch,
        1,          data_nhead};  // batch with beam, mask not.
    if (attn_logn || base_logn <= 0) {
      using SoftmaxType = AttnLognSoftmax<FT, false>;
      typename SoftmaxType::Params softmax_param = {
          input,      output,     alpha,      data_batch,
          data_xseql, data_nhead, data_align, base_logn};
      // printf("softmax_non_aligned_kernel<\n\tMask=%s,\n\tSoftmax=%s,\n\tkreg=%d,
      // kblock=%d> <<<%#d, %3d>>>\n",
      //         mask_param.toString().c_str(),
      //         softmax_param.toString().c_str(), kregs, kblock, kgrid,
      //         kblock);
      softmax_non_aligned_kernel<MaskType, SoftmaxType, kregs, kblock, CPT>
          <<<kgrid, kblock, 0, stream>>>(mask_param, softmax_param);
    } else {
      using SoftmaxType = DefaultSoftmax<FT, false>;
      typename SoftmaxType::Params softmax_param = {
          input, output, alpha, data_batch * data_xseql, data_align};
      // printf("softmax_non_aligned_kernel<\n\tMask=%s,\n\tSoftmax=%s,\n\tkreg=%d,
      // kblock=%d> <<<%#d, %3d>>>\n",
      //         mask_param.toString().c_str(),
      //         softmax_param.toString().c_str(), kregs, kblock, kgrid,
      //         kblock);
      softmax_non_aligned_kernel<MaskType, SoftmaxType, kregs, kblock, CPT>
          <<<kgrid, kblock, 0, stream>>>(mask_param, softmax_param);
    }
  }
};  // softmax_unroll_3dmask_dispatch

}  // namespace softmax
}  // namespace attention

bool softmax_unroll_nomask_valid(cudaDataType_t dt, float alpha, int data_batch,
                                 int data_xseql, int data_align,
                                 bool enable_attn_logn, int base_logn,
                                 bool enable_fuse_log) {
  // dispatch
  if (dt == cudaDataType_t::CUDA_R_32F) {
    if (enable_fuse_log) {
      return attention::softmax::softmax_unroll_nomask_dispatch<float, true>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_nomask_dispatch<float, false>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    }
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    if (enable_fuse_log) {
      return attention::softmax::softmax_unroll_nomask_dispatch<half, true>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_nomask_dispatch<half, false>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    if (enable_fuse_log) {
      return attention::softmax::softmax_unroll_nomask_dispatch<
                 __hie_buildin::bfloat16, true>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_nomask_dispatch<
                 __hie_buildin::bfloat16, false>()
          .is_param_valid(alpha, data_batch, data_xseql, data_align,
                          enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_BF16
  return false;
}

void softmax_unroll_nomask(cudaStream_t stream, cudaDataType_t dt,
                           const void* input,
                           void* output,  //  [batch, xseql, align]
                           float alpha, int data_batch, int data_xseql,
                           int data_align, bool enable_attn_logn, int base_logn,
                           bool enable_fuse_log) {
  // dispatch
  if (dt == cudaDataType_t::CUDA_R_32F) {
    if (enable_fuse_log) {
      attention::softmax::softmax_unroll_nomask_dispatch<float, true>()(
          stream, static_cast<const float*>(input), static_cast<float*>(output),
          alpha, data_batch, data_xseql, data_align, enable_attn_logn,
          base_logn);
    } else {
      attention::softmax::softmax_unroll_nomask_dispatch<float, false>()(
          stream, static_cast<const float*>(input), static_cast<float*>(output),
          alpha, data_batch, data_xseql, data_align, enable_attn_logn,
          base_logn);
    }
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    if (enable_fuse_log) {
      attention::softmax::softmax_unroll_nomask_dispatch<half, true>()(
          stream, static_cast<const half*>(input), static_cast<half*>(output),
          alpha, data_batch, data_xseql, data_align, enable_attn_logn,
          base_logn);
    } else {
      attention::softmax::softmax_unroll_nomask_dispatch<half, false>()(
          stream, static_cast<const half*>(input), static_cast<half*>(output),
          alpha, data_batch, data_xseql, data_align, enable_attn_logn,
          base_logn);
    }
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    if (enable_fuse_log) {
      attention::softmax::softmax_unroll_nomask_dispatch<
          __hie_buildin::bfloat16, true>()(
          stream, static_cast<const __hie_buildin::bfloat16*>(input),
          static_cast<__hie_buildin::bfloat16*>(output), alpha, data_batch,
          data_xseql, data_align, enable_attn_logn, base_logn);
    } else {
      attention::softmax::softmax_unroll_nomask_dispatch<
          __hie_buildin::bfloat16, false>()(
          stream, static_cast<const __hie_buildin::bfloat16*>(input),
          static_cast<__hie_buildin::bfloat16*>(output), alpha, data_batch,
          data_xseql, data_align, enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_BF16
}

void softmax_unroll_3dmask(cudaStream_t stream, cudaDataType_t dt,
                           const void* input,
                           void* output,       //  [batch, xseql, nhead, align]
                           const void* masks,  //  [batch, xseql,        align]
                           float alpha, int data_batch, int data_xseql,
                           int data_nhead, int data_align, int mask_batch,
                           int mask_xseql, int mask_align,
                           bool decoder_layout_mask, bool mask_with_10,
                           bool enable_attn_logn, int base_logn) {
  // check mask is nullptr
  if (masks == nullptr && softmax_unroll_nomask_valid(
                              dt, alpha, data_batch, data_xseql * data_nhead,
                              data_align, enable_attn_logn, base_logn, false)) {
    softmax_unroll_nomask(stream, dt, input, output, alpha, data_batch,
                          data_xseql * data_nhead, data_align, enable_attn_logn,
                          base_logn, false);
    return;
  }

  // dispatch
  using MT = float;
  if (dt == cudaDataType_t::CUDA_R_32F) {
    using FT = float;
    if (mask_with_10) {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith10>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    using FT = half;
    if (mask_with_10) {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith10>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    using FT = __hie_buildin::bfloat16;
    if (mask_with_10) {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith10>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      attention::softmax::softmax_unroll_3dmask_dispatch<
          FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()(
          stream, static_cast<const FT*>(input), static_cast<FT*>(output),
          static_cast<const MT*>(masks), alpha, data_batch, data_xseql,
          data_nhead, data_align, mask_batch, mask_xseql, mask_align,
          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_BF16
}

bool softmax_unroll_3dmask_valid(cudaDataType_t dt, float alpha, int data_batch,
                                 int data_xseql, int data_nhead, int data_align,
                                 int mask_batch, int mask_xseql, int mask_align,
                                 bool decoder_layout_mask, bool mask_with_10,
                                 bool enable_attn_logn, int base_logn) {
  // dispatch
  using MT = float;
  if (dt == cudaDataType_t::CUDA_R_32F) {
    using FT = float;
    if (mask_with_10) {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith10>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#if ENABLE_FP16
  if (dt == cudaDataType_t::CUDA_R_16F) {
    using FT = half;
    if (mask_with_10) {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith10>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_FP16
#if ENABLE_BF16
  if (dt == cudaDataType_t::CUDA_R_16BF) {
    using FT = __hie_buildin::bfloat16;
    if (mask_with_10) {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith10>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    } else {
      return attention::softmax::softmax_unroll_3dmask_dispatch<
                 FT, MT, attention::softmax::MaskMode::MaskWith0Inf>()
          .is_param_valid(alpha, data_batch, data_xseql, data_nhead, data_align,
                          mask_batch, mask_xseql, mask_align,
                          decoder_layout_mask, enable_attn_logn, base_logn);
    }
  }
#endif  // ENABLE_BF16
  return false;
}

}  // namespace cuda
}  // namespace allspark
