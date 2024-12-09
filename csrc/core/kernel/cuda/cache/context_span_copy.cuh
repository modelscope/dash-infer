/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    context_span_copy.cuh
 */

#pragma once

#include <cstdint>
#include <cuda/hie/cuda_intdivider.hpp>
#include <limits>
#include <stdexcept>

#include "../cache_quant/qcache.cuh"
#include "../utils/intrinsic.cuh"
#include "../utils/pack.cuh"
#include "cuda/cuda_kernel_span_cache.h"
#include "utility/check_cuda.h"

namespace allspark {
namespace cuda {
namespace cache {

namespace impl {

constexpr int WARP_SIZE = 32;

template <typename UT>
static inline constexpr bool IsPowerOf2(const UT& x) {
  return (x & (x - 1)) == 0;
}

}  // namespace impl

/**
 * @brief Slice, transpose, and copy to spans.
 *
 * S = 16/32, H = 128, S * H = 2048/4096
 * one warp per block, one block per head
 * n=32
 *
 * grid: (n_spans, n, S)
 *
 * [S, n, H] -> [n, S, H]
 * src: [seq_len, n, H]
 * src stride: {n * H, H, 1}
 * dst stride: {S * H, H, 1}
 */
template <int BLOCK, span::QuantMode QMODE, int SPAN, int HEADSIZE, typename T>
__global__ void __launch_bounds__(impl::WARP_SIZE) QuantSpanCopyKernel(
    typename qcache::QCacheConfig<QMODE, T>::QuantT* const* const spanPtrs,
    const T* const src, const uint32_t xSize) {
  static_assert(HEADSIZE >= BLOCK, "requires HEADSIZE >= BLOCK");
  static_assert(HEADSIZE % BLOCK == 0, "requires HEADSIZE % BLOCK == 0");
  // assert: UNROLL * BLOCK == H
  constexpr int UNROLL = HEADSIZE / BLOCK;
  constexpr int UNDERLYING_SIZE =
      qcache::QCacheConfig<QMODE, T>::UNDERLYING_SIZE;

  static_assert(UNROLL % UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_UNROLL = UNROLL / UNDERLYING_SIZE;
  constexpr int QUANT_HEADSIZE = HEADSIZE / UNDERLYING_SIZE;

  const uint32_t nGroups = gridDim.y;

  const uint32_t spanIdx = blockIdx.x;
  const uint32_t headIdx = blockIdx.y;
  const uint32_t posInSpan = blockIdx.z;

  const uint32_t srcY = headIdx;
  const uint32_t srcX = spanIdx * SPAN + posInSpan;

  const T* ldgPtr = src + (srcX * nGroups + srcY) * HEADSIZE;
  PackT<UNROLL, T> srcPack;
  // TODO: assuming SPAN-aligned now, should take xSize into account
  LdgCS(&srcPack, ldgPtr + threadIdx.x * UNROLL);
  const T(&regs)[UNROLL] = srcPack.data;

  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  using Param = qcache::QuantParam<QMODE, T>;

  const typename Param::Builder<BLOCK, UNROLL, HEADSIZE> builder;
  const Param param = builder(regs);

  using QStgPackT = PackT<QUANT_UNROLL, QT>;
  QStgPackT qStgPack;
#pragma unroll
  for (int i = 0; i < QUANT_UNROLL; ++i) {
    param.Quant(qStgPack.data[i], regs + i * UNDERLYING_SIZE);
  }

  QT* const spanPtr = spanPtrs[spanIdx];
  QT* const qStgPtr = spanPtr + (headIdx * SPAN + posInSpan) * QUANT_HEADSIZE;
  *(reinterpret_cast<QStgPackT*>(qStgPtr + threadIdx.x * QUANT_UNROLL)) =
      qStgPack;

  if constexpr (QMODE == span::QuantMode::NONE) {
    return;
  }

  static_assert(BLOCK == impl::WARP_SIZE, "block should contain only 1 warp");

  Param* const paramPtr =
      reinterpret_cast<Param*>(spanPtr + nGroups * SPAN * QUANT_HEADSIZE);

  // only thread 0 write param
  if (threadIdx.x == 0) {
    *(paramPtr + headIdx * SPAN + posInSpan) = param;
  }
  return;
}

template <span::QuantMode QMODE, int SPAN, int HEADSIZE, typename T>
void ContextSpanCopyKernelLaunch(void* const* spanPtrs, const T* src,
                                 uint32_t nGroups, uint32_t seqLen,
                                 cudaStream_t stream) {
  // each block uses one warp, one warp handles one head
  constexpr int BLOCK = impl::WARP_SIZE;
  const uint32_t nSpans = (seqLen + SPAN - 1) / SPAN;
  const size_t xSize = static_cast<size_t>(HEADSIZE) * nGroups * seqLen;
  if (xSize > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("ContextSpanCopy: src size exceeds uint32_t");
  }

  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  QT* const* const spans = reinterpret_cast<QT* const*>(spanPtrs);

  QuantSpanCopyKernel<BLOCK, QMODE, SPAN, HEADSIZE>
      <<<dim3(nSpans, nGroups, SPAN), BLOCK, 0, stream>>>(
          spans, src, static_cast<uint32_t>(xSize));
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <span::QuantMode QMODE, int SPAN, typename T>
void ContextSpanCopyHeadsizeDispatch(void* const* spanPtrs, const T* src,
                                     uint32_t nGroups, uint32_t headSize,
                                     uint32_t seqLen, cudaStream_t stream) {
  if (!impl::IsPowerOf2(headSize)) {
    throw std::runtime_error("ContextSpanCopy: headSize must be power of 2");
  }

  switch (headSize) {
    // case (1 << 6): {
    //   constexpr int HEADSIZE = 1 << 6;
    //   ContextSpanCopyKernelLaunch<QMODE, SPAN, HEADSIZE>(spanPtrs, src,
    //        nGroups, seqLen, stream);
    //   break;
    // }
    case (1 << 7): {
      constexpr int HEADSIZE = 1 << 7;
      ContextSpanCopyKernelLaunch<QMODE, SPAN, HEADSIZE>(spanPtrs, src, nGroups,
                                                         seqLen, stream);
      break;
    }
    default:
      throw std::runtime_error("ContextSpanCopy: unsupported headSize");
  }
  return;
}

template <span::QuantMode QMODE, typename T>
void ContextSpanCopyModeDispatch(void* const* spanPtrs, const T* src,
                                 uint32_t nGroups, uint32_t headSize,
                                 uint32_t spanLen, uint32_t seqLen,
                                 cudaStream_t stream) {
  switch (spanLen) {
    case 16: {
      constexpr int SPAN = 16;
      cache::ContextSpanCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, src, nGroups, headSize, seqLen, stream);
      break;
    }
    case 32: {
      constexpr int SPAN = 32;
      cache::ContextSpanCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, src, nGroups, headSize, seqLen, stream);
      break;
    }
    case 64: {
      constexpr int SPAN = 64;
      cache::ContextSpanCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, src, nGroups, headSize, seqLen, stream);
      break;
    }
    case 128: {
      constexpr int SPAN = 128;
      cache::ContextSpanCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, src, nGroups, headSize, seqLen, stream);
      break;
    }
    default:
      throw std::runtime_error("ContextSpanCopy: unsupported spanLen");
  }
}

}  // namespace cache

/**
 * @brief During the context phase of MHA, slice the contiguous cache to spans,
 * transpose each span, and copy all spans to the specified span pointers.
 *
 * Shape info:
 *
 * spanPtrs: [nSpans]
 *               |---> [nGroups, spanLen, headSize]
 * src: [maxLen, nGroups, headSize], maxLen >= seqLen
 *
 * @tparam T
 * @param spanPtrs Array of device pointers to target spans.
 * @param src Pointer to the source contiguous cache.
 * @param nGroups Number of K/V heads.
 * @param headSize Size of each head, must be power of 2; support 2**5 ~ 2**9.
 * @param spanLen Length (number of tokens) of each cache span; support 16,
 * 32, 64.
 * @param seqLen Length (number of tokens) of the source contiguous cache.
 * @param cacheMode KV cache quant mode.
 * @param stream CUDA stream.
 */
template <typename T>
void ContextSpanCopyLauncher(void* const* spanPtrs, const T* src, int nGroups,
                             int headSize, int spanLen, int seqLen,
                             span::QuantMode cacheMode, cudaStream_t stream) {
  switch (cacheMode) {
    case span::QuantMode::NONE:
      cache::ContextSpanCopyModeDispatch<span::QuantMode::NONE>(
          spanPtrs, src, nGroups, headSize, spanLen, seqLen, stream);
      break;
    case span::QuantMode::I8:
      cache::ContextSpanCopyModeDispatch<span::QuantMode::I8>(
          spanPtrs, src, nGroups, headSize, spanLen, seqLen, stream);
      break;
    case span::QuantMode::U4:
      cache::ContextSpanCopyModeDispatch<span::QuantMode::U4>(
          spanPtrs, src, nGroups, headSize, spanLen, seqLen, stream);
      break;
    default:
      throw std::runtime_error("ContextSpanCopy: unsupported cacheMode");
  }
  return;
}

}  // namespace cuda

}  // namespace allspark
