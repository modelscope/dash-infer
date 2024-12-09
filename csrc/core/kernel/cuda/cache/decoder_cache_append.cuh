/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    decoder_cache_append.cuh
 */

#pragma once

#include <cstdint>
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
static inline constexpr bool IsPowerOf2(const UT x) {
  return (x & (x - 1)) == 0;
}

template <int BLOCK, int UNROLL, int SPAN, int HEADSIZE, span::QuantMode QMODE,
          typename T, typename QT>
__device__ __forceinline__ void transposeQuantAppend(QT* const spanPtr,
                                                     const T* const inPtr,
                                                     const uint32_t& nGroups,
                                                     const uint32_t& posInSpan,
                                                     const uint32_t& headIdx) {
  using Param = qcache::QuantParam<QMODE, T>;
  constexpr int UNDERLYING_SIZE =
      qcache::QCacheConfig<QMODE, T>::UNDERLYING_SIZE;

  static_assert(UNROLL % UNDERLYING_SIZE == 0, "");
  static constexpr int QUANT_UNROLL = UNROLL / UNDERLYING_SIZE;
  static constexpr int QUANT_HEADSIZE = HEADSIZE / UNDERLYING_SIZE;

  const uint32_t cacheOutSize = nGroups * QUANT_HEADSIZE;
  const T* const inHeadPtr = inPtr + headIdx * HEADSIZE;

  PackT<UNROLL, T> pack;
  LdgCS(&pack, inHeadPtr + threadIdx.x * UNROLL);
  const T(&regs)[UNROLL] = pack.data;

  const typename Param::Builder<BLOCK, UNROLL, HEADSIZE> builder;
  const Param param = builder(regs);

  using QPackT = PackT<QUANT_UNROLL, QT>;
  QPackT quantRegs;
#pragma unroll
  for (int i = 0; i < QUANT_UNROLL; ++i) {
    param.Quant(quantRegs.data[i], regs + i * UNDERLYING_SIZE);
  }

  // (headIdx, packIdx) -> (headIdx, posInSpan, quantPackIdx)
  // packIdx == threadIdx.x * UNROLL
  // quantPackIdx == threadIdx.x * QUANT_UNROLL
  const uint32_t dstX = headIdx;
  const uint32_t dstY = posInSpan;
  const uint32_t dstZ = threadIdx.x * QUANT_UNROLL;
  uint32_t dstOffset = dstZ + (dstY + dstX * SPAN) * QUANT_HEADSIZE;

  *(reinterpret_cast<QPackT*>(spanPtr + dstOffset)) = quantRegs;

  if constexpr (QMODE == span::QuantMode::NONE) {
    return;
  }

  static_assert(BLOCK == impl::WARP_SIZE, "block should contain only 1 warp");

  Param* const paramPtr =
      reinterpret_cast<Param*>(spanPtr + cacheOutSize * SPAN);

  // only thread 0 write param
  if (threadIdx.x == 0) {
    *(paramPtr + headIdx * SPAN + posInSpan) = param;
  }
  return;
}

}  // namespace impl

/**
 * @brief Collect Q, and append K, V to cache spans.
 *
 * span_array: [batch, span_stride]
 * src: [batch, query_len, (n + 2 * g), H]
 * span: [g, S, H], start_pos
 *
 * grid: (batch, n)
 */
template <int BLOCK, int UNROLL, int SPAN, int HEADSIZE, span::QuantMode QMODE,
          typename T>
__global__ void __launch_bounds__(impl::WARP_SIZE) QuantCacheAppendKernel(
    typename qcache::QCacheConfig<QMODE, T>::QuantT* const* const kSpanArray,
    typename qcache::QCacheConfig<QMODE, T>::QuantT* const* const vSpanArray,
    T* const queryOut, const T* const src, const uint32_t* const oldSeqLens,
    const uint32_t nGroups, const uint32_t spanStride) {
  // query length
  constexpr int QLEN = 1;
  const uint32_t nHeads = gridDim.y;
  const uint32_t batchIdx = blockIdx.x;
  const uint32_t headIdx = blockIdx.y;

  const uint32_t querySize = nHeads * HEADSIZE;
  const uint32_t cacheInSize = nGroups * HEADSIZE;

  const uint32_t queryOutStride = QLEN * querySize;
  const uint32_t batchInStride = QLEN * (querySize + 2 * cacheInSize);

  const T* const batchInPtr = src + batchIdx * batchInStride;

  // ----------- query -----------
  const T* const qInPtr = batchInPtr + headIdx * HEADSIZE;
  T* const qOutPtr = queryOut + batchIdx * queryOutStride + headIdx * HEADSIZE;

  using QPackT = PackT<UNROLL, T>;
  QPackT queryPack;
  LdgCS(&queryPack, qInPtr + threadIdx.x * UNROLL);
  *(reinterpret_cast<QPackT*>(qOutPtr + threadIdx.x * UNROLL)) = queryPack;

  // blocks exceeding nGroups can safely exit
  if (headIdx >= nGroups) {
    return;
  }

  // ----------- key/value -----------
  const uint32_t oldSeqLen = oldSeqLens[batchIdx];
  constexpr uint32_t QLEN_OFFSET = 0;
  const uint32_t spanIdx = (oldSeqLen + QLEN_OFFSET) / SPAN;
  const uint32_t posInSpan = (oldSeqLen + QLEN_OFFSET) % SPAN;

  const T* const kInPtr = batchInPtr + querySize;
  const T* const vInPtr = kInPtr + cacheInSize;

  auto* const* const kSpanPtrs = kSpanArray + batchIdx * spanStride;
  auto* const* const vSpanPtrs = vSpanArray + batchIdx * spanStride;

  // transpose & copy key
  impl::transposeQuantAppend<BLOCK, UNROLL, SPAN, HEADSIZE, QMODE>(
      kSpanPtrs[spanIdx], kInPtr, nGroups, posInSpan, headIdx);
  // transpose & copy value
  impl::transposeQuantAppend<BLOCK, UNROLL, SPAN, HEADSIZE, QMODE>(
      vSpanPtrs[spanIdx], vInPtr, nGroups, posInSpan, headIdx);
  return;
}

template <int SPAN, int HEADSIZE, span::QuantMode QMODE, typename T>
void DecoderCacheAppendKernelLaunch(void* const* kSpanArray,
                                    void* const* vSpanArray, T* queryOut,
                                    const T* src, const uint32_t* oldSeqLens,
                                    uint32_t batchSize, uint32_t nHeads,
                                    uint32_t nGroups, int nSpansPerBatch,
                                    cudaStream_t stream) {
  // each block uses one warp, one warp handles one head
  constexpr int BLOCK = impl::WARP_SIZE;
  static_assert(HEADSIZE >= impl::WARP_SIZE,
                "HEADSIZE should be no less than WARP_SIZE");
  static_assert(HEADSIZE % impl::WARP_SIZE == 0,
                "HEADSIZE % WARP_SIZE should be 0");
  constexpr int UNROLL = HEADSIZE / impl::WARP_SIZE;

  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  QT* const* kSpanArrayPtr = reinterpret_cast<QT* const*>(kSpanArray);
  QT* const* vSpanArrayPtr = reinterpret_cast<QT* const*>(vSpanArray);

  if (nGroups > nHeads) {
    throw std::runtime_error(
        "DecoderCacheAppend: nGroups should be no more than nHeads");
  }
  const auto& grid = dim3(batchSize, nHeads);
  QuantCacheAppendKernel<BLOCK, UNROLL, SPAN, HEADSIZE, QMODE>
      <<<grid, BLOCK, 0, stream>>>(kSpanArrayPtr, vSpanArrayPtr, queryOut, src,
                                   oldSeqLens, nGroups, nSpansPerBatch);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int SPAN, int HEADSIZE, typename T>
void DecoderCacheAppendModeDispatch(void* const* kSpanArray,
                                    void* const* vSpanArray, T* queryOut,
                                    const T* src, const uint32_t* oldSeqLens,
                                    uint32_t batchSize, uint32_t nHeads,
                                    uint32_t nGroups, int nSpansPerBatch,
                                    span::QuantMode cacheMode,
                                    cudaStream_t stream) {
  switch (cacheMode) {
    case span::QuantMode::NONE:
      DecoderCacheAppendKernelLaunch<SPAN, HEADSIZE, span::QuantMode::NONE>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize, nHeads,
          nGroups, nSpansPerBatch, stream);
      break;
    case span::QuantMode::I8:
      DecoderCacheAppendKernelLaunch<SPAN, HEADSIZE, span::QuantMode::I8>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize, nHeads,
          nGroups, nSpansPerBatch, stream);
      break;
    case span::QuantMode::U4:
      DecoderCacheAppendKernelLaunch<SPAN, HEADSIZE, span::QuantMode::U4>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize, nHeads,
          nGroups, nSpansPerBatch, stream);
      break;
    default:
      throw std::runtime_error("DecoderCacheAppend: unsupported cacheMode");
  }
}

template <int SPAN, typename T>
void DecoderCacheAppendHeadsizeDispatch(
    void* const* kSpanArray, void* const* vSpanArray, T* queryOut, const T* src,
    const uint32_t* oldSeqLens, uint32_t batchSize, uint32_t headSize,
    uint32_t nHeads, uint32_t nGroups, int nSpansPerBatch,
    span::QuantMode cacheMode, cudaStream_t stream) {
  if (!impl::IsPowerOf2(headSize)) {
    throw std::runtime_error("DecoderCacheAppend: headSize must be power of 2");
  }

  switch (headSize) {
    // case (1 << 6): {
    //   constexpr int HEADSIZE = 1 << 6;
    //   DecoderCacheAppendModeDispatch<SPAN, HEADSIZE>(
    //       kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize,
    //       nHeads, nGroups, nSpansPerBatch, cacheMode, stream);
    //   break;
    // }
    case (1 << 7): {
      constexpr int HEADSIZE = 1 << 7;
      DecoderCacheAppendModeDispatch<SPAN, HEADSIZE>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize, nHeads,
          nGroups, nSpansPerBatch, cacheMode, stream);
      break;
    }
    default:
      throw std::runtime_error("DecoderCacheAppend: unsupported headSize");
  }
  return;
}

}  // namespace cache

/**
 * @brief During the decoder phase of MHA, extract Q, K, V from input tensor,
 * collect Q to queryOut, and append K, V to cache spans.
 *
 * Shape info:
 *
 * kSpanArray/vSpanArray: [batchSize, nSpans]
 *                                       |---> [nHeads, spanLen, headSize]
 * queryOut: [batchSize, (1), nHeads, headSize]
 * src: [batchSize, (1), 3, nHeads, headSize]
 * oldSeqLens: [batchSize]
 *
 * @tparam T
 * @param kSpanArray Array of device pointers to K cache spans.
 * @param vSpanArray Array of device pointers to V cache spans.
 * @param queryOut Device pointer to output Q tensor.
 * @param src Device pointer to input tensor.
 * @param oldSeqLens Device pointer to old (fulfilled) sequence lengths of
 * requests in the batch.
 * @param batchSize Batch size.
 * @param nHeads Number of Q heads.
 * @param nGroups Number of K/V heads.
 * @param headSize Size of each head, must be power of 2; support 2**5 ~ 2**9.
 * @param spanLen Length (number of tokens) of each cache span; support 16,
 * 32, 64.
 * @param nSpansPerBatch Number of spans in each batch (regarded as stride).
 * @param cacheMode KV cache quant mode.
 * @param stream CUDA stream.
 */
template <typename T>
void DecoderCacheAppendLauncher(void* const* kSpanArray,
                                void* const* vSpanArray, T* queryOut,
                                const T* src, const uint32_t* oldSeqLens,
                                int batchSize, int nHeads, int nGroups,
                                int headSize, int spanLen, int nSpansPerBatch,
                                span::QuantMode cacheMode,
                                cudaStream_t stream) {
  constexpr int QLEN = 1;
  const size_t xSize =
      static_cast<size_t>(batchSize) * QLEN * (nHeads + 2 * nGroups) * headSize;
  if (xSize > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DecoderCacheAppend: src size exceeds uint32_t");
  }

  switch (spanLen) {
    case 16: {
      constexpr int SPAN = 16;
      cache::DecoderCacheAppendHeadsizeDispatch<SPAN>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize,
          headSize, nHeads, nGroups, nSpansPerBatch, cacheMode, stream);
      break;
    }
    case 32: {
      constexpr int SPAN = 32;
      cache::DecoderCacheAppendHeadsizeDispatch<SPAN>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize,
          headSize, nHeads, nGroups, nSpansPerBatch, cacheMode, stream);
      break;
    }
    case 64: {
      constexpr int SPAN = 64;
      cache::DecoderCacheAppendHeadsizeDispatch<SPAN>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize,
          headSize, nHeads, nGroups, nSpansPerBatch, cacheMode, stream);
      break;
    }
    case 128: {
      constexpr int SPAN = 128;
      cache::DecoderCacheAppendHeadsizeDispatch<SPAN>(
          kSpanArray, vSpanArray, queryOut, src, oldSeqLens, batchSize,
          headSize, nHeads, nGroups, nSpansPerBatch, cacheMode, stream);
      break;
    }
    default:
      throw std::runtime_error("DecoderCacheAppend: unsupported spanLen");
  }
  return;
}

}  // namespace cuda

}  // namespace allspark
