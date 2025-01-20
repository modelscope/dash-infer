/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_copy.cuh
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
constexpr int MAX_BLOCK = 256;

template <typename UT>
static inline constexpr bool IsPowerOf2(const UT& x) {
  return (x & (x - 1)) == 0;
}

}  // namespace impl

template <int BLOCK_X, int BLOCK_Y, int N_TILES, int PACK,
          span::QuantMode QMODE, typename T>
__global__ void __launch_bounds__(impl::MAX_BLOCK) PrefixCacheCopyKernel(
    const typename qcache::QCacheConfig<QMODE, T>::QuantT* const* const
        spanPtrs,
    T* const contiguousKVCache) {
  using Config = qcache::QCacheConfig<QMODE, T>;
  using Param = qcache::QuantParam<QMODE, T>;
  using QT = typename Config::QuantT;
  using CPT = typename Config::ComputeT;

  constexpr int HEADSIZE = BLOCK_X * PACK;
  constexpr int SPAN = BLOCK_Y * N_TILES;

  constexpr int UNDERLYING_SIZE = Config::UNDERLYING_SIZE;
  static_assert(PACK % UNDERLYING_SIZE == 0, "");
  constexpr int QUANT_PACK = PACK / UNDERLYING_SIZE;
  constexpr int QUANT_HEADSIZE = HEADSIZE / UNDERLYING_SIZE;

  const uint32_t spanId = blockIdx.y;
  const uint32_t headId = blockIdx.x / N_TILES;
  const uint32_t tileId = blockIdx.x % N_TILES;
  const uint32_t nGroups = gridDim.x / N_TILES;

  const uint32_t tidX = threadIdx.x % BLOCK_X;
  const uint32_t tidY = threadIdx.x / BLOCK_X;

  const QT* const spanPtr = spanPtrs[spanId];

  using MemT = WordPackT<QUANT_PACK, QT>;
  const MemT* const ldgPtr =
      reinterpret_cast<const MemT*>(spanPtr + headId * SPAN * QUANT_HEADSIZE +
                                    tileId * BLOCK_Y * QUANT_HEADSIZE);
  const Param* const paramPtr = reinterpret_cast<const Param*>(
                                    spanPtr + nGroups * SPAN * QUANT_HEADSIZE) +
                                headId * SPAN + tileId * BLOCK_Y;

  MemT qVals;
  LdgCS<MemT>(&qVals, ldgPtr + threadIdx.x);

  Param qParam;
  if constexpr (QMODE != span::QuantMode::NONE) {
    LdgNC<Param>(&qParam, paramPtr + tidY);
  }

  CPT cVals[PACK];
  qVals.template Unpack<UNDERLYING_SIZE, Config::UnderlyingT>(
      cVals, Config::ExtractorT());

  if constexpr (QMODE != span::QuantMode::NONE) {
    CPT zero = static_cast<CPT>(qParam.zero);
    CPT scale = static_cast<CPT>(qParam.scale);
    zero *= scale;
#pragma unroll
    for (int p = 0; p < PACK; ++p) {
      cVals[p] = cVals[p] * scale - zero;
    }
  }

  using StgPackT = PackT<PACK, T>;
  StgPackT stgPack;
#pragma unroll
  for (int p = 0; p < PACK; ++p) {
    stgPack.data[p] = static_cast<T>(cVals[p]);
  }

  uint32_t seqIdx = spanId * SPAN + tileId * BLOCK_Y + tidY;
  StgPackT* stgPtr = reinterpret_cast<StgPackT*>(
      contiguousKVCache + (seqIdx * nGroups + headId) * HEADSIZE);
  stgPtr[tidX] = stgPack;
  return;
}

template <span::QuantMode QMODE, int SPAN, int HEADSIZE, typename T>
void LaunchPrefixCacheCopyKernel(const void* const* spanPtrs,
                                 T* contiguousKVCache, uint32_t nGroups,
                                 uint32_t seqLen, cudaStream_t stream) {
  // 16Byte memory access
  constexpr int PACK = 16 / sizeof(T);
  static_assert(impl::IsPowerOf2(HEADSIZE), "invalid HEADSIZE");
  static_assert(HEADSIZE >= PACK, "invalid PACK");

  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  const QT* const* spans = reinterpret_cast<const QT* const*>(spanPtrs);

  constexpr int BLOCK_X = HEADSIZE / PACK;
  static_assert(impl::MAX_BLOCK >= BLOCK_X, "invalid BLOCK_X");
  constexpr int BLOCK_Y =
      SPAN * BLOCK_X <= impl::MAX_BLOCK ? SPAN : impl::MAX_BLOCK / BLOCK_X;
  constexpr int N_TILES = SPAN / BLOCK_Y;
  static_assert(BLOCK_X * BLOCK_Y >= impl::WARP_SIZE, "block too small");

  // restriction: seqLen % SPAN == 0
  uint32_t nSpan = (seqLen + SPAN - 1) / SPAN;
  PrefixCacheCopyKernel<BLOCK_X, BLOCK_Y, N_TILES, PACK, QMODE>
      <<<dim3(N_TILES * nGroups, nSpan), BLOCK_X * BLOCK_Y, 0, stream>>>(
          spans, contiguousKVCache);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <span::QuantMode QMODE, int SPAN, typename T>
void PrefixCacheCopyHeadsizeDispatch(const void* const* spanPtrs, T* dst,
                                     uint32_t nGroups, uint32_t headSize,
                                     uint32_t seqLen, cudaStream_t stream) {
  if (!impl::IsPowerOf2(headSize)) {
    throw std::runtime_error("PrefixCacheCopy: headSize must be power of 2");
  }

  switch (headSize) {
    case (1 << 7): {
      constexpr int HEADSIZE = 1 << 7;
      LaunchPrefixCacheCopyKernel<QMODE, SPAN, HEADSIZE>(spanPtrs, dst, nGroups,
                                                         seqLen, stream);
      break;
    }
    default:
      throw std::runtime_error("PrefixCacheCopy: unsupported headSize");
  }
  return;
}

template <span::QuantMode QMODE, typename T>
void PrefixCacheCopyModeDispatch(const void* const* spanPtrs, T* dst,
                                 uint32_t nGroups, uint32_t headSize,
                                 uint32_t spanLen, uint32_t seqLen,
                                 cudaStream_t stream) {
  switch (spanLen) {
    case 16: {
      constexpr int SPAN = 16;
      cache::PrefixCacheCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, dst, nGroups, headSize, seqLen, stream);
      break;
    }
    case 32: {
      constexpr int SPAN = 32;
      cache::PrefixCacheCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, dst, nGroups, headSize, seqLen, stream);
      break;
    }
    case 64: {
      constexpr int SPAN = 64;
      cache::PrefixCacheCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, dst, nGroups, headSize, seqLen, stream);
      break;
    }
    case 128: {
      constexpr int SPAN = 128;
      cache::PrefixCacheCopyHeadsizeDispatch<QMODE, SPAN>(
          spanPtrs, dst, nGroups, headSize, seqLen, stream);
      break;
    }
    default:
      throw std::runtime_error("PrefixCacheCopy: unsupported spanLen");
  }
}

}  // namespace cache

/**
 * @brief
 * @tparam T
 * @param spanPtrs Array of device pointers to source spans.
 * @param dst Pointer to the target contiguous cache.
 * @param nGroups Number of K/V heads.
 * @param headSize Size of each head, must be power of 2; support 2**5 ~ 2**9.
 * @param spanLen Length (number of tokens) of each cache span; support 16,
 * 32, 64, 128.
 * @param preLen Prefilled sequence length (number of tokens) of the source
 * spans.
 * @param cacheMode KV cache quant mode.
 * @param stream CUDA stream.
 */
template <typename T>
void PrefixCacheCopyLauncher(const void* const* spanPtrs, T* dst, int nGroups,
                             int headSize, int spanLen, int preLen,
                             span::QuantMode cacheMode, cudaStream_t stream) {
  switch (cacheMode) {
    case span::QuantMode::NONE:
      cache::PrefixCacheCopyModeDispatch<span::QuantMode::NONE>(
          spanPtrs, dst, nGroups, headSize, spanLen, preLen, stream);
      break;
    case span::QuantMode::I8:
      cache::PrefixCacheCopyModeDispatch<span::QuantMode::I8>(
          spanPtrs, dst, nGroups, headSize, spanLen, preLen, stream);
      break;
    case span::QuantMode::U4:
      cache::PrefixCacheCopyModeDispatch<span::QuantMode::U4>(
          spanPtrs, dst, nGroups, headSize, spanLen, preLen, stream);
      break;
    default:
      throw std::runtime_error("PrefixCacheCopy: unsupported cacheMode");
  }
  return;
}

}  // namespace cuda
}  // namespace allspark
