/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mass_span_copy.cuh
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

constexpr int MAX_BLOCK = 256;

}  // namespace impl

template <typename QT>
__global__ void __launch_bounds__(impl::MAX_BLOCK)
    copySpanToContKernel(QT* dst, const QT* const* src, int span_num,
                         int elem_num) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int arrayIndex = globalIndex / elem_num;
  int elementIndex = globalIndex % elem_num;
  int dstIndex = globalIndex;

  if (arrayIndex < span_num) {
    if (elementIndex < elem_num) {
      dst[dstIndex] = src[arrayIndex][elementIndex];
    }
  }
}

template <typename QT>
__global__ void __launch_bounds__(impl::MAX_BLOCK)
    copyContToSpanKernel(QT** dst, QT const* src, int span_num, int elem_num) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int arrayIndex = globalIndex / elem_num;
  int elementIndex = globalIndex % elem_num;
  int srcIndex = globalIndex;

  if (arrayIndex < span_num) {
    if (elementIndex < elem_num) {
      dst[arrayIndex][elementIndex] = src[srcIndex];
    }
  }
}

template <typename QT>
void LaunchCopySpanToContKernel(void* dst, const void* const* src, int span_num,
                                int span_size, cudaStream_t stream) {
  int elem_num = span_size / sizeof(QT);
  int totalSize = span_num * elem_num;
  int blockSize = impl::MAX_BLOCK;
  int numBlocksKernel = (totalSize + blockSize - 1) / blockSize;

  dim3 threadsPerBlock(blockSize);
  dim3 numBlocks(numBlocksKernel);

  QT* dst_ptr = reinterpret_cast<QT*>(dst);
  const QT* const* src_ptr = reinterpret_cast<const QT* const*>(src);

  copySpanToContKernel<QT><<<numBlocks, threadsPerBlock, 0, stream>>>(
      dst_ptr, src_ptr, span_num, elem_num);

  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <typename QT>
void LaunchCopyContToSpanKernel(void** dst, void const* src, int span_num,
                                int span_size, cudaStream_t stream) {
  int elem_num = span_size / sizeof(QT);
  int totalSize = span_num * elem_num;
  int blockSize = impl::MAX_BLOCK;
  int numBlocksKernel = (totalSize + blockSize - 1) / blockSize;

  dim3 threadsPerBlock(blockSize);
  dim3 numBlocks(numBlocksKernel);

  QT** dst_ptr = reinterpret_cast<QT**>(dst);
  QT const* src_ptr = reinterpret_cast<QT const*>(src);

  copyContToSpanKernel<QT><<<numBlocks, threadsPerBlock, 0, stream>>>(
      dst_ptr, src_ptr, span_num, elem_num);

  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

}  // namespace cache

/**
 * @brief
 * @param dst Pointer to the target contiguous cache.
 * @param spanPtrs Array of device pointers to source spans.
 * @param span_num Number of spans.
 * @param span_size Span memory size in byte.
 * @param dtype Attention data type.
 * @param cacheMode KV cache quant mode.
 * @param stream CUDA stream.
 */
template <typename T>
void SpanToContCopyLauncher(void* dst, const void* const* spanPtrs,
                            const int span_num, const int span_size,
                            const DataType dtype,
                            const span::QuantMode cacheMode,
                            const cudaStream_t stream) {
  switch (cacheMode) {
    case span::QuantMode::NONE: {
      using Config = qcache::QCacheConfig<span::QuantMode::NONE, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopySpanToContKernel<QT>(dst, spanPtrs, span_num, span_size,
                                            stream);
      break;
    }
    case span::QuantMode::I8: {
      using Config = qcache::QCacheConfig<span::QuantMode::I8, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopySpanToContKernel<QT>(dst, spanPtrs, span_num, span_size,
                                            stream);
      break;
    }
    case span::QuantMode::U4: {
      using Config = qcache::QCacheConfig<span::QuantMode::U4, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopySpanToContKernel<QT>(dst, spanPtrs, span_num, span_size,
                                            stream);
      break;
    }
    default:
      throw std::runtime_error("SpanToContCopyLauncher: unsupported cacheMode");
  }
  return;
}

/**
 * @brief
 * @param spanPtrs Array of device pointers to target spans.
 * @param src Pointer to the source contiguous cache.
 * @param span_num Number of spans.
 * @param span_size Span memory size in byte.
 * @param dtype Attention data type.
 * @param cacheMode KV cache quant mode.
 * @param stream CUDA stream.
 */
template <typename T>
void ContToSpanCopyLauncher(void** spanPtrs, void const* src,
                            const int span_num, const int span_size,
                            const DataType dtype,
                            const span::QuantMode cacheMode,
                            const cudaStream_t stream) {
  switch (cacheMode) {
    case span::QuantMode::NONE: {
      using Config = qcache::QCacheConfig<span::QuantMode::NONE, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopyContToSpanKernel<QT>(spanPtrs, src, span_num, span_size,
                                            stream);
      break;
    }
    case span::QuantMode::I8: {
      using Config = qcache::QCacheConfig<span::QuantMode::I8, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopyContToSpanKernel<QT>(spanPtrs, src, span_num, span_size,
                                            stream);
      break;
    }
    case span::QuantMode::U4: {
      using Config = qcache::QCacheConfig<span::QuantMode::U4, T>;
      using QT = typename Config::QuantT;
      cache::LaunchCopyContToSpanKernel<QT>(spanPtrs, src, span_num, span_size,
                                            stream);
      break;
    }
    default:
      throw std::runtime_error("ContToSpanCopyLauncher: unsupported cacheMode");
  }
  return;
}
}  // namespace cuda
}  // namespace allspark
