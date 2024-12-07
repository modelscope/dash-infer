/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_kernel_span_cache.h
 */

#pragma once
#include <spanattn/span_attn.h>

namespace allspark {
namespace cuda {

template <typename T>
void PrefixCacheCopyLauncher(const void* const* spanPtrs, T* dst, int nGroups,
                             int headSize, int spanLen, int preLen,
                             span::QuantMode cacheMode, cudaStream_t stream);
template <typename T>
void ContextSpanCopyLauncher(void* const* spanPtrs, const T* src, int nGroups,
                             int headSize, int spanLen, int seqLen,
                             span::QuantMode cacheMode, cudaStream_t stream);
template <typename T>
void DecoderCacheAppendLauncher(void* const* kSpanArray,
                                void* const* vSpanArray, T* queryOut,
                                const T* src, const uint32_t* oldSeqLens,
                                int batchSize, int nHeads, int nGroups,
                                int headSize, int spanLen, int nSpansPerBatch,
                                span::QuantMode cacheMode, cudaStream_t stream);

// mass span copy (used in prefix cache)
template <typename T>
void SpanToContCopyLauncher(void* dst, const void* const* spanPtrs,
                            const int span_num, const int span_size,
                            const DataType dtype,
                            const span::QuantMode cacheMode,
                            const cudaStream_t stream);

template <typename T>
void ContToSpanCopyLauncher(void** spanPtrs, void const* src,
                            const int span_num, const int span_size,
                            const DataType dtype,
                            const span::QuantMode cacheMode,
                            const cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
