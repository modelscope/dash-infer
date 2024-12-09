/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_v1.h
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace allspark {
namespace cuda {

// span attention
template <typename T>
class SpanAttention;
template <typename T>
void SpanAttentionCreate(SpanAttention<T>** obj, int batchSize, int nHeads,
                         int nGroups, int headSize, int spanLen,
                         int nSpansPerBatch, int seqLen, int deviceId);
template <typename T>
void SpanAttentionDestroy(SpanAttention<T>* obj);
template <typename T>
void SpanAttentionGetWorkspaceSize(const SpanAttention<T>* obj,
                                   size_t* wsInBytes);
template <typename T>
void SpanAttentionLauncher(const SpanAttention<T>* obj, T* output,
                           const T* query, const void* const* kSpanArray,
                           const void* const* vSpanArray, float QKScale,
                           void* workspace, size_t wsSizeInBytes,
                           cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
