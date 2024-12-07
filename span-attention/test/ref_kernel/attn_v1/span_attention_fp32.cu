/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention_fp32.cu
 */

#include "span_attention.cuh"

namespace allspark {
namespace cuda {

template void SpanAttentionCreate(SpanAttention<float>** obj, int batchSize,
                                  int nHeads, int nGroups, int headSize,
                                  int spanLen, int nSpans, int seqLen,
                                  int deviceId);
template void SpanAttentionDestroy(SpanAttention<float>* obj);
template void SpanAttentionGetWorkspaceSize(const SpanAttention<float>* obj,
                                            size_t* wsInBytes);
template void SpanAttentionLauncher(const SpanAttention<float>* obj,
                                    float* output, const float* query,
                                    const void* const* kSpanArray,
                                    const void* const* vSpanArray,
                                    float QKScale, void* workspace,
                                    size_t wsSizeInBytes, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
