/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention_fp16.cu
 */

#ifdef ENABLE_FP16
#include "span_attention_fp16.cuh"

namespace allspark {
namespace cuda {

template void SpanAttentionCreate(SpanAttention<__half>** obj, int batchSize,
                                  int nHeads, int nGroups, int headSize,
                                  int spanLen, int nSpans, int seqLen,
                                  int deviceId);
template void SpanAttentionDestroy(SpanAttention<__half>* obj);
template void SpanAttentionGetWorkspaceSize(const SpanAttention<__half>* obj,
                                            size_t* wsInBytes);
template void SpanAttentionLauncher(const SpanAttention<__half>* obj,
                                    __half* output, const __half* query,
                                    const void* const* kSpanArray,
                                    const void* const* vSpanArray,
                                    float QKScale, void* workspace,
                                    size_t wsSizeInBytes, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
