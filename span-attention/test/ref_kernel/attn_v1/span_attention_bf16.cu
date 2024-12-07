/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention_bf16.cu
 */

#ifdef ENABLE_BF16
#include "span_attention_bf16.cuh"

namespace allspark {
namespace cuda {

template void SpanAttentionCreate(SpanAttention<hie::bfloat16>** obj,
                                  int batchSize, int nHeads, int nGroups,
                                  int headSize, int spanLen, int nSpans,
                                  int seqLen, int deviceId);
template void SpanAttentionDestroy(SpanAttention<hie::bfloat16>* obj);
template void SpanAttentionGetWorkspaceSize(
    const SpanAttention<hie::bfloat16>* obj, size_t* wsInBytes);
template void SpanAttentionLauncher(const SpanAttention<hie::bfloat16>* obj,
                                    hie::bfloat16* output,
                                    const hie::bfloat16* query,
                                    const void* const* kSpanArray,
                                    const void* const* vSpanArray,
                                    float QKScale, void* workspace,
                                    size_t wsSizeInBytes, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
