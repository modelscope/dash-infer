/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    context_span_copy_fp32.cu
 */

#include "context_span_copy.cuh"

namespace allspark {
namespace cuda {

template void ContextSpanCopyLauncher(void* const* spanPtrs, const float* src,
                                      int nGroups, int headSize, int spanLen,
                                      int seqLen, span::QuantMode cacheMode,
                                      cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
