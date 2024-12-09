/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    context_span_copy_fp16.cu
 */

#ifdef ENABLE_FP16
#include <cuda_fp16.h>

#include "context_span_copy.cuh"

namespace allspark {
namespace cuda {

template void ContextSpanCopyLauncher(void* const* spanPtrs, const half* src,
                                      int nGroups, int headSize, int spanLen,
                                      int seqLen, span::QuantMode cacheMode,
                                      cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
