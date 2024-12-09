/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    context_span_copy_bf16.cu
 */

#ifdef ENABLE_BF16
#include "context_span_copy.cuh"
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"

namespace allspark {
namespace cuda {

template void ContextSpanCopyLauncher(void* const* spanPtrs,
                                      const hie::bfloat16* src, int nGroups,
                                      int headSize, int spanLen, int seqLen,
                                      span::QuantMode cacheMode,
                                      cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
