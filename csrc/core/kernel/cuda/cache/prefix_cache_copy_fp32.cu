/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_copy_fp32.cu
 */

#include "prefix_cache_copy.cuh"

namespace allspark {
namespace cuda {

template void PrefixCacheCopyLauncher(const void* const* spanPtrs, float* dst,
                                      int nGroups, int headSize, int spanLen,
                                      int preLen, span::QuantMode cacheMode,
                                      cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
