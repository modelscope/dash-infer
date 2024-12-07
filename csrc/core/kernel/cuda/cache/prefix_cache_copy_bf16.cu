/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_copy_bf16.cu
 */

#ifdef ENABLE_BF16
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#include "prefix_cache_copy.cuh"

namespace allspark {
namespace cuda {

template void PrefixCacheCopyLauncher(const void* const* spanPtrs,
                                      hie::bfloat16* dst, int nGroups,
                                      int headSize, int spanLen, int preLen,
                                      span::QuantMode cacheMode,
                                      cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
