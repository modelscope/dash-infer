/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mass_span_copy_fp16.cu
 */

#ifdef ENABLE_FP16
#include <cuda_fp16.h>

#include "mass_span_copy.cuh"

namespace allspark {
namespace cuda {

template void SpanToContCopyLauncher<half>(
    void* dst, const void* const* spanPtrs, const int span_num,
    const int span_size, const DataType dtype, const span::QuantMode cacheMode,
    const cudaStream_t stream);

template void ContToSpanCopyLauncher<half>(void** spanPtrs, void const* src,
                                           const int span_num,
                                           const int span_size,
                                           const DataType dtype,
                                           const span::QuantMode cacheMode,
                                           const cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
