/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mass_span_copy_fp32.cu
 */

#include "mass_span_copy.cuh"

namespace allspark {
namespace cuda {

template void SpanToContCopyLauncher<float>(
    void* dst, const void* const* spanPtrs, const int span_num,
    const int span_size, const DataType dtype, const span::QuantMode cacheMode,
    const cudaStream_t stream);

template void ContToSpanCopyLauncher<float>(void** spanPtrs, void const* src,
                                            const int span_num,
                                            const int span_size,
                                            const DataType dtype,
                                            const span::QuantMode cacheMode,
                                            const cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
