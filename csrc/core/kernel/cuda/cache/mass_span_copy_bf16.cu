/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mass_span_copy_bf16.cu
 */

#ifdef ENABLE_BF16
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#include "mass_span_copy.cuh"

namespace allspark {
namespace cuda {

template void SpanToContCopyLauncher<hie::bfloat16>(
    void* dst, const void* const* spanPtrs, const int span_num,
    const int span_size, const DataType dtype, const span::QuantMode cacheMode,
    const cudaStream_t stream);

template void ContToSpanCopyLauncher<hie::bfloat16>(
    void** spanPtrs, void const* src, const int span_num, const int span_size,
    const DataType dtype, const span::QuantMode cacheMode,
    const cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
#endif
