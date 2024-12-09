/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk_radix_bf16.cu
 */

#include "topk_radix.cuh"

namespace allspark {
namespace cuda {

#ifdef ENABLE_BF16
template void TopKRadixGetWorkspaceSize<hie::bfloat16>(size_t* sizeInBytes,
                                                       int batch_size,
                                                       int length);

template void TopKRadixKernelLauncher<hie::bfloat16>(
    hie::bfloat16* output, int* output_indices, const hie::bfloat16* input,
    void* workspace, int batch_size, int length, int64_t k,
    cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
