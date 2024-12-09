/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk_radix_fp32.cu
 */

#include "topk_radix.cuh"

namespace allspark {
namespace cuda {

template void TopKRadixGetWorkspaceSize<float>(size_t* sizeInBytes,
                                               int batch_size, int length);

template void TopKRadixKernelLauncher<float>(float* output, int* output_indices,
                                             const float* input,
                                             void* workspace, int batch_size,
                                             int length, int64_t k,
                                             cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
