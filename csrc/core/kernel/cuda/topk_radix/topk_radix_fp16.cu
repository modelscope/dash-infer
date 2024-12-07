/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk_radix_fp16.cu
 */

#include "topk_radix.cuh"

namespace allspark {
namespace cuda {

#ifdef ENABLE_FP16
template void TopKRadixGetWorkspaceSize<half>(size_t* sizeInBytes,
                                              int batch_size, int length);

template void TopKRadixKernelLauncher<half>(half* output, int* output_indices,
                                            const half* input, void* workspace,
                                            int batch_size, int length,
                                            int64_t k, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
