/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp_fp16.cu
 */

#include "topp.cuh"

namespace allspark {
namespace cuda {

#ifdef ENABLE_FP16
template void TopPSoftmaxGetWorkspaceSize<half>(size_t* sizeInBytes,
                                                int batch_size, int length,
                                                bool input_is_sorted);

template void TopPSoftmaxLauncher<half>(
    int* topp_count, half* topp_probs, int* topp_indices,
    const half* input_logits, const float* p_values, const float* temperatures,
    half* temp_probs, void* workspace, size_t ws_size_in_bytes, int batch_size,
    int length, bool input_is_sorted, hiednnCudaHandle_t handle,
    cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
