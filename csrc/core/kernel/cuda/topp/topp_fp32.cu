/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp_fp32.cu
 */

#include "topp.cuh"

namespace allspark {
namespace cuda {

template void TopPSoftmaxGetWorkspaceSize<float>(size_t* sizeInBytes,
                                                 int batch_size, int length,
                                                 bool input_is_sorted);

template void TopPSoftmaxLauncher<float>(
    int* topp_count, float* topp_probs, int* topp_indices,
    const float* input_logits, const float* p_values, const float* temperatures,
    float* temp_probs, void* workspace, size_t ws_size_in_bytes, int batch_size,
    int length, bool input_is_sorted, hiednnCudaHandle_t handle,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
