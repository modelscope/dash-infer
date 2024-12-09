/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_fp32.cu
 */

#include "softmax.cuh"

namespace allspark {
namespace cuda {

template void StridedSoftmaxGetWorkspaceSize<float>(size_t* wsInBytes,
                                                    int taskNum, int stride);
template void StridedSoftmaxLauncher(float* y, const float* x,
                                     const int* taskLenPtr,
                                     const float* temperatures, void* workspace,
                                     size_t wsSizeInBytes, int taskNum,
                                     int stride, cudaStream_t stream);
template void StridedLogSoftmaxLauncher(float* y, const float* x,
                                        const int* taskLenPtr,
                                        const float* temperatures,
                                        void* workspace, size_t wsSizeInBytes,
                                        int taskNum, int stride,
                                        cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
