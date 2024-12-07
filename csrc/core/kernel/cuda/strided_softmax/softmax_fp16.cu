/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_fp16.cu
 */

#include "softmax.cuh"

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif

namespace allspark {
namespace cuda {

#ifdef ENABLE_FP16
template void StridedSoftmaxGetWorkspaceSize<half>(size_t* wsInBytes,
                                                   int taskNum, int stride);
template void StridedSoftmaxLauncher(half* y, const half* x,
                                     const int* taskLenPtr,
                                     const float* temperatures, void* workspace,
                                     size_t wsSizeInBytes, int taskNum,
                                     int stride, cudaStream_t stream);
template void StridedLogSoftmaxLauncher(half* y, const half* x,
                                        const int* taskLenPtr,
                                        const float* temperatures,
                                        void* workspace, size_t wsSizeInBytes,
                                        int taskNum, int stride,
                                        cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
