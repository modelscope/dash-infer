/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_bf16.cu
 */

#include "softmax.cuh"

#ifdef ENABLE_BF16
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#endif

namespace allspark {
namespace cuda {

#ifdef ENABLE_BF16
template void StridedSoftmaxGetWorkspaceSize<hie::bfloat16>(size_t* wsInBytes,
                                                            int taskNum,
                                                            int stride);
template void StridedSoftmaxLauncher(hie::bfloat16* y, const hie::bfloat16* x,
                                     const int* taskLenPtr,
                                     const float* temperatures, void* workspace,
                                     size_t wsSizeInBytes, int taskNum,
                                     int stride, cudaStream_t stream);
template void StridedLogSoftmaxLauncher(
    hie::bfloat16* y, const hie::bfloat16* x, const int* taskLenPtr,
    const float* temperatures, void* workspace, size_t wsSizeInBytes,
    int taskNum, int stride, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
