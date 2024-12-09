/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp32.cu
 */

#include "attn/softmax/launcher.cuh"

namespace span {
template size_t SoftmaxInplaceWorkspaceBytes<float>(int batch, int nHead,
                                                    int maxLength);

template SaStatus SoftmaxInplace(float* QK, const uint32_t* seqLengths,
                                 const void* softmaxMappingWs, void* softmaxWs,
                                 int batch, int nHead, int stride,
                                 int maxLength, uint32_t totalTiles,
                                 int smCount, cudaStream_t stream);
}  // namespace span
