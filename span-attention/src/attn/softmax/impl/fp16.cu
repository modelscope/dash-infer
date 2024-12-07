/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp16.cu
 */

#ifdef ENABLE_FP16
#include "attn/softmax/launcher.cuh"
#include "common/data_type.h"

namespace span {
template size_t SoftmaxInplaceWorkspaceBytes<half_t>(int batch, int nHead,
                                                     int maxLength);

template SaStatus SoftmaxInplace(half_t* QK, const uint32_t* seqLengths,
                                 const void* softmaxMappingWs, void* softmaxWs,
                                 int batch, int nHead, int stride,
                                 int maxLength, uint32_t totalTiles,
                                 int smCount, cudaStream_t stream);
}  // namespace span
#endif
