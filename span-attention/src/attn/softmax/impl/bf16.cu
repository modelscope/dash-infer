/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bf16.cu
 */

#ifdef ENABLE_BF16
#include "attn/softmax/launcher.cuh"
#include "common/data_type.h"

namespace span {
template size_t SoftmaxInplaceWorkspaceBytes<bfloat16_t>(int batch, int nHead,
                                                         int maxLength);

template SaStatus SoftmaxInplace(bfloat16_t* QK, const uint32_t* seqLengths,
                                 const void* softmaxMappingWs, void* softmaxWs,
                                 int batch, int nHead, int stride,
                                 int maxLength, uint32_t totalTiles,
                                 int smCount, cudaStream_t stream);
}  // namespace span
#endif
