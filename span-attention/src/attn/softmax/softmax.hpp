/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax.hpp
 */

#pragma once

#include <span_attn.h>

namespace span {
template <typename FType>
size_t SoftmaxInplaceWorkspaceBytes(int batch, int nHead, int maxLength);

template <typename FType>
SaStatus SoftmaxInplace(FType* QK, const uint32_t* seqLengths,
                        const void* softmaxMappingWs, void* softmaxWs,
                        int batch, int nHead, int stride, int maxLength,
                        uint32_t totalTiles, int smCount, cudaStream_t stream);
}  // namespace span
