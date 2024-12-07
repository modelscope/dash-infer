/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_ppu_kernel.h
 */

#pragma once
#include <cstdint>
#include <cstdlib>

#include "../cuda_common.h"
// #include "../hie/cuda_activation.hpp"

namespace allspark {
namespace cuda {
void GetWorkspaceSize(size_t* hostWsSize, size_t* deviceWsSize, uint32_t m,
                      uint32_t nMatB);

template <typename T>
void MoeBatchedGemmLauncher(const T* A, const T* B, const uint32_t* matBIndices,
                            T* C, uint32_t* matCRowIndices, void* hostWs,
                            size_t hostWsSize, void* deviceWs,
                            size_t deviceWsSize, uint32_t matARows, uint32_t n,
                            uint32_t k, uint32_t nMatB,
                            uint32_t nMatBPerMatARow, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark