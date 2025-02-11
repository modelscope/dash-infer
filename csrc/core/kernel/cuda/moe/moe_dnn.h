/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_dnn.h
 */

#pragma once
#include <cstdint>
#include <cstdlib>

#include "../cuda_common.h"
// #include "../hie/cuda_activation.hpp"
namespace allspark {
namespace cuda {
size_t GetWorkspaceSizeLauncher(uint32_t matARows, uint32_t nMatB);

template <typename T>
void MoeBatchedGemmLauncher(const T* A, const T* B, const uint32_t* matBIndices,
                            T* C, uint32_t* matCRowIndices, void* deviceWs,
                            size_t deviceWsSize, uint32_t matARows, uint32_t n,
                            uint32_t k, uint32_t nMatB,
                            uint32_t nMatBPerMatARow, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark