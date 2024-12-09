/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_kernel.cu
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>

#include "../cuda_kernel.h"
#include "allspark.pb.h"
#include "moe_kernel.h"
namespace allspark {
namespace cuda {

// TODO gpu kernel
void GetWorkspaceSize(size_t* hostWsSize, size_t* deviceWsSize, uint32_t m,
                      uint32_t nMatB) {}
template <>
void MoeBatchedGemmLauncher<float>(
    const float* A, const float* B, const uint32_t* matBIndices, float* C,
    uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  // TODO
}
#ifdef ENABLE_FP16
template <>
void MoeBatchedGemmLauncher<half>(
    const half* A, const half* B, const uint32_t* matBIndices, half* C,
    uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize, void* deviceWs,
    size_t deviceWsSize, uint32_t matARows, uint32_t n, uint32_t k,
    uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  // TODO
}
#endif
#ifdef ENABLE_BF16
template <>
void MoeBatchedGemmLauncher<hie::bfloat16>(
    const hie::bfloat16* A, const hie::bfloat16* B, const uint32_t* matBIndices,
    hie::bfloat16* C, uint32_t* matCRowIndices, void* hostWs, size_t hostWsSize,
    void* deviceWs, size_t deviceWsSize, uint32_t matARows, uint32_t n,
    uint32_t k, uint32_t nMatB, uint32_t nMatBPerMatARow, cudaStream_t stream) {
  // TODO
}
#endif
}  // namespace cuda
}  // namespace allspark