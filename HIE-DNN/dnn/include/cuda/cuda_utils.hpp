/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_utils.hpp
 */

#ifndef DNN_INCLUDE_CUDA_CUDA_UTILS_HPP_
#define DNN_INCLUDE_CUDA_CUDA_UTILS_HPP_

#include <utils.hpp>

#define CHECK_CUDA(EXPR) \
    ((EXPR) == cudaSuccess ? HIEDNN_STATUS_SUCCESS : \
                             HIEDNN_STATUS_RUNTIME_ERROR);

#define CHECK_CUDA_RETURN(EXPR) { \
    if ((EXPR) != cudaSuccess) { \
        return HIEDNN_STATUS_RUNTIME_ERROR; \
    } \
}

namespace hiednn {

namespace cuda {

constexpr int WARP_SIZE = 32;
constexpr int CTA_MAX_SIZE = 1024;
constexpr int CTA_WARP_MAX = CTA_MAX_SIZE / WARP_SIZE;

/*
 * brief:
 * for tensor with @n elements and thread block size = block:
 *     threads with tid < @nPack access memory vectorized,
 *     threads with @nPack <= tid < @nThreads access memory normally,
 *         and access the elements with index (tid + @unpackedOffset)
 *     @nBlock is grid size, or the number of thread blocks
 */
struct PackedEltwiseConfig {
    int64_t nPack;
    int64_t nThread;
    int64_t unpackedOffset;
    int64_t nBlock;

    // @n: number of elements
    // @packSize: number of elements accessed by a thread
    // @block: size of thread block
    PackedEltwiseConfig(int64_t n, int64_t packSize, int64_t block) {
        nPack = n / packSize;
        nThread = nPack + n % packSize;
        unpackedOffset = (packSize - 1) * nPack;
        nBlock = UIntDivRU(nThread, block);
    }
};

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_CUDA_UTILS_HPP_

