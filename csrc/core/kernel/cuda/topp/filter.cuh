/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    filter.cuh
 */

#pragma once

#include <hiednn.h>
#include <hiednn_cuda.h>
#include <utility/check_cuda.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <vector>

#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace topp {

namespace filter {

template <int BLOCK_X, int BLOCK_Y, bool STRICT, typename T, typename IdxT>
__global__ void __launch_bounds__(512)
    cutoffKernel(const T* ascendingVal, const float* cutoffs,
                 const int* taskLenPtr, IdxT* numToKeep, const uint32_t taskNum,
                 const uint32_t stride, const uint32_t minNumToKeep,
                 const float minCutoff, const float maxCutoff) {
  static_assert(BLOCK_X == WARP_SIZE, "blockDim.x (BLOCK_X) must be WARP_SIZE");
  const uint32_t stepSize = BLOCK_Y * gridDim.x;
  const uint32_t task0 = threadIdx.y + blockIdx.x * BLOCK_Y;

  for (uint32_t taskId = task0; taskId < taskNum; taskId += stepSize) {
    const uint32_t taskLen =
        taskLenPtr != nullptr ? taskLenPtr[taskId] : stride;
    const float cutoffValF = cutoffs[taskId];

    uint32_t nKeep;
    if (minCutoff < cutoffValF && cutoffValF < maxCutoff) {
      const T cutoffVal = static_cast<T>(cutoffValF);
      const T* valPtr = ascendingVal + taskId * stride;
      uint32_t bound = cub::LowerBound(valPtr, taskLen, cutoffVal);
      if (STRICT) {
        // include the first val greater than p
        if (bound < taskLen && valPtr[bound] == cutoffVal) {
          bound += 1;
        }
      }
      nKeep = min(max(bound + 1, minNumToKeep), taskLen);
    } else {
      nKeep = taskLen;
    }

    if (threadIdx.x == 0) {
      numToKeep[taskId] = static_cast<IdxT>(nKeep);
    }
  }
  return;
}

}  // namespace filter

/**
 * @brief Find the first item in an ascending array which is strictly greater
 * than the given cut-off value, returning the number of items preceding and
 * including this one.
 *
 * @param[out] numToKeep Number of remaining items in each task.
 * @param[in] valPtr Input strided arrays, taking value in [0, 1].
 * @param[in] cutoffs Cut-off values for each task; a cut-off value beyond (0,
 * 1) takes no effect for the corresponding task.
 * @param[in] taskLenPtr Valid lengths of each task in the batch.
 * @param[in] taskNum Batch size.
 * @param[in] stride Stride of input array.
 * @param[in] stream CUDA stream.
 */
template <typename T, typename IdxT>
void LaunchCutoff(IdxT* numToKeep, const T* valPtr, const float* cutoffs,
                  const int* taskLenPtr, uint32_t taskNum, uint32_t stride,
                  cudaStream_t stream) {
  constexpr bool STRICT = true;
  constexpr int MIN_NUM_KEEP = 1;
  constexpr int BLOCK_Y = 512 / WARP_SIZE;
  uint32_t nBlocks = UIntDivUp<uint32_t>(taskNum, BLOCK_Y);

  filter::cutoffKernel<WARP_SIZE, BLOCK_Y, STRICT>
      <<<nBlocks, dim3(WARP_SIZE, BLOCK_Y), 0, stream>>>(
          valPtr, cutoffs, taskLenPtr, numToKeep, taskNum, stride, MIN_NUM_KEEP,
          0.f, 1.f);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

}  // namespace topp
}  // namespace cuda
}  // namespace allspark
