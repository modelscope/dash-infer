/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    block_softmax.cuh
 */

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "attn/common/tile_mapping.hpp"
#include "attn/common/utils.hpp"
#include "utils.cuh"
#include "utils/shuffle.cuh"

namespace span {

// length <= BLOCK * UNROLL
template <int BLOCK_X, int BLOCK_Y, int UNROLL, typename FT, typename ComputeT>
__global__ void __launch_bounds__(1024)
    InplaceSoftmaxKernel(FT* x, const uint32_t* lengths,
                         U32DivMod channelDivMod, uint32_t nTasks,
                         uint32_t taskStride) {
  uint32_t tidX = threadIdx.x % BLOCK_X;
  uint32_t tidY = threadIdx.x / BLOCK_X;
  uint32_t taskId = blockIdx.x * BLOCK_Y + tidY;
  uint32_t batchId = channelDivMod.Div(taskId);
  uint32_t length = lengths[batchId];
  FT* xPtr = x + taskId * taskStride + tidX;

  FT xData[UNROLL];
  int nLdg =
      length > tidX ? static_cast<int>(U32DivRU(length - tidX, BLOCK_X)) : 0;
  if (taskId < nTasks) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const double NEG_INF = -1. / 0.;
      xData[i] = i < nLdg ? xPtr[i * BLOCK_X] : static_cast<FT>(NEG_INF);
    }
  }

  ComputeT xCompute[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = static_cast<ComputeT>(xData[i]);
  }

  ComputeT max =
      BlockReduce<BLOCK_X, BLOCK_Y, UNROLL, MaxReduceFunctor, ComputeT>(
          xCompute);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = Expf(xCompute[i] - max);
  }

  ComputeT expSum =
      BlockReduce<BLOCK_X, BLOCK_Y, UNROLL, SumReduceFunctor, ComputeT>(
          xCompute);
  ComputeT expSumRcp = Rcpf(expSum);

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] *= expSumRcp;
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xData[i] = static_cast<FT>(xCompute[i]);
  }

  if (taskId < nTasks) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < nLdg) {
        xPtr[i * BLOCK_X] = xData[i];
      }
    }
  }
}

}  // namespace span
