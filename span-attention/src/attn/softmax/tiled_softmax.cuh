/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tiled_softmax.cuh
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

/**
 * maxReduceFinishCount and sumReduceFinishCount should set to 0
 * before kernel launch
 *
 * gridDim.x: number of tiles
 * gridDim.y: softmax batch size (batch * nHead)
 * softmax batch size <= 65535 (limited by gridDim.y)
 */
template <int BLOCK, int UNROLL, int MAX_NTILES_PER_TASK, typename FT,
          typename ComputeT>
__global__ void __launch_bounds__(1024)
    TiledInplaceSoftmaxKernel(FT* x, ComputeT* maxReduceWs,
                              uint32_t* maxReduceFinishCount,
                              ComputeT* sumReduceWs,
                              uint32_t* sumReduceFinishCount,
                              const uint32_t* lengths,
                              const TileMapping* mappings, uint32_t taskStride,
                              uint32_t maxNTiles) {
  uint32_t headId = blockIdx.y;
  uint32_t globalTileId = blockIdx.x;
  uint32_t batchId = mappings[globalTileId].batchId;
  uint32_t tileId = mappings[globalTileId].tileId;

  uint32_t nHeads = gridDim.y;
  uint32_t length = lengths[batchId];
  uint32_t nTiles = U32DivRU(length, BLOCK * UNROLL);

  uint32_t tileOffset = tileId * BLOCK * UNROLL;
  uint32_t taskId = batchId * nHeads + headId;
  FT* xPtr = x + taskId * taskStride + tileOffset + threadIdx.x;

  FT xData[UNROLL];
  bool fullTile = tileOffset + BLOCK * UNROLL <= length;

  // load the input
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      xData[i] = xPtr[i * BLOCK];
    }
  } else {
    uint32_t tileSize = length - tileOffset;
    int nLdg = tileSize > threadIdx.x
                   ? static_cast<int>(U32DivRU(tileSize - threadIdx.x, BLOCK))
                   : 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const double NEG_INF = -1. / 0.;
      xData[i] = i < nLdg ? xPtr[i * BLOCK] : static_cast<FT>(NEG_INF);
    }
  }

  // convert to ComputeT
  ComputeT xCompute[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = static_cast<ComputeT>(xData[i]);
  }

  // max reduction
  ComputeT max =
      GridXReduce<BLOCK, UNROLL, MAX_NTILES_PER_TASK, MaxReduceFunctor,
                  ComputeT>(xCompute, maxReduceWs, maxReduceFinishCount, taskId,
                            tileId, nTiles, maxNTiles);

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] = Expf(xCompute[i] - max);
  }

  // expsum reduction
  ComputeT expSum =
      GridXReduce<BLOCK, UNROLL, MAX_NTILES_PER_TASK, SumReduceFunctor,
                  ComputeT>(xCompute, sumReduceWs, sumReduceFinishCount, taskId,
                            tileId, nTiles, maxNTiles);

  ComputeT expSumRcp = Rcpf(expSum);
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCompute[i] *= expSumRcp;
  }

  // convert to FT
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xData[i] = static_cast<FT>(xCompute[i]);
  }

  // store the output
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      xPtr[i * BLOCK] = xData[i];
    }
  } else {
    uint32_t tileSize = length - tileOffset;
    int nLdg = tileSize > threadIdx.x
                   ? static_cast<int>(U32DivRU(tileSize - threadIdx.x, BLOCK))
                   : 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < nLdg) {
        xPtr[i * BLOCK] = xData[i];
      }
    }
  }
  return;
}

}  // namespace span
