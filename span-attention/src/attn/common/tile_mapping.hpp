/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tile_mapping.hpp
 */

#pragma once

#include <span_attn.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "attn/common/utils.hpp"
#include "common/logger.h"

namespace span {

struct TileMapping {
  /**
   * @brief Index inside the batch.
   */
  uint16_t batchId;
  /**
   * @brief Tile index inside the sequence.
   */
  uint16_t tileId;

  TileMapping() = default;

  TileMapping(uint16_t batchId, uint16_t tileId)
      : batchId(batchId), tileId(tileId) {}
};

template <typename SrcIt>
SaStatus BuildTileMapping(std::vector<TileMapping>& mappings,
                          const SrcIt& seqLengths, int TILE_SIZE, int batch,
                          int maxLength) {
  mappings.reserve(batch * U32DivRU(maxLength, TILE_SIZE));
  for (uint16_t i = 0; i < batch; ++i) {
    uint32_t nTiles = U32DivRU(*(seqLengths + i), TILE_SIZE);
    if (nTiles > std::numeric_limits<uint16_t>::max()) {
      LOG(ERROR) << "BuildTileMapping: nTiles=" << nTiles
                 << " exceeds uint16_t, with batch=" << batch
                 << " maxLength=" << maxLength << " TILE_SIZE=" << TILE_SIZE
                 << std::endl;
      return SaStatus::EXCEED_LIMIT_ERROR;
    }
    for (uint16_t j = 0; j < nTiles; ++j) {
      mappings.emplace_back(i, j);
    }
  }
  return SaStatus::SUCCESS;
}

}  // namespace span
