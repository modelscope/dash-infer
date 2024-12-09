/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    config.h
 */

#pragma once

namespace span {

constexpr int WARP_SIZE = 32;
constexpr int CTA_MAX_SIZE = 1024;
constexpr int CTA_MAX_WARP = CTA_MAX_SIZE / WARP_SIZE;

constexpr int MEM_ALIGN = 128;

template <int HEAD_SIZE, typename FType>
struct QKGemvConfig {
  static constexpr int BLOCK = 256;
  static constexpr int UNROLL = 4;

  static constexpr int PACK_SIZE = 16 / sizeof(FType);
  static constexpr int BLOCK_X = HEAD_SIZE / PACK_SIZE;  // 8-half vectorized
  static constexpr int BLOCK_Y = BLOCK / BLOCK_X;
  static constexpr int SEQ_TILE_SIZE = BLOCK_Y * UNROLL;
};

template <int HEAD_SIZE, typename FType>
struct QKVGemvConfig {
  static constexpr int BLOCK = 256;
  static constexpr int UNROLL = 8;

  static constexpr int PACK_SIZE = 16 / sizeof(FType);
  static constexpr int BLOCK_X = HEAD_SIZE / PACK_SIZE;  // 8-half vectorized
  static constexpr int BLOCK_Y = BLOCK / BLOCK_X;
  static constexpr int SEQ_TILE_SIZE = BLOCK_Y * UNROLL;
};

template <typename FType>
struct SoftmaxConfig {
  using ComputeType = float;

  // 1-CTA single pass softmax configuration
  static constexpr int SOFTMAX_MAX_BLOCK = 1024;
  static constexpr int SOFTMAX_MAX_UNROLL = 16;

  // multi-CTA single pass softmax configuration
  static constexpr int TILED_SOFTMAX_BLOCK = 256;
  static constexpr int TILED_SOFTMAX_UNROLL = 32;
  static constexpr int TILED_SOFTMAX_MAX_NTILES_PER_TASK = 32;
  static constexpr int TILED_SOFTMAX_SEQ_TILE_SIZE =
      TILED_SOFTMAX_BLOCK * TILED_SOFTMAX_UNROLL;
};

}  // namespace span
