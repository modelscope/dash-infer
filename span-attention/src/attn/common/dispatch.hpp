/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dispatch.hpp
 */

#pragma once

#include <span_attn.h>

#include <utility>

#include "common/enums.h"
#include "common/logger.h"

namespace span {

template <QuantMode QMODE, int CHUNK_SIZE, int HEAD_SIZE, typename T,
          template <QuantMode, int, int, int, class> class Func, class... Args>
[[nodiscard]] SaStatus DispatchGroupSize(int headsPerGroup, Args&&... args) {
  if (headsPerGroup <= 0) {
    LOG(ERROR) << "query heads per group should be a positive integer, got "
               << headsPerGroup << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (headsPerGroup <= 1) {
    // MHA
    return Func<QMODE, CHUNK_SIZE, HEAD_SIZE, 1, T>()(
        std::forward<Args>(args)...);
  } else if (headsPerGroup <= 8) {
    // most GQA settings
    return Func<QMODE, CHUNK_SIZE, HEAD_SIZE, 8, T>()(
        std::forward<Args>(args)...);
  } else if (headsPerGroup <= 32) {
    // fallback
    return Func<QMODE, CHUNK_SIZE, HEAD_SIZE, 32, T>()(
        std::forward<Args>(args)...);
  } else {
    LOG(ERROR) << "too many query heads per group: " << headsPerGroup
               << ", should be at most 32" << std::endl;
    return SaStatus::PARAM_ERROR;
  }
}

template <QuantMode QMODE, int CHUNK_SIZE, typename T,
          template <QuantMode, int, int, int, class> class Func, class... Args>
[[nodiscard]] SaStatus DispatchHeadSize(int headSize, int headsPerGroup,
                                        Args&&... args) {
  switch (headSize) {
    case 128:
      return DispatchGroupSize<QMODE, CHUNK_SIZE, 128, T, Func>(
          headsPerGroup, std::forward<Args>(args)...);
    default:
      LOG(ERROR) << "unsupported head size: " << headSize << std::endl;
      return SaStatus::PARAM_ERROR;
  }
}

template <QuantMode QMODE, typename T,
          template <QuantMode, int, int, int, class> class Func, class... Args>
[[nodiscard]] SaStatus DispatchChunkSize(int chunkSize, int headSize,
                                         int headsPerGroup, Args&&... args) {
  switch (chunkSize) {
    case 16:
      return DispatchHeadSize<QMODE, 16, T, Func>(headSize, headsPerGroup,
                                                  std::forward<Args>(args)...);
    case 32:
      return DispatchHeadSize<QMODE, 32, T, Func>(headSize, headsPerGroup,
                                                  std::forward<Args>(args)...);
    case 64:
      return DispatchHeadSize<QMODE, 64, T, Func>(headSize, headsPerGroup,
                                                  std::forward<Args>(args)...);
    case 128:
      return DispatchHeadSize<QMODE, 128, T, Func>(headSize, headsPerGroup,
                                                   std::forward<Args>(args)...);
    default:
      LOG(ERROR) << "unsupported span length (tokens per span): " << chunkSize
                 << std::endl;
      return SaStatus::PARAM_ERROR;
  }
}

template <typename T, template <QuantMode, int, int, int, class> class Func,
          class... Args>
[[nodiscard]] SaStatus DispatchCacheMode(QuantMode kvQuantMode, int chunkSize,
                                         int headSize, int headsPerGroup,
                                         Args&&... args) {
  switch (kvQuantMode) {
    case QuantMode::NONE: {
      return DispatchChunkSize<QuantMode::NONE, T, Func>(
          chunkSize, headSize, headsPerGroup, std::forward<Args>(args)...);
    }
    case QuantMode::I8: {
      return DispatchChunkSize<QuantMode::I8, T, Func>(
          chunkSize, headSize, headsPerGroup, std::forward<Args>(args)...);
    }
    case QuantMode::U4: {
      return DispatchChunkSize<QuantMode::U4, T, Func>(
          chunkSize, headSize, headsPerGroup, std::forward<Args>(args)...);
    }
    default:
      LOG(ERROR) << "unsupported KV cache quantization mode: "
                 << to_string(kvQuantMode) << std::endl;
      return SaStatus::PARAM_ERROR;
  }
}

}  // namespace span
