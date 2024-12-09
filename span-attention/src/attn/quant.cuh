/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    quant.cuh
 */

#pragma once

#include "utils/intrinsic.cuh"

namespace span {

template <int OUTER_UNROLL, int INNER_UNROLL, int INNER_STRIDE,
          typename QParamT>
DEVICE_FUNC void LoadQuantParam(
    QParamT (&qParam)[OUTER_UNROLL][INNER_UNROLL],
    const QParamT* (&qParamPtr)[OUTER_UNROLL],
    const bool (&seqIdxInBound)[OUTER_UNROLL][INNER_UNROLL], bool fullTile) {
  if (fullTile) {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        LdgCS(&qParam[i][j], qParamPtr[i] + j * INNER_STRIDE);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < INNER_UNROLL; ++j) {
        RegSet0(&qParam[i][j]);
        if (seqIdxInBound[i][j]) {
          LdgCS(&qParam[i][j], qParamPtr[i] + j * INNER_STRIDE);
        }
      }
    }
  }
  return;
}

template <int OUTER_UNROLL, int INNER_UNROLL, int PACK_SIZE, typename ComputeT,
          typename QParamT>
DEVICE_FUNC void InplaceDequant(
    ComputeT (&quantedVal)[OUTER_UNROLL][INNER_UNROLL][PACK_SIZE],
    const QParamT (&qParam)[OUTER_UNROLL][INNER_UNROLL]) {
  ComputeT zero[OUTER_UNROLL][INNER_UNROLL];
  ComputeT scale[OUTER_UNROLL][INNER_UNROLL];
  /*
   * NOTE: in most cases zero and scale are typed ComputeT,
   * so conversion is omitted.
   */
#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      zero[i][j] = static_cast<ComputeT>(qParam[i][j].zero);
      scale[i][j] = static_cast<ComputeT>(qParam[i][j].scale);
    }
  }

#pragma unroll
  for (int i = 0; i < OUTER_UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < INNER_UNROLL; ++j) {
      /**
       * transform '(KCompute - zero) * scale' to 'KCompute * scale -
       * (zero * scale)' to utilize FMA and prune instructions
       */
      zero[i][j] *= scale[i][j];
#pragma unroll
      for (int p = 0; p < PACK_SIZE; ++p) {
        quantedVal[i][j][p] = quantedVal[i][j][p] * scale[i][j] - zero[i][j];
      }
    }
  }
  return;
}

}  // namespace span
