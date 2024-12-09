/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sparse_util.cpp
 */

#include "sparse_util.h"

#include <stdlib.h>

#include "cmath"

#ifdef ENABLE_FP16
#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#else
#include <common/float16.h>
#endif
#endif

namespace allspark {
namespace util {
template <>
void dense_to_csc_padding<float>(const float* denseMatrix, int m, int n,
                                 float* sparseMatrix, int* rowIdx,
                                 int* colOffset, int VECT) {
  float threshold = 1e-9;
  int nnz = 0;
  colOffset[0] = 0;

  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        rowIdx[nnz] = i;
        sparseMatrix[nnz] = denseMatrix[i * n + j];
        ++nnz;
      }
    }

    // this padding may generate more zero.
    if ((nnz - last_nnz) % VECT != 0) {
      int padding = VECT - ((nnz - last_nnz) % VECT);
      int last_idx = rowIdx[nnz - 1];
      for (int i = 0; i < padding; ++i) {
        rowIdx[nnz] = last_idx;
        sparseMatrix[nnz] = 0.0f;
        ++nnz;
      }
    }

    colOffset[j + 1] = nnz;
  }
}

#ifdef ENABLE_FP16
template <>
void dense_to_csc_padding<half>(const half* denseMatrix, int m, int n,
                                half* sparseMatrix, int* rowIdx, int* colOffset,
                                int VECT) {
  float threshold = 1e-9;
  int nnz = 0;
  colOffset[0] = 0;

  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        rowIdx[nnz] = i;
        sparseMatrix[nnz] = denseMatrix[i * n + j];
        ++nnz;
      }
    }

    // this padding may generate more zero.
    if ((nnz - last_nnz) % VECT != 0) {
      int padding = VECT - ((nnz - last_nnz) % VECT);
      int last_idx = rowIdx[nnz - 1];
      for (int i = 0; i < padding; ++i) {
        rowIdx[nnz] = last_idx;
        sparseMatrix[nnz] = 0.0f;
        ++nnz;
      }
    }

    colOffset[j + 1] = nnz;
  }
}
#endif

template <>
void dense_to_ell_padding<float>(const float* denseMatrix, int m, int n,
                                 int nnz_, float* sparseMatrix,
                                 unsigned short* rowIdx, int VECT) {
  float threshold = 1e-9;
  int max_c = nnz_ / n;
  int nnz = 0;
  unsigned short* cscRowIdx =
      (unsigned short*)malloc(sizeof(unsigned short) * nnz_);
  ;
  float* cscVal = (float*)malloc(sizeof(float) * nnz_);
  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        cscRowIdx[nnz] = i;
        cscVal[nnz] = denseMatrix[i * n + j];
        ++nnz;
      }
    }
    int padding = max_c - (nnz - last_nnz);
    unsigned short last_idx = rowIdx[nnz - 1];
    // this padding may generate more zero.
    for (int i = 0; i < padding; ++i) {
      cscRowIdx[nnz] = last_idx;
      cscVal[nnz] = 0.0f;
      ++nnz;
    }
  }
  int pos0 = 0;
  for (int i = 0; i < max_c / VECT; ++i) {
    int pos1 = i * VECT;
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < VECT; ++k) {
        rowIdx[pos0] = cscRowIdx[pos1 + k];
        sparseMatrix[pos0] = cscVal[pos1 + k];
        pos0++;
      }
      pos1 += max_c;
    }
  }
  free(cscRowIdx);
  free(cscVal);
}

#ifdef ENABLE_FP16
template <>
void dense_to_ell_padding<half>(const half* denseMatrix, int m, int n, int nnz_,
                                half* sparseMatrix, unsigned short* rowIdx,
                                int VECT) {
  float threshold = 1e-9;
  int max_c = nnz_ / n;
  int nnz = 0;
  unsigned short* cscRowIdx =
      (unsigned short*)malloc(sizeof(unsigned short) * nnz_);
  ;
  half* cscVal = (half*)malloc(sizeof(half) * nnz_);
  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        cscRowIdx[nnz] = i;
        cscVal[nnz] = denseMatrix[i * n + j];
        ++nnz;
      }
    }
    int padding = max_c - (nnz - last_nnz);
    unsigned short last_idx = rowIdx[nnz - 1];
    // this padding may generate more zero.
    for (int i = 0; i < padding; ++i) {
      cscRowIdx[nnz] = last_idx;
      cscVal[nnz] = 0.0f;
      ++nnz;
    }
  }
  int pos0 = 0;
  for (int i = 0; i < max_c / VECT; ++i) {
    int pos1 = i * VECT;
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < VECT; ++k) {
        rowIdx[pos0] = cscRowIdx[pos1 + k];
        sparseMatrix[pos0] = cscVal[pos1 + k];
        pos0++;
      }
      pos1 += max_c;
    }
  }
  free(cscRowIdx);
  free(cscVal);
}
#endif
template <>
int get_nnz<float>(const float* denseMatrix, int m, int n, int VECT) {
  float threshold = 1e-9;
  int nnz = 0;
  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        ++nnz;
      }
    }

    // this padding may generate more zero.

    if ((nnz - last_nnz) % VECT != 0) {
      int padding = VECT - ((nnz - last_nnz) % VECT);
      for (int i = 0; i < padding; ++i) {
        ++nnz;
      }
    }
  }
  return nnz;
}
#ifdef ENABLE_FP16
template <>
int get_nnz<half>(const half* denseMatrix, int m, int n, int VECT) {
  float threshold = 1e-9;
  int nnz = 0;
  for (int j = 0; j < n; ++j) {
    int last_nnz = nnz;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        ++nnz;
      }
    }

    // this padding may generate more zero.
    if ((nnz - last_nnz) % VECT != 0) {
      int padding = VECT - ((nnz - last_nnz) % VECT);
      for (int i = 0; i < padding; ++i) {
        ++nnz;
      }
    }
  }
  return nnz;
}
#endif
template <>
int get_nnz_ell<float>(const float* denseMatrix, int m, int n, int VECT) {
  float threshold = 1e-9;
  int max_c = 0;
  for (int j = 0; j < n; ++j) {
    int sum = 0;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        ++sum;
      }
    }
    if (sum > max_c) {
      max_c = sum;
    }
  }
  if (max_c % VECT != 0) {
    max_c += VECT - max_c % VECT;
  }
  return max_c * n;
}
#ifdef ENABLE_FP16
template <>
int get_nnz_ell<half>(const half* denseMatrix, int m, int n, int VECT) {
  float threshold = 1e-9;
  int max_c = 0;
  for (int j = 0; j < n; ++j) {
    int sum = 0;
    for (int i = 0; i < m; ++i) {
      if (std::fabs(static_cast<float>(denseMatrix[i * n + j])) > threshold) {
        ++sum;
      }
    }
    if (sum > max_c) {
      max_c = sum;
    }
  }
  if (max_c % VECT != 0) {
    max_c += VECT - max_c % VECT;
  }
  return max_c * n;
}
#endif
}  // namespace util
}  // namespace allspark
