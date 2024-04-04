/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sparse_util.h
 */

#pragma once

namespace allspark {
namespace util {

template <typename T>
void dense_to_csc_padding(const T* denseMatrix, int m, int n, T* sparseMatrix,
                          int* rowIdx, int* colOffset, int VECT);

template <typename T>
void dense_to_ell_padding(const T* denseMatrix, int m, int n, int nnz,
                          T* sparseMatrix, unsigned short* rowIdx, int VECT);

template <typename T>
int get_nnz(const T* denseMatrix, int m, int n, int VECT);

template <typename T>
int get_nnz_ell(const T* denseMatrix, int m, int n, int VECT);
}  // namespace util
}  // namespace allspark
