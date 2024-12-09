/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm.cpp
 */

#include <memory.h>

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <>
void GemmWraper<float>(float* matrix_C, const float* matrix_A,
                       const float* matrix_B, const float* bias, int m, int n,
                       int k, bool transA, bool transB, int lda, int ldb,
                       int ldc, float alpha, float beta, const float* bin_res) {
  CBLAS_TRANSPOSE transA_ = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB_ = transB ? CblasTrans : CblasNoTrans;
  // assert beta = 0.f
  if (bias) {
    // broadcast_kernel(matrix_C, bias, m * n, m * n, n);
    parallel_for(
        m, [&](int j) { memcpy(matrix_C + j * n, bias, n * sizeof(float)); });
    beta = 1.f;
  }
  if (bin_res) {
    parallel_for(m, [&](int j) {
      memcpy(matrix_C + j * n, bin_res + j * n, n * sizeof(float));
    });
    beta = 1.f;
  }
  cblas_sgemm(CblasRowMajor, transA_, transB_, m, n, k, alpha, matrix_A, lda,
              matrix_B, ldb, beta, matrix_C, ldc);
}
template <>
void StridedBatchGemmWraper<float>(float* matrix_C, const float* matrix_A,
                                   const float* matrix_B, const float* bias,
                                   int m, int n, int k, bool transA,
                                   bool transB, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batch,
                                   const float* bin_res) {
  // int strideA = m * k;
  int strideA = 0;
  int strideB = k * n;
  int strideC = m * n;
  CBLAS_TRANSPOSE transA_ = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB_ = transB ? CblasTrans : CblasNoTrans;
  // assert beta = 0.f
  if (bias) {
    parallel_for(batch, [&](int i) {
      parallel_for(m, [&](int j) {
        memcpy(matrix_C + i * m * n + j * n, bias + i * n, n * sizeof(float));
      });
    });
    beta = 1.f;
  }
  if (bin_res) {
    parallel_for(batch, [&](int i) {
      parallel_for(m, [&](int j) {
        memcpy(matrix_C + i * m * n + j * n, bin_res + i * m * n + j * n,
               n * sizeof(float));
      });
    });
    beta = 1.f;
  }
#ifdef ALLSPARK_USE_MKL_
  cblas_sgemm_batch_strided(CblasRowMajor, transA_, transB_, m, n, k, alpha,
                            matrix_A, lda, strideA, matrix_B, ldb, strideB,
                            beta, matrix_C, ldc, strideC, batch);
#elif defined(ALLSPARK_USE_CBLAS_)
  cblas_sgemm_batch(CblasRowMajor, &transA_, &transB_, &m, &n, &k, &alpha,
                    &matrix_A, &lda, &matrix_B, &ldb, &beta, &matrix_C, &ldc, 1,
                    &batch);
#endif
}

}  // namespace cpu
}  // namespace allspark
