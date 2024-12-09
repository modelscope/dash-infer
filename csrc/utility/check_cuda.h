/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    check_cuda.h
 */

#pragma once

#include <execinfo.h>
#include <glog/logging.h>
#include <interface/allspark_check.h>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
// clang-format off
#include <cublas_v2.h>
#include <cublas_api.h>
// clang-format on
#include <nccl.h>
#endif

#include "check.h"

#ifdef ENABLE_CUDA
#define AS_CHECK_CUDA(cmd)                                           \
  do {                                                               \
    cudaError_t cuda_status = cmd;                                   \
    if (cuda_status != cudaSuccess) {                                \
      std::string err_str = cudaGetErrorString(cuda_status);         \
      LOG(ERROR) << "Failed: " << __FILE__ << ":" << __LINE__ << " " \
                 << err_str;                                         \
      print_backtrace();                                             \
      throw AsException("[Cuda error]" + err_str);                   \
    }                                                                \
  } while (0)

#define AS_CHECK_CUDA_LAST_ERROR()                                   \
  do {                                                               \
    cudaError_t err = cudaGetLastError();                            \
    if (err != cudaSuccess) {                                        \
      std::string err_str = cudaGetErrorString(err);                 \
      LOG(ERROR) << "Failed: " << __FILE__ << ":" << __LINE__ << " " \
                 << err_str;                                         \
      print_backtrace();                                             \
      throw AsException("[Cuda error]" + err_str);                   \
    }                                                                \
  } while (0)

#define AS_CHECK_CUBLAS(cmd)                                           \
  do {                                                                 \
    cublasStatus_t cublas_status = cmd;                                \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                      \
      /* why cublas_api.h has no cublasGetErrorString() ??? */         \
      /*std::string err_str = cublasGetErrorString(cublas_status);*/   \
      std::string err_str =                                            \
          "cublasStatus_t code=" + std::to_string(int(cublas_status)); \
      LOG(ERROR) << "Failed:  " << __FILE__ << ":" << __LINE__ << " "  \
                 << err_str;                                           \
      print_backtrace();                                               \
      throw AsException("[CuBlas error]" + err_str);                   \
    }                                                                  \
  } while (0)

#define AS_CHECK_HIEDNN(expr)                                                  \
  do {                                                                         \
    hiednnStatus_t err = (expr);                                               \
    if (err != HIEDNN_STATUS_SUCCESS) {                                        \
      std::string err_str = "hiednnStatus_t code=" + std::to_string(int(err)); \
      LOG(ERROR) << "Failed:  " << __FILE__ << ":" << __LINE__ << " "          \
                 << err_str;                                                   \
      print_backtrace();                                                       \
      throw AsException("[HIE-DNN error]" + err_str);                          \
    }                                                                          \
  } while (0)

#define AS_CHECK_CUDRIVER(cmd)                                       \
  do {                                                               \
    CUresult cu_status = cmd;                                        \
    if (cu_status != CUDA_SUCCESS) {                                 \
      std::string err_str;                                           \
      cuGetErrorString(cu_status, &err_str.c_str());                 \
      LOG(ERROR) << "Failed: " << __FILE__ << ":" << __LINE__ << " " \
                 << err_str;                                         \
      print_backtrace();                                             \
      throw AsException("[Cuda driver error]" + err_str);            \
    }                                                                \
  } while (0)

#define AS_CHECK_NCCL(cmd)                                            \
  do {                                                                \
    ncclResult_t r = cmd;                                             \
    if (r != ncclSuccess) {                                           \
      std::string err_str = ncclGetErrorString(r);                    \
      LOG(ERROR) << "Failed:  " << __FILE__ << ":" << __LINE__ << " " \
                 << err_str;                                          \
      print_backtrace();                                              \
      throw AsException("[nccl error]" + err_str);                    \
    }                                                                 \
  } while (0)
#ifdef ENABLE_CUSPARSELT
#define AS_CHECK_CUSPARSE(func)                                           \
  do {                                                                    \
    cusparseStatus_t status = (func);                                     \
    if (status != CUSPARSE_STATUS_SUCCESS) {                              \
      printf("CUSPARSELT API failed at %s line %d with error: %s (%d)\n", \
             __FILE__, __LINE__, cusparseGetErrorString(status), status); \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)
#endif

#endif  // ENABLE_CUDA
