/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    check_cuda.h
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#define AS_CHECK_CUDA(cmd)                                                     \
  do {                                                                         \
    cudaError_t cuda_status = cmd;                                             \
    if (cuda_status != cudaSuccess) {                                          \
      std::string err_str = cudaGetErrorString(cuda_status);                   \
      std::cerr << "Failed: " << __FILE__ << ":" << __LINE__ << " " << err_str \
                << std::endl;                                                  \
      throw std::runtime_error("[Cuda error]" + err_str);                      \
    }                                                                          \
  } while (0)

#define AS_CHECK_CUDA_LAST_ERROR()                                             \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::string err_str = cudaGetErrorString(err);                           \
      std::cerr << "Failed: " << __FILE__ << ":" << __LINE__ << " " << err_str \
                << std::endl;                                                  \
      throw std::runtime_error("[Cuda error]" + err_str);                      \
    }                                                                          \
  } while (0)
