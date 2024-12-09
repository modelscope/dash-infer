/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    check_cuda.h
 */

#pragma once

#include <span_attn.h>

#include <sstream>

#include "common/enums.h"
#include "common/error.h"
#include "common/logger.h"

namespace span {

#define SA_CHECK(expr)                                           \
  do {                                                           \
    SaStatus __error_code = (expr);                              \
    if (__error_code != SaStatus::SUCCESS) {                     \
      std::stringstream __error_msg;                             \
      __error_msg << "SpanAttn error (" << int(__error_code)     \
                  << "): " << __FILE__ << ":" << __LINE__ << " " \
                  << GetErrorString(__error_code);               \
      LOG(ERROR) << __error_msg.str() << std::endl;              \
      throw SpanAttnError(__error_code, __error_msg.str());      \
    }                                                            \
  } while (0)

#define SA_CHECK_RET(expr)                                       \
  do {                                                           \
    SaStatus __error_code = (expr);                              \
    if (__error_code != SaStatus::SUCCESS) {                     \
      std::stringstream __error_msg;                             \
      __error_msg << "SpanAttn error (" << int(__error_code)     \
                  << "): " << __FILE__ << ":" << __LINE__ << " " \
                  << GetErrorString(__error_code);               \
      LOG(ERROR) << __error_msg.str() << std::endl;              \
      return __error_code;                                       \
    }                                                            \
  } while (0)

#define SA_CHECK_CUDA(expr)                                                   \
  do {                                                                        \
    cudaError_t __error_code = (expr);                                        \
    if (__error_code != cudaSuccess) {                                        \
      std::stringstream __error_msg;                                          \
      __error_msg << "CUDA error (" << int(__error_code) << "): " << __FILE__ \
                  << ":" << __LINE__ << " "                                   \
                  << cudaGetErrorString(__error_code);                        \
      LOG(ERROR) << __error_msg.str() << std::endl;                           \
      throw SpanAttnError(SaStatus::CUDA_ERROR, __error_msg.str());           \
    }                                                                         \
  } while (0)

#define SA_CHECK_CUDA_RET(expr)                                               \
  do {                                                                        \
    cudaError_t __error_code = (expr);                                        \
    if (__error_code != cudaSuccess) {                                        \
      std::stringstream __error_msg;                                          \
      __error_msg << "CUDA error (" << int(__error_code) << "): " << __FILE__ \
                  << ":" << __LINE__ << " "                                   \
                  << cudaGetErrorString(__error_code);                        \
      LOG(ERROR) << __error_msg.str() << std::endl;                           \
      return SaStatus::CUDA_ERROR;                                            \
    }                                                                         \
  } while (0)

#define SA_CHECK_KERNEL()                                              \
  do {                                                                 \
    cudaError_t __error_code = cudaGetLastError();                     \
    if (__error_code != cudaSuccess) {                                 \
      std::stringstream __error_msg;                                   \
      __error_msg << "CUDA kernel launch error (" << int(__error_code) \
                  << "): " << __FILE__ << ":" << __LINE__ << " "       \
                  << cudaGetErrorString(__error_code);                 \
      LOG(ERROR) << __error_msg.str() << std::endl;                    \
      throw SpanAttnError(SaStatus::CUDA_ERROR, __error_msg.str());    \
    }                                                                  \
  } while (0)

#define SA_CHECK_KERNEL_RET()                                          \
  do {                                                                 \
    cudaError_t __error_code = cudaGetLastError();                     \
    if (__error_code != cudaSuccess) {                                 \
      std::stringstream __error_msg;                                   \
      __error_msg << "CUDA kernel launch error (" << int(__error_code) \
                  << "): " << __FILE__ << ":" << __LINE__ << " "       \
                  << cudaGetErrorString(__error_code);                 \
      LOG(ERROR) << __error_msg.str() << std::endl;                    \
      return SaStatus::CUDA_ERROR;                                     \
    }                                                                  \
  } while (0)

}  // namespace span
