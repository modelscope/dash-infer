/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax.cuh
 */

#pragma once

#include <limits>

#include "softmax_impl.cuh"
#include "utility/check_cuda.h"

namespace allspark {
namespace cuda {

namespace strided_softmax {
constexpr int SFTMX_BLOCK_PHASE1 = 512;
constexpr int SFTMX_BLOCK_PHASE2 = 32;
}  // namespace strided_softmax

template <typename T>
void StridedSoftmaxGetWorkspaceSize(size_t* wsInBytes, int taskNum,
                                    int stride) {
  using CompT = typename strided_softmax::ComputeT<T>::type;
  constexpr int BLOCK = strided_softmax::SFTMX_BLOCK_PHASE1;
  constexpr int UNROLL = 16 / sizeof(T);
  strided_softmax::getSoftmaxWorkspaceSize<BLOCK, UNROLL, T, CompT>(
      wsInBytes, taskNum, stride);
  return;
}

/**
 * @brief Compute batched strided softmax. Tasks can be of different lengths.
 * In-place supported, i.e. y can equal to x.
 *
 * @param[out] y Device output, shape { @c taskNum, @c stride }.
 * @param[in] x Device input, shape { @c taskNum, @c stride }.
 * @param[in] taskLenPtr Device array, valid lengths of each task in the batch.
 * @param[in] temperatures Device array. For temperature T, softmax computes
 * exp(x_i / T) / sum_i^n{ exp(x_i / T) }.
 * @param[in] workspace Pointer to the workspace.
 * @param[in] wsSizeInBytes Size of workspace in bytes, the minimum possible
 * size should be computed with #StridedSoftmaxGetWorkspaceSize.
 * @param[in] taskNum Batch size.
 * @param[in] stride Stride of input array.
 * @param[in] stream CUDA stream.
 */
template <typename T>
void StridedSoftmaxLauncher(T* y, const T* x, const int* taskLenPtr,
                            const float* temperatures, void* workspace,
                            size_t wsSizeInBytes, int taskNum, int stride,
                            cudaStream_t stream) {
  using CompT = typename strided_softmax::ComputeT<T>::type;

  if (workspace == nullptr) {
    throw std::runtime_error(
        "StridedSoftmaxLauncher: workspace cannot be nullptr");
  }

  size_t minWsSize(0);
  StridedSoftmaxGetWorkspaceSize<T>(&minWsSize, taskNum, stride);
  if (wsSizeInBytes < minWsSize) {
    throw std::runtime_error(
        "StridedSoftmaxLauncher: insufficient workspace size");
  }

  // prepare workspace
  CompT* maxVal = static_cast<CompT*>(workspace);
  CompT* rcpSumExp = maxVal + taskNum;
  void* wsPtr = static_cast<void*>(rcpSumExp + taskNum);

  // TODO: reduce max with temperature
  strided_softmax::launchReduceMax<strided_softmax::SFTMX_BLOCK_PHASE1,
                                   strided_softmax::SFTMX_BLOCK_PHASE2>(
      x, maxVal, static_cast<T*>(wsPtr), taskLenPtr, stride, taskNum, stream);
  strided_softmax::launchRcpSumExp<strided_softmax::SFTMX_BLOCK_PHASE1,
                                   strided_softmax::SFTMX_BLOCK_PHASE2>(
      x, maxVal, rcpSumExp, static_cast<CompT*>(wsPtr), taskLenPtr,
      temperatures, stride, taskNum, stream);
  strided_softmax::launchExpMulRcp<strided_softmax::SFTMX_BLOCK_PHASE1>(
      x, maxVal, rcpSumExp, y, taskLenPtr, temperatures, stride, taskNum,
      stream);
  return;
}

/**
 * @brief Compute batched strided log softmax. Tasks can be of different
 * lengths. In-place supported, i.e. y can equal to x.
 *
 * @param[out] y Device output, shape { @c taskNum, @c stride }.
 * @param[in] x Device input, shape { @c taskNum, @c stride }.
 * @param[in] taskLenPtr Device array, valid lengths of each task in the batch.
 * @param[in] temperatures Device array. For temperature T, softmax computes
 * exp(x_i / T) / sum_i^n{ exp(x_i / T) }.
 * @param[in] workspace Pointer to the workspace.
 * @param[in] wsSizeInBytes Size of workspace in bytes, the minimum possible
 * size should be computed with #StridedSoftmaxGetWorkspaceSize.
 * @param[in] taskNum Batch size.
 * @param[in] stride Stride of input array.
 * @param[in] stream CUDA stream.
 */
template <typename T>
void StridedLogSoftmaxLauncher(T* y, const T* x, const int* taskLenPtr,
                               const float* temperatures, void* workspace,
                               size_t wsSizeInBytes, int taskNum, int stride,
                               cudaStream_t stream) {
  using CompT = typename strided_softmax::ComputeT<T>::type;

  if (workspace == nullptr) {
    throw std::runtime_error(
        "StridedSoftmaxLauncher: workspace cannot be nullptr");
  }

  size_t minWsSize(0);
  StridedSoftmaxGetWorkspaceSize<T>(&minWsSize, taskNum, stride);
  if (wsSizeInBytes < minWsSize) {
    throw std::runtime_error(
        "StridedSoftmaxLauncher: insufficient workspace size");
  }

  // prepare workspace
  CompT* maxVal = static_cast<CompT*>(workspace);
  CompT* logSumExp = maxVal + taskNum;
  void* wsPtr = static_cast<void*>(logSumExp + taskNum);

  // TODO: reduce max with temperature
  strided_softmax::launchReduceMax<strided_softmax::SFTMX_BLOCK_PHASE1,
                                   strided_softmax::SFTMX_BLOCK_PHASE2>(
      x, maxVal, static_cast<T*>(wsPtr), taskLenPtr, stride, taskNum, stream);
  strided_softmax::launchLogSumExp<strided_softmax::SFTMX_BLOCK_PHASE1,
                                   strided_softmax::SFTMX_BLOCK_PHASE2>(
      x, maxVal, logSumExp, static_cast<CompT*>(wsPtr), taskLenPtr,
      temperatures, stride, taskNum, stream);
  strided_softmax::launchSubLog<strided_softmax::SFTMX_BLOCK_PHASE1>(
      x, maxVal, logSumExp, y, taskLenPtr, temperatures, stride, taskNum,
      stream);
  return;
}

}  // namespace cuda
}  // namespace allspark
