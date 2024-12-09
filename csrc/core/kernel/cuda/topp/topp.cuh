/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp.cuh
 */

#pragma once

#include "core/kernel/cuda/cuda_kernel.h"
#include "topp_impl.cuh"
#include "utility/check_cuda.h"

#ifdef ALLSPARK_DEBUG_MODE
#define CUDA_KERNEL_TOPP_DBG_SYNC(stream) \
  AS_CHECK_CUDA(cudaStreamSynchronize((stream)))
#else
#define CUDA_KERNEL_TOPP_DBG_SYNC(stream) ((void)(stream))
#endif  // #ifdef ALLSPARK_DEBUG_MODE

namespace allspark {
namespace cuda {

template <typename T>
void TopPSoftmaxGetWorkspaceSize(size_t* sizeInBytes, int batch_size,
                                 int length, bool input_is_sorted) {
  constexpr bool SORT_ASCEND = false;

  // sorting
  size_t sortWsBytes(0);
  if (!input_is_sorted) {
    topp::GetCubSortWorkspaceSize<SORT_ASCEND, T, int64_t>(&sortWsBytes,
                                                           batch_size, length);
    sortWsBytes = topp::AlignUpPow2<sizeof(int64_t)>(sortWsBytes);
    // input indices
    sortWsBytes += sizeof(int64_t) * length * batch_size;
  }

  // softmax
  size_t softmaxWsBytes(0);
  StridedSoftmaxGetWorkspaceSize<T>(&softmaxWsBytes, batch_size, length);

  *sizeInBytes = std::max(sortWsBytes, softmaxWsBytes);
  return;
}

/**
 * @brief In-place supported, i.e., topp_probs can be the same pointer as
 * input_logits. If input_is_sorted == true, topp_indices is untouched.
 *
 * @param[in,out] topp_count Input as the number of valid items in each task,
 * output as the number of remaining items in each task after top-p.
 * @param[out] topp_probs Adjusted probabilities of top-p filtered items.
 * @param[out] topp_indices Top-p indices.
 * @param[in] input_logits Input logits.
 * @param[in] p_values Cut-off p values.
 * @param[in] temperatures Temperatures for softmax.
 * @param[in] temp_probs Temporary space for computing probabilities.
 * @param[in] workspace Workpace pointer.
 * @param[in] ws_size_in_bytes Workspace size in bytes.
 * @param[in] batch_size Batch size.
 * @param[in] length Length of each task.
 * @param[in] input_is_sorted Whether input logits have been sorted with index.
 * @param[in] handle HIE-DNN handle.
 * @param[in] stream CUDA stream.
 */
template <typename T>
void TopPSoftmaxLauncher(int* topp_count, T* topp_probs, int* topp_indices,
                         const T* input_logits, const float* p_values,
                         const float* temperatures, T* temp_probs,
                         void* workspace, size_t ws_size_in_bytes,
                         int batch_size, int length, bool input_is_sorted,
                         hiednnCudaHandle_t handle, cudaStream_t stream) {
  constexpr bool SORT_ASCEND = false;

  // clear previous errors
  AS_CHECK_CUDA_LAST_ERROR();

  if (workspace == nullptr) {
    throw std::runtime_error(
        "TopPSoftmaxLauncher: workspace cannot be nullptr");
  }

  size_t minWsBytes(0);
  TopPSoftmaxGetWorkspaceSize<T>(&minWsBytes, batch_size, length,
                                 input_is_sorted);
  if (ws_size_in_bytes < minWsBytes) {
    throw std::runtime_error(
        "TopPSoftmaxLauncher: insufficient workspace size");
  }

  // dbg sync before any kernel
  CUDA_KERNEL_TOPP_DBG_SYNC(stream);

  const T* sortedLogits = input_logits;
  T* sortedAllProbBuf = temp_probs;
  if (!input_is_sorted) {
    size_t sortWsBytes(0);
    topp::GetCubSortWorkspaceSize<SORT_ASCEND, T, int64_t>(&sortWsBytes,
                                                           batch_size, length);
    sortWsBytes = topp::AlignUpPow2<sizeof(int64_t)>(sortWsBytes);
    int* input_indices =
        reinterpret_cast<int*>(static_cast<char*>(workspace) + sortWsBytes);

    topp::LaunchGenIndices(input_indices, static_cast<int*>(workspace), length,
                           batch_size, stream);
    CUDA_KERNEL_TOPP_DBG_SYNC(stream);
    topp::CubSortLogits<SORT_ASCEND>(temp_probs, topp_indices, input_logits,
                                     input_indices, workspace, sortWsBytes,
                                     batch_size, length, stream);
    CUDA_KERNEL_TOPP_DBG_SYNC(stream);

    sortedLogits = temp_probs;
    sortedAllProbBuf = topp_probs;
  }

  //* NOTE: no need to touch `topp_indices` from now on

  StridedSoftmaxLauncher(sortedAllProbBuf, sortedLogits, topp_count,
                         temperatures, workspace, ws_size_in_bytes, batch_size,
                         length, stream);
  CUDA_KERNEL_TOPP_DBG_SYNC(stream);

  //! NOTE: rely on in-place prefix sum
  topp::HiednnPrefixSum(sortedAllProbBuf, sortedAllProbBuf, length, batch_size,
                        handle, stream);
  CUDA_KERNEL_TOPP_DBG_SYNC(stream);

  // determine top-p values for each task
  topp::LaunchCutoff(topp_count, sortedAllProbBuf, p_values, topp_count,
                     batch_size, length, stream);
  CUDA_KERNEL_TOPP_DBG_SYNC(stream);

  StridedSoftmaxLauncher(topp_probs, sortedLogits, topp_count, temperatures,
                         workspace, ws_size_in_bytes, batch_size, length,
                         stream);
  CUDA_KERNEL_TOPP_DBG_SYNC(stream);
  return;
}

}  // namespace cuda
}  // namespace allspark
