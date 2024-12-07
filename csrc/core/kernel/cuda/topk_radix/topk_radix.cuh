/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topk_radix.cuh
 */

#pragma once

#include <thrust/functional.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "bitonic_sort.cuh"
#include "radix_select.cuh"
#include "utility/check_cuda.h"
#include "utils.cuh"

// #define RADIX_TOPK_KERNEL_DEBUG

namespace allspark {
namespace cuda {
namespace radix_topk {

static constexpr int MAX_GRID_SIZE = 1280;

#ifdef ALLSPARK_DEBUG_MODE
#define RADIX_TOPK_KERNEL_CHECK_SYNC(stream) \
  AS_CHECK_CUDA(cudaStreamSynchronize((stream)))
#else
#define RADIX_TOPK_KERNEL_CHECK_SYNC(stream) (void)(stream)
#endif  // ALLSPARK_DEBUG_MODE

/**
 * @brief Entry function of radix top-k selection, requiring @p K <= @p stride .
 *
 * @tparam IdxType index type
 * @tparam LARGEST true: largest k; false: smallest k
 * @tparam ASCEND true: sorted ascend; false: sorted descend
 * @tparam WITHSCALE true: scaling; false: no scaling
 * @tparam WITHIDXIN true: input indices; false: no input indices (default:
 * false)
 * @tparam ValType value type
 *
 * @param[in] valIn input value ptr
 * @param[in] idxIn input indices ptr
 * @param[out] valOut output value ptr
 * @param[out] idxOut output indices ptr
 * @param[in] workSpace workspace ptr
 * @param[in] stride max length of task in the batch
 * @param[in] taskNum batch size
 * @param[in] K k for top-k
 * @param[in] stream CUDA stream
 *
 * @throws @c std::runtime_error if @p workSpace is @c nullptr .
 * @throws @c std::runtime_error if @p K > @p stride .
 */
template <typename IdxType, bool LARGEST, bool ASCEND, bool WITHSCALE,
          bool WITHIDXIN = 0, typename ValType>
void topKRadixSelectL(const ValType* valIn, const IdxType* idxIn,
                      ValType* valOut, IdxType* idxOut, void* workSpace,
                      const int& stride, const int& taskNum, const int& K,
                      cudaStream_t stream) {
  using CompT = typename ComputeT<ValType>::type;

  static_assert(sizeof(ValType) <= 16,
                "radix topk: requires sizeof(ValType) <= 16");
  constexpr int PACKSIZE = 16 / sizeof(ValType);

  if (K > stride) {
    throw std::runtime_error("radix topk: requires K <= stride");
  }
  if (workSpace == nullptr) {
    throw std::runtime_error("radix topk: workspace ptr should not be null");
  }

  // clear previous error if any
  AS_CHECK_CUDA_LAST_ERROR();

  // set valBuffer histPtr globalCountPtr
  CompT* valBuffer[2]{static_cast<CompT*>(workSpace),
                      static_cast<CompT*>(workSpace) + taskNum * stride};

  int* histPtr = reinterpret_cast<int*>(valBuffer[1] + taskNum * stride);
  int* globalCountPtr = histPtr + (1 << 12) * taskNum;

  // set taskOffsetPtr
  int* taskOffsetPtr = globalCountPtr + taskNum;
  std::vector<int> taskOffset(taskNum + 1);
  taskOffset[0] = 0;
  for (int i = 0; i < taskNum; ++i) {
    taskOffset[i + 1] = taskOffset[i] + stride;
  }
  AS_CHECK_CUDA(cudaMemcpyAsync(taskOffsetPtr, taskOffset.data(),
                                sizeof(int) * (taskNum + 1),
                                cudaMemcpyHostToDevice, stream));

  // set taskLenPtr, zero-init
  int* taskLenPtr[2]{taskOffsetPtr + taskNum + 1,
                     taskOffsetPtr + 2 * taskNum + 1};
  AS_CHECK_CUDA(
      cudaMemsetAsync(taskLenPtr[0], 0, sizeof(int) * taskNum * 2, stream));

  // set kPtr
  int* kPtr = taskLenPtr[1] + taskNum;
  std::vector<int> tmpK(taskNum);
  for (int i = 0; i < taskNum; ++i) {
    tmpK[i] = K;
  }
  AS_CHECK_CUDA(cudaMemcpyAsync(kPtr, tmpK.data(), sizeof(int) * taskNum,
                                cudaMemcpyHostToDevice, stream));

  // set binIdPtr
  int* binIdPtr = kPtr + taskNum;

  const int minTaskLen = stride;
  int gridSizeX =
      std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (1024 * PACKSIZE)));
  int gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);

  std::vector<int> taskLenHost(taskNum);

  // clear hist and globalCount
  AS_CHECK_CUDA(cudaMemsetAsync(
      histPtr, 0, sizeof(int) * ((1 << 12) + 1) * taskNum, stream));

  // first iter
  // get hist
  countBinEx<1024, 0, 20, PACKSIZE, WITHSCALE, LARGEST>
      <<<dim3(gridSizeX, gridSizeY), 1024, 0, stream>>>(valIn, taskOffsetPtr,
                                                        histPtr, taskNum);
  AS_CHECK_CUDA_LAST_ERROR();
  RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

  // select bin
  selectBin<LARGEST, 1024, (1 << 12)>
      <<<taskNum, 1024, 0, stream>>>(histPtr, binIdPtr, kPtr, taskLenPtr[0]);
  AS_CHECK_CUDA_LAST_ERROR();
  RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

  // select candidate
  gridSizeX =
      std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (256 * PACKSIZE)));
  gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);
  selectCandidateEx<256, 0, 20, PACKSIZE, WITHSCALE, LARGEST>
      <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(
          valIn, valBuffer[0], globalCountPtr, binIdPtr, taskOffsetPtr, stride,
          taskNum);
  AS_CHECK_CUDA_LAST_ERROR();
  RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

  // update taskLen
  AS_CHECK_CUDA(cudaMemcpyAsync(taskLenHost.data(), taskLenPtr[0],
                                sizeof(int) * taskNum, cudaMemcpyDeviceToHost,
                                stream));
  AS_CHECK_CUDA(cudaStreamSynchronize(stream));
  int maxTaskLen = *std::max_element(taskLenHost.begin(), taskLenHost.end());

  int flag = 0;

  // second iter
  if (maxTaskLen != 1) {
    // clear hist and globalCount
    int gridSize = (maxTaskLen + 1023) / 1024;
    AS_CHECK_CUDA(cudaMemsetAsync(
        histPtr, 0, sizeof(int) * ((1 << 12) + 1) * taskNum, stream));

    // get hist
    countBin<1024, 12, 20><<<dim3(gridSize, taskNum), 1024, 0, stream>>>(
        valBuffer[flag], taskLenPtr[flag], histPtr, stride);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    // select bin
    selectBin<LARGEST, 1024, (1 << 12)><<<taskNum, 1024, 0, stream>>>(
        histPtr, binIdPtr, kPtr, taskLenPtr[flag ^ 1]);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    // shift candidate element
    gridSize = (maxTaskLen + 255) / 256;
    selectCandidate<256, 12, 20><<<dim3(gridSize, taskNum), 256, 0, stream>>>(
        valBuffer[flag], valBuffer[flag ^ 1], globalCountPtr, binIdPtr,
        taskLenPtr[flag], stride);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    // update taskLen
    AS_CHECK_CUDA(cudaMemcpyAsync(taskLenHost.data(), taskLenPtr[flag ^ 1],
                                  sizeof(int) * taskNum, cudaMemcpyDeviceToHost,
                                  stream));
    AS_CHECK_CUDA(cudaStreamSynchronize(stream));
    maxTaskLen = *std::max_element(taskLenHost.begin(), taskLenHost.end());

    flag ^= 1;
  }

  // third iter
  if (maxTaskLen != 1) {
    int gridSize = (maxTaskLen + 255) / 256;
    // clear hist and globalCount
    AS_CHECK_CUDA(cudaMemsetAsync(histPtr + 3840 * taskNum, 0,
                                  sizeof(int) * ((1 << 8) + 1) * taskNum,
                                  stream));

    // get hist
    countBin<256, 24, 24><<<dim3(gridSize, taskNum), 256, 0, stream>>>(
        valBuffer[flag], taskLenPtr[flag], histPtr + 3840 * taskNum, stride);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    // select bin
    selectBin<LARGEST, 256, (1 << 8)><<<taskNum, 256, 0, stream>>>(
        histPtr + 3840 * taskNum, binIdPtr, kPtr, taskLenPtr[flag ^ 1]);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    // shift candidate element
    gridSize = (maxTaskLen + 255) / 256;
    selectCandidate<256, 24, 24><<<dim3(gridSize, taskNum), 256, 0, stream>>>(
        valBuffer[flag], valBuffer[flag ^ 1], globalCountPtr, binIdPtr,
        taskLenPtr[flag], stride);
    AS_CHECK_CUDA_LAST_ERROR();
    RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

    flag ^= 1;
  }

  // clear globalCount
  AS_CHECK_CUDA(
      cudaMemsetAsync(globalCountPtr, 0, sizeof(int) * taskNum, stream));

  // select result
  gridSizeX =
      std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (256 * PACKSIZE)));
  gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);

#define RADIX_TOPK_CALL_FILTER(CACHE_SIZE)                                 \
  do {                                                                     \
    filter<LARGEST, 256, PACKSIZE, (CACHE_SIZE), WITHSCALE, WITHIDXIN>     \
        <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(                  \
            valIn, idxIn, valBuffer[flag], valOut, idxOut, globalCountPtr, \
            kPtr, taskOffsetPtr, stride, K, taskNum);                      \
  } while (0)

  if (K <= 128) {
    RADIX_TOPK_CALL_FILTER(128);
  } else if (K <= 256) {
    RADIX_TOPK_CALL_FILTER(256);
  } else if (K <= 512) {
    RADIX_TOPK_CALL_FILTER(512);
  } else if (K <= 1024) {
    RADIX_TOPK_CALL_FILTER(1024);
  } else {
    // for K > 1024, use a general filter
    filter_general<LARGEST, 256, PACKSIZE, WITHSCALE, WITHIDXIN>
        <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(
            valIn, idxIn, valBuffer[flag], valOut, idxOut, globalCountPtr, kPtr,
            taskOffsetPtr, stride, K, taskNum);
  }
#undef RADIX_TOPK_CALL_FILTER
  AS_CHECK_CUDA_LAST_ERROR();
  RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

#define RADIX_TOPK_CALL_BITONIC_SORT(LENGTH, BLOCK)                          \
  do {                                                                       \
    bitonic::bitonicSort_##LENGTH<ASCEND>                                    \
        <<<taskNum, (BLOCK), 0, stream>>>(valOut, idxOut, K, taskOffsetPtr); \
  } while (0)

  if (K <= 128) {
    RADIX_TOPK_CALL_BITONIC_SORT(128, 128);
  } else if (K <= 256) {
    RADIX_TOPK_CALL_BITONIC_SORT(256, 256);
  } else if (K <= 512) {
    RADIX_TOPK_CALL_BITONIC_SORT(512, 512);
  } else if (K <= 1024) {
    RADIX_TOPK_CALL_BITONIC_SORT(1024, 1024);
  } else if (K <= 2048) {
    RADIX_TOPK_CALL_BITONIC_SORT(2048, 1024);
  } else if (K <= 4096) {
    RADIX_TOPK_CALL_BITONIC_SORT(4096, 1024);
  } else {
    // sort outputs with thrust::sort
#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
#ifdef CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
    // preprocessing to ensure order
    uint32_t nPass = (K + 1024 * PACKSIZE - 1) / (1024 * PACKSIZE);
    convertNanInPlace<1024, PACKSIZE, ASCEND>
        <<<dim3(taskNum, nPass), 1024, 0, stream>>>(valOut, K);
    AS_CHECK_CUDA_LAST_ERROR();
#endif  // CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
#endif  // CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
    for (int i = 0; i < taskNum; ++i) {
      if (ASCEND) {
        thrust::sort_by_key(thrust::cuda::par.on(stream), valOut + i * K,
                            valOut + (i + 1) * K, idxOut + i * K,
                            thrust::less<ValType>());
      } else {
        thrust::sort_by_key(thrust::cuda::par.on(stream), valOut + i * K,
                            valOut + (i + 1) * K, idxOut + i * K,
                            thrust::greater<ValType>());
      }
    }
  }
#undef RADIX_TOPK_CALL_BITONIC_SORT
  AS_CHECK_CUDA_LAST_ERROR();
  RADIX_TOPK_KERNEL_CHECK_SYNC(stream);

  return;
}

#undef RADIX_TOPK_KERNEL_CHECK_SYNC

}  // namespace radix_topk

/* Computes workspace size in bytes.
 * When batch_size > 1, `length` is length of the longest task.
 */
template <typename ValType>
void TopKRadixGetWorkspaceSize(size_t* sizeInBytes, int batch_size,
                               int length) {
  using CompT = typename radix_topk::ComputeT<ValType>::type;
  *sizeInBytes =
      batch_size * (/* buffer for hist */
                    sizeof(int) * (1 << 12)
                    /* buffer for val */
                    + sizeof(CompT) * length * 2
                    /* buffer for K taskLen (old & new) binId and globalCount */
                    + sizeof(int) * 5)
      /* buffer for taskOffset */
      + sizeof(int) * (batch_size + 1);
  return;
}

template <typename T>
void TopKRadixKernelLauncher(T* output, int* output_indices, const T* input,
                             void* workspace, int batch_size, int length,
                             int64_t k, cudaStream_t stream) {
  if (k <= 0 || k > length) {
    throw std::runtime_error(
        "radix topk: k should be a positive integer not greater than "
        "length");
  }

  // largest, sorted descend
  radix_topk::topKRadixSelectL<int, true, 0, false>(
      input, nullptr, output, output_indices, workspace, length, batch_size,
      static_cast<int>(k), stream);
}

}  // namespace cuda
}  // namespace allspark
