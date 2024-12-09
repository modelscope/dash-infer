/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp_impl.cuh
 */

#pragma once

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cub/cub.cuh>
#include <vector>

#include "filter.cuh"
#include "utility/check_cuda.h"
#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace topp {

namespace impl {

template <int BLOCK, int UNROLL, typename IdxT>
__global__ void __launch_bounds__(1024)
    genIndicesKernel(IdxT* output, int* dOffsets, const uint32_t taskLen) {
  const uint32_t taskId = blockIdx.y;
  const uint32_t taskNum = gridDim.y;
  const uint32_t offset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
  const uint32_t xCount =
      offset0 < taskLen ? UIntDivUp<uint32_t>(taskLen - offset0, BLOCK) : 0;
  IdxT* yPtr = output + taskId * taskLen + offset0;

  if (threadIdx.x == 0) {
    dOffsets[taskId] = taskId * taskLen;
    if (taskId == taskNum - 1) {
      dOffsets[taskNum] = taskNum * taskLen;
    }
  }

  uint32_t res[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    res[i] = offset0 + i * BLOCK;
  }

  storeRegs<UNROLL>(yPtr, res, xCount, [](const int i) { return i * BLOCK; });
  return;
}

template <typename T>
struct GetHiednnDatatype {
  static constexpr hiednnDataType_t type = HIEDNN_DATATYPE_FP32;
};

#ifdef ENABLE_FP16
template <>
struct GetHiednnDatatype<half> {
  static constexpr hiednnDataType_t type = HIEDNN_DATATYPE_FP16;
};
#endif

#ifdef ENABLE_BF16
template <>
struct GetHiednnDatatype<bfloat16> {
  static constexpr hiednnDataType_t type = HIEDNN_DATATYPE_BF16;
};
#endif

}  // namespace impl

// ------------------------------------
// Generate indices
// ------------------------------------
template <typename IdxT>
void LaunchGenIndices(IdxT* output, int* dOffsets, const int taskLen,
                      const int taskNum, cudaStream_t stream) {
  constexpr int BLOCK = 1024;
  static_assert(sizeof(IdxT) <= 16, "sizeof(IdxT) too large");
  constexpr int UNROLL = 16 / sizeof(IdxT);

  uint32_t nBlocks = UIntDivUp<uint32_t>(taskLen, BLOCK * UNROLL);
  impl::genIndicesKernel<BLOCK, UNROLL>
      <<<dim3(nBlocks, taskNum), BLOCK, 0, stream>>>(output, dOffsets, taskLen);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

// ------------------------------------
// Sorting
// ------------------------------------
template <bool ASCEND, typename ValT, typename IdxT>
void GetCubSortWorkspaceSize(size_t* wsInBytes, int taskNum, int length) {
  // buffer for task offset
  size_t bufInBytes = sizeof(int) * (taskNum + 1);

  // CUB ws size: no double buffer version
  const int itemNum = length * taskNum;
  size_t cubWsInBytes = 0;
  if (ASCEND) {
    AS_CHECK_CUDA((cub::DeviceSegmentedRadixSort::SortPairs<ValT, IdxT>(
        nullptr, cubWsInBytes, nullptr, nullptr, nullptr, nullptr, itemNum,
        taskNum, static_cast<int*>(nullptr), static_cast<int*>(nullptr))));
  } else {
    AS_CHECK_CUDA(
        (cub::DeviceSegmentedRadixSort::SortPairsDescending<ValT, IdxT>(
            nullptr, cubWsInBytes, nullptr, nullptr, nullptr, nullptr, itemNum,
            taskNum, static_cast<int*>(nullptr), static_cast<int*>(nullptr))));
  }

  bufInBytes = AlignUpPow2<MAX_ALIGNMENT>(bufInBytes);
  bufInBytes += cubWsInBytes;
  *wsInBytes = bufInBytes;
  return;
}

template <bool ASCEND, typename ValT, typename IdxT>
void CubSortLogits(ValT* valOut, IdxT* idxOut, const ValT* valIn,
                   const IdxT* idxIn, void* workspace, size_t wsSizeInBytes,
                   const int taskNum, const int length, cudaStream_t stream) {
  if (workspace == nullptr) {
    throw std::runtime_error("CubSortLogits: workspace cannot be nullptr");
  }

  size_t minWsSize(0);
  GetCubSortWorkspaceSize<ASCEND, ValT, IdxT>(&minWsSize, taskNum, length);
  if (wsSizeInBytes < minWsSize) {
    throw std::runtime_error("CubSortLogits: insufficient workspace size");
  }

  int* dOffsets = static_cast<int*>(workspace);
  size_t usedBytes = sizeof(int) * (taskNum + 1);
  usedBytes = AlignUpPow2<MAX_ALIGNMENT>(usedBytes);
  void* cubWsPtr =
      static_cast<void*>(static_cast<char*>(workspace) + usedBytes);
  size_t vWsSizeInBytes = wsSizeInBytes - usedBytes;

  const int itemNum = length * taskNum;
  if (ASCEND) {
    // Run sorting operation
    AS_CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
        cubWsPtr, vWsSizeInBytes, valIn, valOut, idxIn, idxOut, itemNum,
        taskNum, dOffsets, dOffsets + 1, 0, sizeof(ValT) * 8, stream));
  } else {
    AS_CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cubWsPtr, vWsSizeInBytes, valIn, valOut, idxIn, idxOut, itemNum,
        taskNum, dOffsets, dOffsets + 1, 0, sizeof(ValT) * 8, stream));
  }
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

// ------------------------------------
// Prefix sum
// ------------------------------------
template <typename T>
void HiednnPrefixSum(T* output, const T* input, const int taskLen,
                     const int taskNum, hiednnCudaHandle_t handle,
                     cudaStream_t stream) {
  constexpr hiednnDataType_t dataType = impl::GetHiednnDatatype<T>::type;

  const int64_t dims[2]{taskNum, taskLen};
  hiednnTensorDesc_t dataDesc;
  AS_CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));
  AS_CHECK_HIEDNN(hiednnSetNormalTensorDesc(dataDesc, dataType, 2, dims));

  constexpr int AXIS = 1;
  constexpr int EXCLUSIVE = 0;
  constexpr int REVERSE = 0;
  AS_CHECK_HIEDNN(hiednnCudaPrefixSum(handle, dataDesc, input, AXIS, EXCLUSIVE,
                                      REVERSE, dataDesc, output));
  AS_CHECK_HIEDNN(hiednnDestroyTensorDesc(dataDesc));
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

}  // namespace topp
}  // namespace cuda
}  // namespace allspark
