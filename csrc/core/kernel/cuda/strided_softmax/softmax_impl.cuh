/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_impl.cuh
 */

#pragma once

#include <algorithm>

#include "intrinsics.cuh"
#include "reduce.cuh"
#include "utility/check_cuda.h"

namespace allspark {
namespace cuda {
namespace strided_softmax {

template <typename T>
struct ComputeT {
  using type = T;
};

#ifdef ENABLE_FP16
template <>
struct ComputeT<half> {
  using type = float;
};
#endif

#ifdef ENABLE_BF16
template <>
struct ComputeT<hie::bfloat16> {
  using type = float;
};
#endif

// ------------------------------------
// Functors
// ------------------------------------
namespace functor {

template <typename MathT>
struct Log {
  __device__ __forceinline__ MathT operator()(const MathT x) const {
    return logf(x);
  }
};

template <typename MathT>
struct Reciprocal {
  __device__ __forceinline__ MathT operator()(const MathT x) const {
    // TODO: use intrinsic if needed
    return MathT(1.f) / x;
  }
};

template <typename MathT>
struct MinusMaxExp {
  MinusMaxExp(const MathT* maxValPtr, const MathT* temperaturePtr)
      : _maxValPtr(maxValPtr),
        _temperaturePtr(temperaturePtr),
        _temperatureRcp(0),
        _scaledMaxVal(0),
        _init(false) {
    /*
     * _temperatureRcp and _scaledMaxVal cannot be initialized in
     * constructor, because the two array pointers are both on device
     * and can only be accessed inside the kernel.
     */
  }

  __device__ __forceinline__ MathT operator()(const MathT x) {
    if (!_init) {
      // 1 / T
      _temperatureRcp = _temperaturePtr != nullptr
                            ? Reciprocal<MathT>()(_temperaturePtr[blockIdx.y])
                            : 1.0;
      // C / T
      _scaledMaxVal = _maxValPtr[blockIdx.y] * _temperatureRcp;
      _init = true;
    }
    // exp(x_i / T - C / T)
    return Exp<MathT>(x * _temperatureRcp - _scaledMaxVal);
  }

  const MathT* _maxValPtr;
  const MathT* _temperaturePtr;
  MathT _temperatureRcp;
  MathT _scaledMaxVal;
  bool _init;
};

}  // namespace functor

// ------------------------------------
// Kernel
// ------------------------------------
/**
 * @brief In-place supported, i.e. y can equal to x.
 */
template <int BLOCK, int UNROLL, typename ST, typename DT>
__global__ void __launch_bounds__(512)
    expMulRcpKernel(const ST* x, const typename ComputeT<ST>::type* maxValPtr,
                    const typename ComputeT<ST>::type* rcpSumExpPtr, DT* y,
                    const int* taskLenPtr, const float* temperatures,
                    const uint32_t stride) {
  using CompT = typename ComputeT<ST>::type;

  const uint32_t taskId = blockIdx.y;
  const float temperatureRcp =
      temperatures != nullptr
          ? functor::Reciprocal<float>()(temperatures[taskId])
          : 1.0;
  const uint32_t taskLen = taskLenPtr != nullptr ? taskLenPtr[taskId] : stride;
  const uint32_t offset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
  const uint32_t xCount =
      offset0 < taskLen ? UIntDivUp<uint32_t>(taskLen - offset0, BLOCK) : 0;
  const ST* xPtr = x + taskId * stride + offset0;
  DT* yPtr = y + taskId * stride + offset0;

  // step 1: load data
  const CompT maxVal = maxValPtr[taskId];
  const CompT sumRcp = rcpSumExpPtr[taskId];

  CompT xRegs[UNROLL];
  loadRegs<UNROLL>(xRegs, xCount,
                   [xPtr](const int i) { return xPtr[i * BLOCK]; });

  // step 2: compute
  // C / T
  const CompT scaledMaxVal = temperatureRcp * maxVal;
  CompT res[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    // x_i / T - C / T
    CompT adjusted =
        temperatureRcp * static_cast<CompT>(xRegs[i]) - scaledMaxVal;
    // exp(x_i / T - C / T) / Z
    res[i] = Exp(adjusted) * sumRcp;
  }

  // step 3: write back
  storeRegs<UNROLL>(yPtr, res, xCount, [](const int i) { return i * BLOCK; });
  return;
}

/**
 * @brief In-place supported, i.e. y can equal to x.
 */
template <int BLOCK, int UNROLL, typename ST, typename DT>
__global__ void __launch_bounds__(512)
    subLogKernel(const ST* x, const typename ComputeT<ST>::type* maxValPtr,
                 const typename ComputeT<ST>::type* logSumExpPtr, DT* y,
                 const int* taskLenPtr, const float* temperatures,
                 const uint32_t stride) {
  using CompT = typename ComputeT<ST>::type;

  const uint32_t taskId = blockIdx.y;
  const float temperatureRcp =
      temperatures != nullptr
          ? functor::Reciprocal<float>()(temperatures[taskId])
          : 1.0;
  const uint32_t taskLen = taskLenPtr != nullptr ? taskLenPtr[taskId] : stride;
  const uint32_t offset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
  const uint32_t xCount =
      offset0 < taskLen ? UIntDivUp<uint32_t>(taskLen - offset0, BLOCK) : 0;
  const ST* xPtr = x + taskId * stride + offset0;
  DT* yPtr = y + taskId * stride + offset0;

  // step 1: load data
  const CompT maxVal = maxValPtr[taskId];
  const CompT sumLog = logSumExpPtr[taskId];

  CompT xRegs[UNROLL];
  loadRegs<UNROLL>(xRegs, xCount,
                   [xPtr](const int i) { return xPtr[i * BLOCK]; });

  // step 2: compute
  // C / T
  const CompT scaledMaxVal = temperatureRcp * maxVal;
  CompT res[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    // x_i / T - C / T
    CompT adjusted =
        temperatureRcp * static_cast<CompT>(xRegs[i]) - scaledMaxVal;
    // (x_i / T - C / T) - log(Z)
    res[i] = adjusted - sumLog;
  }

  // step 3: write back
  storeRegs<UNROLL>(yPtr, res, xCount, [](const int i) { return i * BLOCK; });
  return;
}

// ------------------------------------
// Entry
// ------------------------------------
template <int BLOCK_1, int BLOCK_2, typename T, typename OutT>
void launchReduceMax(const T* x, OutT* y, T* blockRes, const int* taskLenPtr,
                     const int stride, const int taskNum, cudaStream_t stream) {
  static_assert(sizeof(T) <= 16, "sizeof(T) too large");
  constexpr int UNROLL = 16 / sizeof(T);

  uint32_t nBlocks = UIntDivUp<uint32_t>(stride, BLOCK_1 * UNROLL);
  reduce::reduceKernel<BLOCK_1, UNROLL, true,
                       allspark::cuda::strided_softmax::reduce::functor::Max>
      <<<dim3(nBlocks, taskNum), BLOCK_1, 0, stream>>>(
          x, static_cast<T*>(nullptr), blockRes, taskLenPtr, stride);
  AS_CHECK_CUDA_LAST_ERROR();

  reduce::reduceKernel<BLOCK_2, UNROLL, false,
                       allspark::cuda::strided_softmax::reduce::functor::Max>
      <<<dim3(1, taskNum), BLOCK_2, 0, stream>>>(
          blockRes, y, static_cast<T*>(nullptr), static_cast<int*>(nullptr),
          nBlocks);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int BLOCK_1, int BLOCK_2, typename T>
void launchRcpSumExp(const T* x, const typename ComputeT<T>::type* maxVal,
                     typename ComputeT<T>::type* y,
                     typename ComputeT<T>::type* blockRes,
                     const int* taskLenPtr, const float* temperatures,
                     const int stride, const int taskNum, cudaStream_t stream) {
  using CompT = typename ComputeT<T>::type;
  static_assert(sizeof(T) <= 16, "sizeof(T) too large");
  constexpr int UNROLL = 16 / sizeof(T);

  uint32_t nBlocks = UIntDivUp<uint32_t>(stride, BLOCK_1 * UNROLL);
  reduce::reduceKernel<BLOCK_1, UNROLL, true,
                       allspark::cuda::strided_softmax::reduce::functor::Sum>
      <<<dim3(nBlocks, taskNum), BLOCK_1, 0, stream>>>(
          x, static_cast<T*>(nullptr), blockRes, taskLenPtr, stride,
          allspark::cuda::strided_softmax::functor::MinusMaxExp<CompT>(
              maxVal, temperatures));
  AS_CHECK_CUDA_LAST_ERROR();

  reduce::reduceKernel<
      BLOCK_2, UNROLL, false,
      allspark::cuda::strided_softmax::reduce::functor::Sum,
      allspark::cuda::strided_softmax::reduce::transform::Default,
      allspark::cuda::strided_softmax::functor::Reciprocal>
      <<<dim3(1, taskNum), BLOCK_2, 0, stream>>>(
          blockRes, y, static_cast<CompT*>(nullptr), static_cast<int*>(nullptr),
          nBlocks);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int BLOCK_1, int BLOCK_2, typename T>
void launchLogSumExp(const T* x, const typename ComputeT<T>::type* maxVal,
                     typename ComputeT<T>::type* y,
                     typename ComputeT<T>::type* blockRes,
                     const int* taskLenPtr, const float* temperatures,
                     const int stride, const int taskNum, cudaStream_t stream) {
  using CompT = typename ComputeT<T>::type;
  static_assert(sizeof(T) <= 16, "sizeof(T) too large");
  constexpr int UNROLL = 16 / sizeof(T);

  uint32_t nBlocks = UIntDivUp<uint32_t>(stride, BLOCK_1 * UNROLL);
  reduce::reduceKernel<BLOCK_1, UNROLL, true,
                       allspark::cuda::strided_softmax::reduce::functor::Sum>
      <<<dim3(nBlocks, taskNum), BLOCK_1, 0, stream>>>(
          x, static_cast<T*>(nullptr), blockRes, taskLenPtr, stride,
          allspark::cuda::strided_softmax::functor::MinusMaxExp<CompT>(
              maxVal, temperatures));
  AS_CHECK_CUDA_LAST_ERROR();

  reduce::reduceKernel<
      BLOCK_2, UNROLL, false,
      allspark::cuda::strided_softmax::reduce::functor::Sum,
      allspark::cuda::strided_softmax::reduce::transform::Default,
      allspark::cuda::strided_softmax::functor::Log>
      <<<dim3(1, taskNum), BLOCK_2, 0, stream>>>(
          blockRes, y, static_cast<CompT*>(nullptr), static_cast<int*>(nullptr),
          nBlocks);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int BLOCK, typename T>
void launchExpMulRcp(const T* x, const typename ComputeT<T>::type* maxVal,
                     const typename ComputeT<T>::type* rcpSumExp, T* y,
                     const int* taskLenPtr, const float* temperatures,
                     const int stride, const int taskNum, cudaStream_t stream) {
  using CompT = typename ComputeT<T>::type;
  static_assert(sizeof(T) <= 16, "sizeof(T) too large");
  constexpr int UNROLL = 16 / sizeof(T);

  uint32_t nBlocks = UIntDivUp<uint32_t>(stride, BLOCK * UNROLL);
  expMulRcpKernel<BLOCK, UNROLL><<<dim3(nBlocks, taskNum), BLOCK, 0, stream>>>(
      x, maxVal, rcpSumExp, y, taskLenPtr, temperatures, stride);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int BLOCK, typename T>
void launchSubLog(const T* x, const typename ComputeT<T>::type* maxVal,
                  const typename ComputeT<T>::type* logSumExp, T* y,
                  const int* taskLenPtr, const float* temperatures,
                  const int stride, const int taskNum, cudaStream_t stream) {
  using CompT = typename ComputeT<T>::type;
  static_assert(sizeof(T) <= 16, "sizeof(T) too large");
  constexpr int UNROLL = 16 / sizeof(T);

  uint32_t nBlocks = UIntDivUp<uint32_t>(stride, BLOCK * UNROLL);
  subLogKernel<BLOCK, UNROLL><<<dim3(nBlocks, taskNum), BLOCK, 0, stream>>>(
      x, maxVal, logSumExp, y, taskLenPtr, temperatures, stride);
  AS_CHECK_CUDA_LAST_ERROR();
  return;
}

template <int BLOCK, int UNROLL, typename T, typename CompT>
void getSoftmaxWorkspaceSize(size_t* wsInBytes, const int taskNum,
                             const int length) {
  size_t reduceMaxWsBytes(0);
  reduce::getReduceWorkspaceSize<BLOCK, UNROLL, T>(&reduceMaxWsBytes, taskNum,
                                                   length);

  uint32_t nBlocks = UIntDivUp<uint32_t>(length, BLOCK * UNROLL);
  size_t softmaxWsBytes = sizeof(CompT) * taskNum * nBlocks;
  *wsInBytes = std::max(reduceMaxWsBytes, softmaxWsBytes)
               /* max val, reciprocal of sum of exp */
               + 2 * sizeof(CompT) * taskNum;
  return;
}

}  // namespace strided_softmax
}  // namespace cuda
}  // namespace allspark
