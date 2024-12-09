/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce.cuh
 */

#pragma once
#include <limits>
enum class ReduceOp : int {
  kSUM = 0,   //!< sum of the elements
  kPROD = 1,  //!< product of the elements
  kMAX = 2,   //!< maximum of the elements
  kMIN = 3,   //!< minimum of the elements
};

template <typename T, ReduceOp op>
__inline__ __device__ T computeFillValue() {
  T result = 0.0f;
  switch (op) {
    case ReduceOp::kPROD:
      result = 1.0f;
      break;
    case ReduceOp::kMAX:
      result = std::numeric_limits<int>::min();
      break;
    case ReduceOp::kMIN:
      result = std::numeric_limits<int>::max();
      break;
  }
  return static_cast<T>(result);
}

template <typename T, ReduceOp op>
__inline__ __device__ void reduceFunction(T* val, T acc) {
  switch (op) {
    case ReduceOp::kSUM:
      *val += acc;
      break;
    case ReduceOp::kPROD:
      *val *= acc;
      break;
    case ReduceOp::kMAX:
      *val = *val > acc ? *val : acc;
      break;
    case ReduceOp::kMIN:
      *val = *val < acc ? *val : acc;
      break;
  }
}

#define FINAL_MASK 0xffffffff
template <typename T, ReduceOp op>
__inline__ __device__ void warpReduce(T* val) {
  reduceFunction<T, op>(val, __shfl_down_sync(FINAL_MASK, *val, 16, 32));
  reduceFunction<T, op>(val, __shfl_down_sync(FINAL_MASK, *val, 8, 32));
  reduceFunction<T, op>(val, __shfl_down_sync(FINAL_MASK, *val, 4, 32));
  reduceFunction<T, op>(val, __shfl_down_sync(FINAL_MASK, *val, 2, 32));
  reduceFunction<T, op>(val, __shfl_down_sync(FINAL_MASK, *val, 1, 32));
}

template <typename T, ReduceOp op>
__forceinline__ __device__ void blockReduce(T* val) {
  static __shared__ T shared[32];
  int lane_id = threadIdx.x & 0x1f;
  int warp_id = threadIdx.x >> 5;
  warpReduce<T, op>(val);
  if (lane_id == 0) shared[warp_id] = *val;
  __syncthreads();
  *val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane_id]
                                           : computeFillValue<T, op>();
  warpReduce<T, op>(val);
}
#undef FINAL_MASK