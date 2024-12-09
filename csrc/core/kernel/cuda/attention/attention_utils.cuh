/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    attention_utils.cuh
 */

#ifndef __MHA_UTILS__
#define __MHA_UTILS__
#include <cuda/cuda_common.h>
#include <stdint.h>

#include <cuda/hie/cuda_intdivider.hpp>
#include <limits>

namespace allspark {
namespace cuda {
namespace attention {
using IDX = int32_t;
using LID = int64_t;
using u32div_t = hie::internal::IntDivModer<uint32_t>;
// #ifdef ENABLE_FP16
// using half = half;
// #else   // ENABLE_FP16
// using half = uint16_t;
// #endif  // ENABLE_FP16
#ifdef ENABLE_BF16
using bf16 = __hie_buildin::bfloat16;
#else   // ENABLE_BF16
using bf16 = uint16_t;
#endif  // ENABLE_BF16

namespace utils {
template <typename T>
T cal_ceil(T ind, T div) {
  return (ind % div) ? (1 + ind / div) : (ind / div);
}
template <typename T>
T cal_align(T ind, T div) {
  return div * cal_ceil<T>(ind, div);
}

constexpr int32_t warp_size = 32;
template <int32_t ALIGN, typename TYPE>
struct alignas(ALIGN * sizeof(TYPE)) packed_data {
  TYPE pack[ALIGN];
};
template <int32_t ALIGN, typename TYPE>
struct non_packed_data {
  TYPE pack[ALIGN];
};
template <typename T>
struct MaxOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x > y ? x : y;
  }
  constexpr static T init = std::numeric_limits<T>::min();
};
template <>
struct MaxOp<float> {
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x > y ? x : y;
  }
  constexpr static float init = -INFINITY;
};
template <typename T>
struct MinOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x < y ? x : y;
  }
  constexpr static T init = std::numeric_limits<T>::max();
};
template <>
struct MinOp<float> {
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x < y ? x : y;
  }
  constexpr static float init = INFINITY;
};
template <typename T>
struct SumOp {
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x + y;
  }
  constexpr static T init = 0;
};

template <template <class> class Func, typename T, int32_t nTid = warp_size>
__device__ __forceinline__ T ReduceThread(T val) {
  static_assert(nTid == 2 || nTid == 4 || nTid == 8 || nTid == 16 ||
                nTid == 32);
#pragma unroll
  for (int i = nTid; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = Func<T>::op(tmp, val);
  }
  return val;
}

template <template <class> class Func, typename T, int32_t nWarp>
__device__ __forceinline__ T ReduceBlock(T val) {
  static_assert(nWarp <= warp_size);
  __shared__ T reduce[nWarp];
  T warp_reduce = ReduceThread<Func, T, warp_size>(val);
  if (threadIdx.x % warp_size == 0)
    reduce[threadIdx.x / warp_size] = warp_reduce;
  __syncthreads();
  warp_reduce = threadIdx.x % warp_size < nWarp ? reduce[threadIdx.x % nWarp]
                                                : Func<T>::init;
  return ReduceThread<Func, T, warp_size>(warp_reduce);
}

}  // namespace utils
}  // namespace attention
}  // namespace cuda
}  // namespace allspark

#endif  // __MHA_UTILS__