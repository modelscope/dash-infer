/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.cuh
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_BF16
#include "hie_bfloat16.hpp"
#include "hie_bfloat16_cmath.hpp"
#endif

/**
 * @brief If enabled, the NaN filter will quietly parse potential NaNs in
 * input arrays without throwing kernel launch error. If disabled, kernels
 * may throw error if the number of NaN is larger than N - K. \par
 *
 * When the number of NaN and infinity is less than or equal to N - K, the
 * output is well defined and guaranteed to be correct.
 *
 * Otherwise, the output is illy defined and may contain NaN and/or infinity.
 * When K <= 4096, the numbers that are not NaN or infinity are guaranteed to
 * be in front of NaN and/or infinity in the sorted output.
 *
 * But when K > 4096, the order of the output is unguaranteed. In such cases,
 * if requiring the numbers that are not NaN or infinity to appear in the front
 * of the output, please enable CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096.
 */
#define CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER

#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
/**
 * @brief If enabled, when K > 4096, the numbers that are not NaN or infinity
 * will be enforced to appear in the front of the output.
 *
 * @warning This will change the NaNs in the output into max/min finite values
 * of that type. Use with care!
 */
#define CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
#endif

namespace allspark {
namespace cuda {
namespace radix_topk {

constexpr int WARP_SIZE = 32;

//===================================
// packing
//===================================
template <typename T, int N>
struct alignas(sizeof(T) * N) VT {
  T data[N];

  __device__ __forceinline__ T& operator[](const int& idx) { return data[idx]; }

  __device__ __forceinline__ const T& operator[](const int& idx) const {
    return data[idx];
  }

  __device__ __forceinline__ VT<T, N> operator-(const T& s) const {
    VT<T, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = data[i] - s;
    }
    return res;
  }

  template <typename P>
  __device__ __forceinline__ explicit operator VT<P, N>() const {
    VT<P, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = static_cast<P>(data[i]);
    }
    return res;
  }
};

//===================================
// compute type
//===================================
template <typename T>
struct ComputeT {
  using type = T;
};

template <int N, typename T>
struct ComputeT<VT<T, N>> {
  using CompT = typename ComputeT<T>::type;
  using type = VT<CompT, N>;
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

//===================================
// shuffle type
//===================================
template <typename T>
struct ShuffleComputeT {
  using type = T;
};

#ifdef ENABLE_FP16
template <>
struct ShuffleComputeT<half> {
  using type = float;
};
#endif

#ifdef ENABLE_BF16
template <>
struct ShuffleComputeT<hie::bfloat16> {
  using type = float;
};
#endif

//===================================
// isnan
//===================================
template <typename T>
__device__ __forceinline__ bool isNaN(T x) {
  return isnan(x);
}

#ifdef ENABLE_FP16
template <>
__device__ __forceinline__ bool isNaN<>(half x) {
  return __hisnan(x);
}
#endif

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ bool isNaN<>(hie::bfloat16 x) {
  return std::isnan(x);
}
#endif

//===================================
// isinf
//===================================
template <typename T>
__device__ __forceinline__ bool isInf(T x) {
  return isinf(x);
}

#ifdef ENABLE_FP16
template <>
__device__ __forceinline__ bool isInf<>(half x) {
  return __hisinf(x);
}
#endif

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ bool isInf<>(hie::bfloat16 x) {
  return std::isinf(x);
}
#endif

//===================================
// limits
//===================================
template <typename T>
__device__ __forceinline__ T getMax();
template <typename T>
__device__ __forceinline__ T getMin();
template <typename T>
__device__ __forceinline__ T getPosInf();
template <typename T>
__device__ __forceinline__ T getNegInf();

static constexpr float __FP32_MIN = std::numeric_limits<float>::lowest();
static constexpr float __FP32_MAX = std::numeric_limits<float>::max();
static constexpr float __FP32_INF = std::numeric_limits<float>::infinity();

template <>
__device__ __forceinline__ float getMax<float>() {
  return __FP32_MAX;
}
template <>
__device__ __forceinline__ float getMin<float>() {
  return __FP32_MIN;
}
template <>
__device__ __forceinline__ float getPosInf<float>() {
  return __FP32_INF;
}
template <>
__device__ __forceinline__ float getNegInf<float>() {
  return (-__FP32_INF);
}

#ifdef ENABLE_FP16
inline __device__ __host__ half rawToHalf(uint16_t v) {
  __half_raw h;
  h.x = v;
  return __half(h);
}
template <>
__device__ __forceinline__ half getMax<half>() {
  return rawToHalf(0x7bff);
}
template <>
__device__ __forceinline__ half getMin<half>() {
  return rawToHalf(0xfbff);
}
template <>
__device__ __forceinline__ half getPosInf<half>() {
  return rawToHalf(0x7c00);
}
template <>
__device__ __forceinline__ half getNegInf<half>() {
  return rawToHalf(0xfc00);
}
#endif

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ hie::bfloat16 getMax<hie::bfloat16>() {
  return std::numeric_limits<hie::bfloat16>::max();
}
template <>
__device__ __forceinline__ hie::bfloat16 getMin<hie::bfloat16>() {
  return std::numeric_limits<hie::bfloat16>::lowest();
}
template <>
__device__ __forceinline__ hie::bfloat16 getPosInf<hie::bfloat16>() {
  return std::numeric_limits<hie::bfloat16>::infinity();
}
template <>
__device__ __forceinline__ hie::bfloat16 getNegInf<hie::bfloat16>() {
  return (-std::numeric_limits<hie::bfloat16>::infinity());
}
#endif

//===================================
// scaling
//===================================
template <bool WITHSCALE, typename T, typename IdxT>
__device__ __forceinline__ typename ComputeT<T>::type sampleScaler(
    const T* dataIn, const IdxT& start) {
  using CompT = typename ComputeT<T>::type;
  CompT scaler(0);
  if (WITHSCALE) {
    // TODO: use PTX LDG to improve caching
    scaler = static_cast<CompT>(dataIn[start]);
    if (isNaN(scaler) || isInf(scaler)) {
      scaler = 0;
    }
  }
  return scaler;
}

template <typename T>
__device__ __forceinline__ T getScale(const T& a, const T& s) {
  return a - s;
}

template <int N, typename T>
__device__ __forceinline__ VT<T, N> getScale(const VT<T, N>& arr, const T& s) {
  return arr - s;
}

template <bool WITHSCALE, bool LARGEST, typename T>
__device__ __forceinline__ typename ComputeT<T>::type scaling(
    const T& val_o, const typename ComputeT<T>::type& scaler) {
  using CompT = typename ComputeT<T>::type;
  CompT val = static_cast<CompT>(val_o);
#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
  //! filter NaN
  val = isNaN(val) ? (LARGEST ? getMin<CompT>() : getMax<CompT>()) : val;
#endif
  return WITHSCALE ? getScale(val, scaler) : val;
}

template <bool WITHSCALE, bool LARGEST, int N, typename T>
__device__ __forceinline__ typename ComputeT<VT<T, N>>::type scaling(
    const VT<T, N>& val_o, const typename ComputeT<T>::type& scaler) {
  using CompT = typename ComputeT<T>::type;
  using CompVec = typename ComputeT<VT<T, N>>::type;
  CompVec val = static_cast<CompVec>(val_o);
#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
//! filter NaN
#pragma unroll
  for (int i = 0; i < N; ++i) {
    val[i] =
        isNaN(val[i]) ? (LARGEST ? getMin<CompT>() : getMax<CompT>()) : val[i];
  }
#endif
  return WITHSCALE ? getScale(val, scaler) : val;
}

template <bool WITHSCALE, bool LARGEST, typename T>
__device__ __forceinline__ typename ComputeT<T>::type loadScaling(
    const T* dataIn, int index, const typename ComputeT<T>::type& scaler) {
  // TODO: use PTX LDG to improve caching
  T val_o = dataIn[index];
  return scaling<WITHSCALE, LARGEST>(val_o, scaler);
}

template <bool WITHSCALE, bool LARGEST, int N, typename T>
__device__ __forceinline__ typename ComputeT<VT<T, N>>::type loadScaling(
    const VT<T, N>* readPtr, int index,
    const typename ComputeT<T>::type& scaler) {
  // TODO: use PTX LDG to improve caching
  VT<T, N> val_o = readPtr[index];
  return scaling<WITHSCALE, LARGEST>(val_o, scaler);
}

//===================================
// get radix bin ID
//===================================
template <int LEFT = 0, int RIGHT = 0>
__device__ __forceinline__ int getBinId(const float& a) {
  const uint32_t& u_a = reinterpret_cast<const uint32_t&>(a);
  uint32_t mask = ((~(u_a >> 31)) + 1) | 0x80000000;
  return static_cast<int>(((u_a ^ mask) << LEFT) >> RIGHT);
}

//===================================
// prefix sum
//===================================
template <bool LARGEST, typename T>
__device__ __forceinline__ void warpPrefixSum(const int& lane, T* data) {
  if (LARGEST) {
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
      T new_data = __shfl_down_sync(0xffffffff, *data, i);
      if (lane < WARP_SIZE - i) {
        *data += new_data;
      }
    }
  } else {
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
      T new_data = __shfl_up_sync(0xffffffff, *data, i);
      if (lane >= i) {
        *data += new_data;
      }
    }
  }
}

}  // namespace radix_topk
}  // namespace cuda
}  // namespace allspark
