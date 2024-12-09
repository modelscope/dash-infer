/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bitonic_sort.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace radix_topk {
namespace bitonic {
namespace impl {

template <bool ASCEND, typename T>
__device__ __forceinline__ T convertNaN(const T& val) {
  T out = val;
#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
  if (isNaN(out)) {
    out = ASCEND ? getMax<T>() : getMin<T>();
  }
#endif
  return out;
}

//===================================
// swap
//===================================
template <typename T>
__device__ __forceinline__ void swap(T* cache, const int& oldPos,
                                     const int& newPos) {
  T tmp = cache[oldPos];
  cache[oldPos] = cache[newPos];
  cache[newPos] = tmp;
}

template <bool ASCEND, int SEGMENT, typename ValType, typename IdxType>
__device__ __forceinline__ void cas(ValType* valCache, IdxType* idxCache,
                                    const int& tid, const bool& ascend) {
  constexpr int stride = SEGMENT / 2;
  const bool high = (tid / stride) & 1;
  const int dst = high ? tid - stride : tid + stride;

  using ShuffleT = typename ShuffleComputeT<ValType>::type;

  ShuffleT myVal = static_cast<ShuffleT>(valCache[tid]);
  IdxType myIdx = idxCache[tid];

  ShuffleT dstVal;
  IdxType dstIdx;
  if (SEGMENT > WARP_SIZE) {
    dstVal = static_cast<ShuffleT>(valCache[dst]);
    dstIdx = idxCache[dst];
  } else {
    dstVal = __shfl_sync(0xffffffff, myVal, dst & (WARP_SIZE - 1));
    dstIdx = __shfl_sync(0xffffffff, myIdx, dst & (WARP_SIZE - 1));
  }
  __syncthreads();

  ShuffleT myVal_s = convertNaN<ASCEND>(myVal);
  ShuffleT dstVal_s = convertNaN<ASCEND>(dstVal);

  bool swap = ascend ? (high ? (dstVal_s > myVal_s) : (dstVal_s < myVal_s))
                     : (high ? (dstVal_s < myVal_s) : (dstVal_s > myVal_s));
  if (swap) {
    valCache[tid] = static_cast<ValType>(dstVal);
    idxCache[tid] = dstIdx;
  } else {
    valCache[tid] = static_cast<ValType>(myVal);
    idxCache[tid] = myIdx;
  }
  __syncthreads();
}

//===================================
// sorting networks
//===================================

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_2(ValType* valCache, IdxType* idxCache,
                                       const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 2) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 2>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_4(ValType* valCache, IdxType* idxCache,
                                       const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 4) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 4>(valCache, idxCache, tid, segmentAscend);
  sort_2<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_8(ValType* valCache, IdxType* idxCache,
                                       const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 8) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 8>(valCache, idxCache, tid, segmentAscend);
  sort_4<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_16(ValType* valCache, IdxType* idxCache,
                                        const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 16) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 16>(valCache, idxCache, tid, segmentAscend);
  sort_8<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_32(ValType* valCache, IdxType* idxCache,
                                        const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 32) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 32>(valCache, idxCache, tid, segmentAscend);
  sort_16<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_64(ValType* valCache, IdxType* idxCache,
                                        const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 64) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 64>(valCache, idxCache, tid, segmentAscend);
  sort_32<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_128(ValType* valCache, IdxType* idxCache,
                                         const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 128) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 128>(valCache, idxCache, tid, segmentAscend);
  sort_64<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_256(ValType* valCache, IdxType* idxCache,
                                         const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 256) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 256>(valCache, idxCache, tid, segmentAscend);
  sort_128<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_512(ValType* valCache, IdxType* idxCache,
                                         const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 512) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 512>(valCache, idxCache, tid, segmentAscend);
  sort_256<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

template <bool ASCEND, bool WITHFLAG, typename ValType, typename IdxType>
__device__ __forceinline__ void sort_1024(ValType* valCache, IdxType* idxCache,
                                          const int& tid, bool segmentAscend) {
  if (!WITHFLAG) {
    const bool segment_is_odd = static_cast<bool>((tid / 1024) & 1);
    segmentAscend = ASCEND ? !segment_is_odd : segment_is_odd;
  }
  cas<ASCEND, 1024>(valCache, idxCache, tid, segmentAscend);
  sort_512<ASCEND, 1>(valCache, idxCache, tid, segmentAscend);
}

}  // namespace impl

//===================================
// bitonic sort interfaces
//===================================
template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(128)
    bitonicSort_128(ValType* valIn, IdxType* idxIn, const int maxLen,
                    const int* taskOffsetPtr) {
  __shared__ ValType valCache[128];
  __shared__ IdxType idxCache[128];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

  if (tid < validLen) {
    valCache[tid] = valIn[offset + tid];
    idxCache[tid] = idxIn[offset + tid];
  } else {
    valCache[tid] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid] = 0;
  }
  __syncthreads();

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid, 0);

  if (tid < validLen) {
    valIn[offset + tid] = valCache[tid];
    idxIn[offset + tid] = idxCache[tid];
  }
}

template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(256)
    bitonicSort_256(ValType* valIn, IdxType* idxIn, const int maxLen,
                    const int* taskOffsetPtr) {
  __shared__ ValType valCache[256];
  __shared__ IdxType idxCache[256];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

  if (tid < validLen) {
    valCache[tid] = valIn[offset + tid];
    idxCache[tid] = idxIn[offset + tid];
  } else {
    valCache[tid] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid] = 0;
  }
  __syncthreads();

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_256<ASCEND, 0>(valCache, idxCache, tid, 0);

  if (tid < validLen) {
    valIn[offset + tid] = valCache[tid];
    idxIn[offset + tid] = idxCache[tid];
  }
}

template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(512)
    bitonicSort_512(ValType* valIn, IdxType* idxIn, const int maxLen,
                    const int* taskOffsetPtr) {
  __shared__ ValType valCache[512];
  __shared__ IdxType idxCache[512];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

  if (tid < validLen) {
    valCache[tid] = valIn[offset + tid];
    idxCache[tid] = idxIn[offset + tid];
  } else {
    valCache[tid] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid] = 0;
  }
  __syncthreads();

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_256<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_512<ASCEND, 0>(valCache, idxCache, tid, 0);

  if (tid < validLen) {
    valIn[offset + tid] = valCache[tid];
    idxIn[offset + tid] = idxCache[tid];
  }
}

template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(1024)
    bitonicSort_1024(ValType* valIn, IdxType* idxIn, const int maxLen,
                     const int* taskOffsetPtr) {
  __shared__ ValType valCache[1024];
  __shared__ IdxType idxCache[1024];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

  if (tid < validLen) {
    valCache[tid] = valIn[offset + tid];
    idxCache[tid] = idxIn[offset + tid];
  } else {
    valCache[tid] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid] = 0;
  }
  __syncthreads();

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_256<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_512<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_1024<ASCEND, 0>(valCache, idxCache, tid, 0);

  if (tid < validLen) {
    valIn[offset + tid] = valCache[tid];
    idxIn[offset + tid] = idxCache[tid];
  }
}

template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(1024)
    bitonicSort_2048(ValType* valIn, IdxType* idxIn, const int maxLen,
                     const int* taskOffsetPtr) {
  __shared__ ValType valCache[2048];
  __shared__ IdxType idxCache[2048];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

  // first half
  if (tid < validLen) {
    valCache[tid] = valIn[offset + tid];
    idxCache[tid] = idxIn[offset + tid];
  } else {
    valCache[tid] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid] = 0;
  }

  // second half
  if (tid + 1024 < validLen) {
    valCache[tid + 1024] = valIn[offset + tid + 1024];
    idxCache[tid + 1024] = idxIn[offset + tid + 1024];
  } else {
    valCache[tid + 1024] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
    idxCache[tid + 1024] = 0;
  }
  __syncthreads();

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_256<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_512<ASCEND, 0>(valCache, idxCache, tid, 0);
  impl::sort_1024<ASCEND, 0>(valCache, idxCache, tid, 0);

  impl::sort_2<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_4<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_8<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_16<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_32<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_64<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_128<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_256<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_512<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);
  impl::sort_1024<ASCEND, 0>(valCache, idxCache, tid + 1024, 0);

  ValType lowVal = valCache[tid];
  ValType highVal = valCache[tid + 1024];
  ValType lowVal_s = impl::convertNaN<ASCEND>(lowVal);
  ValType highVal_s = impl::convertNaN<ASCEND>(highVal);

  if (ASCEND ? (lowVal_s > highVal_s) : (lowVal_s < highVal_s)) {
    impl::swap(valCache, tid, tid + 1024);
    impl::swap(idxCache, tid, tid + 1024);
  }
  __syncthreads();
  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid, ASCEND);
  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid + 1024, ASCEND);

  // first half
  if (tid < validLen) {
    valIn[offset + tid] = valCache[tid];
    idxIn[offset + tid] = idxCache[tid];
  }

  // second half
  if (tid + 1024 < validLen) {
    valIn[offset + tid + 1024] = valCache[tid + 1024];
    idxIn[offset + tid + 1024] = idxCache[tid + 1024];
  }
}

template <bool ASCEND, typename ValType, typename IdxType>
__global__ void __launch_bounds__(1024)
    bitonicSort_4096(ValType* valIn, IdxType* idxIn, const int maxLen,
                     const int* taskOffsetPtr) {
  __shared__ ValType valCache[4096];
  __shared__ IdxType idxCache[4096];

  const uint32_t taskId = blockIdx.x;
  const int taskLen = taskOffsetPtr
                          ? taskOffsetPtr[taskId + 1] - taskOffsetPtr[taskId]
                          : maxLen;

  const uint32_t tid = threadIdx.x;
  const uint32_t offset = taskId * maxLen;

  const int validLen = taskLen < maxLen ? taskLen : maxLen;

#pragma unroll
  for (int i = 0, pos = tid; i < 4; ++i, pos += 1024) {
    if (pos < validLen) {
      valCache[pos] = valIn[offset + pos];
      idxCache[pos] = idxIn[offset + pos];
    } else {
      valCache[pos] = ASCEND ? getPosInf<ValType>() : getNegInf<ValType>();
      idxCache[pos] = 0;
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0, pos = tid; i < 4; ++i, pos += 1024) {
    impl::sort_2<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_4<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_8<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_16<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_32<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_64<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_128<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_256<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_512<ASCEND, 0>(valCache, idxCache, pos, 0);
    impl::sort_1024<ASCEND, 0>(valCache, idxCache, pos, 0);
  }

  ValType lowVal = valCache[tid];
  ValType highVal = valCache[tid + 1024];
  ValType lowVal_s = impl::convertNaN<ASCEND>(lowVal);
  ValType highVal_s = impl::convertNaN<ASCEND>(highVal);

  if (ASCEND ? (lowVal_s > highVal_s) : (lowVal_s < highVal_s)) {
    impl::swap(valCache, tid, tid + 1024);
    impl::swap(idxCache, tid, tid + 1024);
  }
  __syncthreads();

  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid, ASCEND);
  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid + 1024, ASCEND);

  lowVal = valCache[tid + 2048];
  highVal = valCache[tid + 3072];
  lowVal_s = impl::convertNaN<ASCEND>(lowVal);
  highVal_s = impl::convertNaN<ASCEND>(highVal);

  if ((!ASCEND) ? (lowVal_s > highVal_s) : (lowVal_s < highVal_s)) {
    impl::swap(valCache, tid + 2048, tid + 3072);
    impl::swap(idxCache, tid + 2048, tid + 3072);
  }
  __syncthreads();

  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid + 2048, !ASCEND);
  impl::sort_1024<ASCEND, 1>(valCache, idxCache, tid + 3072, !ASCEND);

#pragma unroll
  for (int i = 0, pos = tid; i < 2; ++i, pos += 1024) {
    lowVal = valCache[pos];
    highVal = valCache[pos + 2048];
    lowVal_s = impl::convertNaN<ASCEND>(lowVal);
    highVal_s = impl::convertNaN<ASCEND>(highVal);

    if (ASCEND ? (lowVal_s > highVal_s) : (lowVal_s < highVal_s)) {
      impl::swap(valCache, pos, pos + 2048);
      impl::swap(idxCache, pos, pos + 2048);
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0, pos = tid; i < 2; ++i, pos += 2048) {
    lowVal = valCache[pos];
    highVal = valCache[pos + 1024];
    lowVal_s = impl::convertNaN<ASCEND>(lowVal);
    highVal_s = impl::convertNaN<ASCEND>(highVal);

    if (ASCEND ? (lowVal_s > highVal_s) : (lowVal_s < highVal_s)) {
      impl::swap(valCache, pos, pos + 1024);
      impl::swap(idxCache, pos, pos + 1024);
    }
    __syncthreads();

    impl::sort_1024<ASCEND, 1>(valCache, idxCache, pos, ASCEND);
    impl::sort_1024<ASCEND, 1>(valCache, idxCache, pos + 1024, ASCEND);
  }

  int pos = tid;
  while (pos < validLen) {
    valIn[offset + pos] = valCache[pos];
    idxIn[offset + pos] = idxCache[pos];
    pos += 1024;
  }
}

}  // namespace bitonic
}  // namespace radix_topk
}  // namespace cuda
}  // namespace allspark
