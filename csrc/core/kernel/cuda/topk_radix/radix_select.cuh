/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    radix_select.cuh
 */

#pragma once

#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace radix_topk {

//===================================
// type- & scaling-unaware kernels
//===================================
template <int BLOCK, int LEFT, int RIGHT>
__global__ void __launch_bounds__(1024)
    countBin(const float* dataIn, const int* taskLenPtr, int* histPtr,
             const int stride) {
  constexpr int histLen = 1 << (8 * sizeof(float) - RIGHT);
  __shared__ int blockHist[histLen];

#pragma unroll
  for (int i = threadIdx.x; i < histLen; i += BLOCK) {
    blockHist[i] = 0;
  }
  __syncthreads();

  const int taskId = blockIdx.y;
  const int taskLen = taskLenPtr[taskId];
  const int tid = blockIdx.x * BLOCK + threadIdx.x;
  if (tid < taskLen) {
    int binId = getBinId<LEFT, RIGHT>(dataIn[taskId * stride + tid]);
    atomicAdd(&blockHist[binId], 1);
  }
  __syncthreads();

#pragma unroll
  for (int i = threadIdx.x; i < histLen; i += BLOCK) {
    if (blockHist[i] > 0) {
      atomicAdd(&histPtr[taskId * histLen + i], blockHist[i]);
    }
  }
  return;
}

template <bool LARGEST, int BLOCK, int HISTLEN>
__global__ void __launch_bounds__(1024)
    selectBin(const int* histPtr, int* binIdPtr, int* kPtr, int* taskLenPtr) {
  static_assert(HISTLEN % BLOCK == 0,
                "selectBin requires HISTLEN % BLOCK == 0");
  constexpr int UNROLL = HISTLEN / BLOCK;

  __shared__ int prefixSum[BLOCK];

  const int taskId = blockIdx.x;
  const int warpId = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;

  int oldK = kPtr[taskId];
  int count[UNROLL];
  int sum = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    count[i] = histPtr[taskId * HISTLEN + threadIdx.x * UNROLL + i];
    sum += count[i];
  }
  warpPrefixSum<LARGEST>(lane, &sum);
  prefixSum[threadIdx.x] = sum;
  __syncthreads();

  static_assert(BLOCK % 32 == 0, "selectBin requires BLOCK % 32 == 0");
  constexpr int NUM_WARPS = BLOCK / 32;

  if (LARGEST) {
#pragma unroll
    for (int i = 1; i < NUM_WARPS; i *= 2) {
      if ((warpId + i) < NUM_WARPS) {
        sum += prefixSum[(warpId + i) * 32];
      }
      __syncthreads();
      prefixSum[threadIdx.x] = sum;
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int i = 1; i < NUM_WARPS; i *= 2) {
      if ((warpId - i) >= 0) {
        sum += prefixSum[(warpId - i) * 32 + 31];
      }
      __syncthreads();
      prefixSum[threadIdx.x] = sum;
      __syncthreads();
    }
  }

  int neighbor = 0;
  if (LARGEST) {
    neighbor = threadIdx.x < BLOCK - 1 ? prefixSum[threadIdx.x + 1] : 0;
  } else {
    neighbor = threadIdx.x > 0 ? prefixSum[threadIdx.x - 1] : 0;
  }
  if (oldK > neighbor && oldK <= sum) {
    oldK -= neighbor;
    if (LARGEST) {
      for (int i = UNROLL - 1; i >= 0; --i) {
        if (count[i] >= oldK) {
          binIdPtr[taskId] = threadIdx.x * UNROLL + i;
          kPtr[taskId] = oldK;
          taskLenPtr[taskId] = count[i];
          break;
        }
        oldK -= count[i];
      }
    } else {
      for (int i = 0; i < UNROLL; ++i) {
        if (count[i] >= oldK) {
          binIdPtr[taskId] = threadIdx.x * UNROLL + i;
          kPtr[taskId] = oldK;
          taskLenPtr[taskId] = count[i];
          break;
        }
        oldK -= count[i];
      }
    }
  }
  return;
}

template <int BLOCK, int LEFT, int RIGHT>
__global__ void __launch_bounds__(256)
    selectCandidate(float* dataIn, float* dataOut, int* globalCountPtr,
                    const int* binIdPtr, const int* taskLenPtr,
                    const int stride) {
  __shared__ int blockCount[1];
  __shared__ float blockCache[BLOCK];

  const int taskId = blockIdx.y;
  const int taskLen = taskLenPtr[taskId];
  const int mask = binIdPtr[taskId];
  int idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx < taskLen && threadIdx.x == 0) {
    blockCount[0] = 0;
  }
  __syncthreads();

  if (idx < taskLen) {
    float data = dataIn[taskId * stride + idx];
    if (mask == getBinId<LEFT, RIGHT>(data)) {
      int pos = atomicAdd(blockCount, 1);
      blockCache[pos] = data;
    }
  }
  __syncthreads();

  int count = blockCount[0];
  __syncthreads();

  if (idx < taskLen && threadIdx.x == 0) {
    blockCount[0] = atomicAdd(globalCountPtr + taskId, count);
  }
  __syncthreads();

  if (idx < taskLen && threadIdx.x < count) {
    dataOut[taskId * stride + blockCount[0] + threadIdx.x] =
        blockCache[threadIdx.x];
  }
  return;
}

//===================================
// type- & scaling-aware kernels
//===================================
template <int BLOCK, int LEFT, int RIGHT, int PACKSIZE, bool WITHSCALE,
          bool LARGEST, typename T>
__global__ void __launch_bounds__(1024)
    countBinEx(const T* dataIn, const int* taskOffsetPtr, int* histPtr,
               const int taskNum) {
  using InVec = VT<T, PACKSIZE>;
  using CompT = typename ComputeT<T>::type;

  constexpr int histLen = 1 << (8 * sizeof(CompT) - RIGHT);
  __shared__ int blockHist[histLen];

  const int tid = blockIdx.x * BLOCK + threadIdx.x;
  const int stepSize = BLOCK * PACKSIZE * gridDim.x;

  // lambda to update histogram
  auto updateHist = [&](const CompT& value) {
#ifndef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
    // for potential NaN, just return
    if (isNaN(value)) {
      return;
    }
#endif
    int binId = getBinId<LEFT, RIGHT>(value);
    atomicAdd(blockHist + binId, 1);
    return;
  };

  for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
#pragma unroll
    for (int i = threadIdx.x; i < histLen; i += BLOCK) {
      blockHist[i] = 0;
    }
    __syncthreads();

    int offset = taskOffsetPtr[taskId];
    const int pad = offset & (PACKSIZE - 1);
    offset -= pad;

    // scaling factor
    const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);

    const int taskLen = taskOffsetPtr[taskId + 1] - offset;
    const int step = taskLen / stepSize;

    if (step > 0) {
      int idx = tid;
      const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

      // first loop
      auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
      if (tid == 0) {
#pragma unroll
        for (int j = pad; j < PACKSIZE; ++j) {
          updateHist(val[j]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          updateHist(val[j]);
        }
      }
      idx += BLOCK * gridDim.x;

      // main loop
      for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
        auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          updateHist(val[j]);
        }
      }

      // tail
      for (idx = tid + step * stepSize; idx < taskLen;
           idx += BLOCK * gridDim.x) {
        auto val =
            loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
        updateHist(val);
      }
    } else {
      for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
        auto val =
            loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
        updateHist(val);
      }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < histLen; i += BLOCK) {
      if (blockHist[i] > 0) {
        atomicAdd(&histPtr[taskId * histLen + i], blockHist[i]);
      }
    }
  }
  return;
}

template <int BLOCK, int LEFT, int RIGHT, int PACKSIZE, bool WITHSCALE,
          bool LARGEST, typename T>
__global__ void __launch_bounds__(256)
    selectCandidateEx(const T* dataIn, typename ComputeT<T>::type* dataOut,
                      int* globalCount, const int* binId,
                      const int* taskOffsetPtr, const int stride,
                      const int taskNum) {
  using InVec = VT<T, PACKSIZE>;
  using CompT = typename ComputeT<T>::type;

  __shared__ int blockCount[1];
  __shared__ CompT blockCache[2 * BLOCK * PACKSIZE];

  auto resetBlockCountSync = [&]() {
    if (threadIdx.x == 0) {
      blockCount[0] = 0;
    }
    __syncthreads();
    return;
  };

  // check whether hit and stage to cache if necessary
  auto stageIfHit = [&](const CompT& val, const int& mask) {
    if (mask == getBinId<LEFT, RIGHT>(val)) {
      int pos = atomicAdd(blockCount, 1);
      blockCache[pos] = val;
    }
    return;
  };

  /* count: count of occupied cache slots.
   * WARNING: after flushing, threads may be unsync.
   */
  auto flushCache = [&](const int& taskId, const int& count) {
    if (threadIdx.x == 0) {
      blockCount[0] = atomicAdd(globalCount + taskId, count);
    }
    __syncthreads();

    const int offset = blockCount[0];
    __syncthreads();

    const int writeBase = taskId * stride + offset;
    int pos = threadIdx.x;
    while (pos < count) {
      dataOut[writeBase + pos] = blockCache[pos];
      pos += BLOCK;
    }
    return;
  };

  const int tid = blockIdx.x * BLOCK + threadIdx.x;
  const int stepSize = gridDim.x * BLOCK * PACKSIZE;

  for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
    // reset for each task
    resetBlockCountSync();

    int offset = taskOffsetPtr[taskId];
    const int pad = offset & (PACKSIZE - 1);
    offset -= pad;

    // scaling factor
    const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);
    const int mask = binId[taskId];

    const int taskLen = taskOffsetPtr[taskId + 1] - offset;
    const int step = taskLen / stepSize;

    if (step > 0) {
      int idx = tid;
      const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

      // first loop
      auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
      if (tid == 0) {
#pragma unroll
        for (int j = pad; j < PACKSIZE; ++j) {
          stageIfHit(val[j], mask);
        }
      } else {
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          stageIfHit(val[j], mask);
        }
      }
      idx += BLOCK * gridDim.x;

      // main loop
      for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
        auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          stageIfHit(val[j], mask);
        }

        // check cache capacity, write back to flush cache if necessary
        __syncthreads();
        const int count = blockCount[0];
        __syncthreads();

        if (count > BLOCK * PACKSIZE) {
          flushCache(taskId, count);
          // reset & sync after flushing
          resetBlockCountSync();
        }
      }

      // tail
      for (idx = step * stepSize + tid; idx < taskLen;
           idx += BLOCK * gridDim.x) {
        auto val =
            loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
        stageIfHit(val, mask);
      }

      // flush cache anyway
      __syncthreads();
      const int count = blockCount[0];
      __syncthreads();
      flushCache(taskId, count);
      // no need to reset & sync, because this is done at the beginning of
      // each pass
    } else {
      for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
        auto val =
            loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
        stageIfHit(val, mask);
      }

      // flush cache anyway
      __syncthreads();
      const int count = blockCount[0];
      __syncthreads();
      flushCache(taskId, count);
      // no need to reset & sync, because this is done at the beginning of
      // each pass
    }
  }
  return;
}

#define RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j) \
  do {                                                     \
    if (LARGEST && (val_s)[(j)] > kThEle) {                \
      stagePair((val)[(j)], getPackIndex((idx), (j)));     \
    } else if (!LARGEST && (val_s)[(j)] < kThEle) {        \
      stagePair((val)[(j)], getPackIndex((idx), (j)));     \
    }                                                      \
    if ((val_s)[(j)] == kThEle) {                          \
      if (atomicSub(boundaryCount + taskId, 1) > 0) {      \
        stagePair((val)[(j)], getPackIndex((idx), (j)));   \
      }                                                    \
    }                                                      \
  } while (0)

#define RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx) \
  do {                                                \
    if (LARGEST && (val_s) > kThEle) {                \
      stagePair((val), getItemIndex((idx)));          \
    } else if (!LARGEST && (val_s) < kThEle) {        \
      stagePair((val), getItemIndex((idx)));          \
    }                                                 \
    if ((val_s) == kThEle) {                          \
      if (atomicSub(boundaryCount + taskId, 1) > 0) { \
        stagePair((val), getItemIndex((idx)));        \
      }                                               \
    }                                                 \
  } while (0)

template <bool LARGEST, int BLOCK, int PACKSIZE, int CACHESIZE, bool WITHSCALE,
          bool WITHIDXIN = 0, typename IdxType, typename ValType>
__global__ void __launch_bounds__(256)
    filter(const ValType* dataIn, const IdxType* idxIn,
           const typename ComputeT<ValType>::type* kThElePtr, ValType* valOut,
           IdxType* idxOut, int* globalCount, int* boundaryCount,
           const int* taskOffsetPtr, const int stride, const int K,
           const int taskNum) {
  using InVec = VT<ValType, PACKSIZE>;
  using CompT = typename ComputeT<ValType>::type;

  __shared__ int blockCount[1];
  __shared__ ValType valBlockCache[CACHESIZE];
  __shared__ IdxType idxBlockCache[CACHESIZE];

  /*
   * WARNING: after flushing, threads may be unsync.
   */
  auto flushCache = [&](const int& taskId) {
    __syncthreads();
    int count = blockCount[0];
    __syncthreads();

    if (threadIdx.x == 0) {
      blockCount[0] = atomicAdd(globalCount + taskId, count);
    }
    __syncthreads();

    const int offset = blockCount[0];
    __syncthreads();

    const int writeBase = taskId * K + offset;
    int pos = threadIdx.x;
    while (pos < count) {
      valOut[writeBase + pos] = valBlockCache[pos];
      idxOut[writeBase + pos] = idxBlockCache[pos];
      pos += BLOCK;
    }
    return;
  };

  const int tid = blockIdx.x * BLOCK + threadIdx.x;
  const int stepSize = PACKSIZE * BLOCK * gridDim.x;

  for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
    // fixed length output, only reset cache once for each task
    if (threadIdx.x == 0) {
      blockCount[0] = 0;
    }
    __syncthreads();

    int offset = taskOffsetPtr[taskId];
    const int originalLen = taskOffsetPtr[taskId + 1] - offset;

    // padding if misaligned
    const int pad = offset & (PACKSIZE - 1);
    offset -= pad;
    const int taskLen = originalLen + pad;
    const int step = taskLen / stepSize;

    auto getPackIndex = [&](const int& idx, const int& j) {
      return WITHIDXIN ? idxIn[offset + idx * PACKSIZE + j]
                       : static_cast<IdxType>(idx * PACKSIZE + j - pad);
    };

    auto getItemIndex = [&](const int& idx) {
      return WITHIDXIN ? idxIn[offset + idx] : static_cast<IdxType>(idx - pad);
    };

    auto stagePair = [&](const auto& value, const IdxType& index) {
      int pos = atomicAdd(blockCount, 1);
      valBlockCache[pos] = static_cast<ValType>(value);
      idxBlockCache[pos] = index;
      return;
    };

    /*
     * N <= K, just copy all
     */
    if (originalLen <= K) {
      if (step > 0) {
        int idx = tid;
        const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

        // first loop
        InVec val = readPtr[idx];
        if (tid == 0) {
#pragma unroll
          for (int j = pad; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
        } else {  // tid != 0
#pragma unroll
          for (int j = 0; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
        }
        idx += BLOCK * gridDim.x;

        // main loop
        for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
          InVec val = readPtr[idx];
#pragma unroll
          for (int j = 0; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
        }

        // tail
        for (idx = tid + step * stepSize; idx < taskLen;
             idx += BLOCK * gridDim.x) {
          ValType val = dataIn[offset + idx];
          stagePair(val, getItemIndex(idx));
        }
      } else {  // step == 0
        for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
          ValType val = dataIn[offset + idx];
          stagePair(val, getItemIndex(idx));
        }
      }
      // flush cache once for each task
      flushCache(taskId);

      continue;
    }

    /*
     * otherwise, filter by k-th element
     */
    // scaling factor
    const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);
    const CompT kThEle = *(kThElePtr + taskId * stride);

    if (step > 0) {
      int idx = tid;
      const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

      // first loop
      InVec val = readPtr[idx];
      auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);

      if (tid == 0) {
#pragma unroll
        for (int j = pad; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
      } else {  // tid != 0
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
      }
      idx += BLOCK * gridDim.x;

      // main loop
      for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
        InVec val = readPtr[idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
      }

      // tail
      for (idx = tid + step * stepSize; idx < taskLen;
           idx += BLOCK * gridDim.x) {
        ValType val = dataIn[offset + idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
        RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
      }
    } else {  // step == 0
      for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
        ValType val = dataIn[offset + idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
        RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
      }
    }

    // flush cache once for each task
    flushCache(taskId);
  }
  return;
}

template <bool LARGEST, int BLOCK, int PACKSIZE, bool WITHSCALE,
          bool WITHIDXIN = 0, typename IdxType, typename ValType>
__global__ void __launch_bounds__(256)
    filter_general(const ValType* dataIn, const IdxType* idxIn,
                   const typename ComputeT<ValType>::type* kThElePtr,
                   ValType* valOut, IdxType* idxOut, int* globalCount,
                   int* boundaryCount, const int* taskOffsetPtr,
                   const int stride, const int K, const int taskNum) {
  using InVec = VT<ValType, PACKSIZE>;
  using CompT = typename ComputeT<ValType>::type;

  __shared__ int blockCount[1];
  __shared__ ValType valBlockCache[BLOCK * PACKSIZE];
  __shared__ IdxType idxBlockCache[BLOCK * PACKSIZE];

  auto resetBlockCountSync = [&]() {
    if (threadIdx.x == 0) {
      blockCount[0] = 0;
    }
    __syncthreads();
    return;
  };

  /*
   * WARNING: after flushing, threads may be unsync.
   */
  auto flushCache = [&](const int& taskId) {
    __syncthreads();
    int count = blockCount[0];
    __syncthreads();

    if (threadIdx.x == 0) {
      blockCount[0] = atomicAdd(globalCount + taskId, count);
    }
    __syncthreads();

    const int offset = blockCount[0];
    __syncthreads();

    const int writeBase = taskId * K + offset;
    int pos = threadIdx.x;
    while (pos < count) {
      valOut[writeBase + pos] = valBlockCache[pos];
      idxOut[writeBase + pos] = idxBlockCache[pos];
      pos += BLOCK;
    }
    return;
  };

  const int tid = blockIdx.x * BLOCK + threadIdx.x;
  const int stepSize = PACKSIZE * BLOCK * gridDim.x;

  for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
    // reset for each task
    resetBlockCountSync();

    int offset = taskOffsetPtr[taskId];
    const int originalLen = taskOffsetPtr[taskId + 1] - offset;

    // padding if misaligned
    const int pad = offset & (PACKSIZE - 1);
    offset -= pad;
    const int taskLen = originalLen + pad;
    const int step = taskLen / stepSize;

    auto getPackIndex = [&](const int& idx, const int& j) {
      return WITHIDXIN ? idxIn[offset + idx * PACKSIZE + j]
                       : static_cast<IdxType>(idx * PACKSIZE + j - pad);
    };

    auto getItemIndex = [&](const int& idx) {
      return WITHIDXIN ? idxIn[offset + idx] : static_cast<IdxType>(idx - pad);
    };

    auto stagePair = [&](const auto& value, const IdxType& index) {
      int pos = atomicAdd(blockCount, 1);
      valBlockCache[pos] = static_cast<ValType>(value);
      idxBlockCache[pos] = index;
      return;
    };

    /*
     * N <= K, just copy all
     */
    if (originalLen <= K) {
      if (step > 0) {
        int idx = tid;
        const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

        // first loop
        InVec val = readPtr[idx];
        if (tid == 0) {
#pragma unroll
          for (int j = pad; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
        } else {  // tid != 0
#pragma unroll
          for (int j = 0; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
        }
        flushCache(taskId);
        idx += BLOCK * gridDim.x;

        // main loop
        for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
          resetBlockCountSync();

          InVec val = readPtr[idx];
#pragma unroll
          for (int j = 0; j < PACKSIZE; ++j) {
            stagePair(val[j], getPackIndex(idx, j));
          }
          flushCache(taskId);
        }

        // tail
        resetBlockCountSync();

        for (idx = tid + step * stepSize; idx < taskLen;
             idx += BLOCK * gridDim.x) {
          ValType val = dataIn[offset + idx];
          stagePair(val, getItemIndex(idx));
        }
      } else {  // step == 0
        for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
          ValType val = dataIn[offset + idx];
          stagePair(val, getItemIndex(idx));
        }
      }
      flushCache(taskId);
      // no need to reset & sync, because this is done at the beginning of
      // each pass

      continue;
    }

    /*
     * otherwise, filter by k-th element
     */
    // scaling factor
    const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);
    const CompT kThEle = *(kThElePtr + taskId * stride);

    if (step > 0) {
      int idx = tid;
      const InVec* readPtr = reinterpret_cast<const InVec*>(dataIn + offset);

      // first loop
      InVec val = readPtr[idx];
      auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);

      if (tid == 0) {
#pragma unroll
        for (int j = pad; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
      } else {  // tid != 0
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
      }
      flushCache(taskId);
      idx += BLOCK * gridDim.x;

      // main loop
      for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
        resetBlockCountSync();

        InVec val = readPtr[idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
#pragma unroll
        for (int j = 0; j < PACKSIZE; ++j) {
          RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
        }
        flushCache(taskId);
      }

      // tail
      resetBlockCountSync();

      for (idx = tid + step * stepSize; idx < taskLen;
           idx += BLOCK * gridDim.x) {
        ValType val = dataIn[offset + idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
        RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
      }
    } else {  // step == 0
      for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
        ValType val = dataIn[offset + idx];
        auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
        RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
      }
    }
    flushCache(taskId);
    // no need to reset & sync, because this is done at the beginning of
    // each pass
  }
  return;
}

#undef RADIX_TOPK_FILTER_STAGE_PACKED
#undef RADIX_TOPK_FILTER_STAGE_ITEM

//===================================
// thrust::sort helper kernel
//===================================
#ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
#ifdef CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096

template <int BLOCK, int UNROLL, bool ASCEND, typename T>
__global__ void __launch_bounds__(1024)
    convertNanInPlace(T* valPtr, const int K) {
  const uint32_t taskId = blockIdx.x;
  const uint32_t pass = blockIdx.y;

  const uint32_t tid = pass * BLOCK * UNROLL + threadIdx.x;
  T* dataIn = valPtr + taskId * K;

  T regs[UNROLL];

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (tid + i * BLOCK < K) {
      regs[i] = dataIn[tid + i * BLOCK];
    }
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (isNaN(regs[i])) {
      regs[i] = ASCEND ? getMax<T>() : getMin<T>();
    }
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    if (tid + i * BLOCK < K) {
      dataIn[tid + i * BLOCK] = regs[i];
    }
  }
  return;
}

#endif  // CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
#endif  // CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER

}  // namespace radix_topk
}  // namespace cuda
}  // namespace allspark
