/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_sum.hpp
 */

#ifndef DNN_INCLUDE_CUDA_PREFIX_SUM_HPP_
#define DNN_INCLUDE_CUDA_PREFIX_SUM_HPP_

#include <utils.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>
#include <cuda/intrinsic/warp_shuffle.hpp>
#include <cuda/intrinsic/integer_arithmetic.hpp>

namespace hiednn {

namespace cuda {

namespace prefix_sum {

// ------------------------------------------------------------
// warp scan step by 'shfl'
// ------------------------------------------------------------
__device__ __forceinline__ int32_t WarpScanStep(
        uint32_t mask, const int32_t &var, int offset, int width) {
    int32_t ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .s32 r0; \n"
        "  .reg .pred p0; \n"
        "  shfl.sync.up.b32 r0|p0, %0, %1, %2, %3; \n"
        "  @p0 add.s32 %0, %0, r0; }\n"
        : "+r"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

__device__ __forceinline__ uint32_t WarpScanStep(
        uint32_t mask, const uint32_t &var, int offset, int width) {
    uint32_t ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .u32 r0; \n"
        "  .reg .pred p0; \n"
        "  shfl.sync.up.b32 r0|p0, %0, %1, %2, %3; \n"
        "  @p0 add.u32 %0, %0, r0; }\n"
        : "+r"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

__device__ __forceinline__ int64_t WarpScanStep(
        uint32_t mask, const int64_t &var, int offset, int width) {
    int64_t ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .s64 r0; \n"
        "  .reg .u32 lo; \n"
        "  .reg .u32 hi; \n"
        "  .reg .pred p0; \n"
        "  mov.b64 {lo, hi}, %0; \n"
        "  shfl.sync.up.b32 lo|p0, lo, %1, %2, %3; \n"
        "  shfl.sync.up.b32 hi|p0, hi, %1, %2, %3; \n"
        "  mov.b64 r0, {lo, hi}; \n"
        "  @p0 add.s64 %0, %0, r0; }\n"
        : "+l"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

__device__ __forceinline__ uint64_t WarpScanStep(
        uint32_t mask, const uint64_t &var, int offset, int width) {
    uint64_t ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .u64 r0; \n"
        "  .reg .u32 lo; \n"
        "  .reg .u32 hi; \n"
        "  .reg .pred p0; \n"
        "  mov.b64 {lo, hi}, %0; \n"
        "  shfl.sync.up.b32 lo|p0, lo, %1, %2, %3; \n"
        "  shfl.sync.up.b32 hi|p0, hi, %1, %2, %3; \n"
        "  mov.b64 r0, {lo, hi}; \n"
        "  @p0 add.u64 %0, %0, r0; }\n"
        : "+l"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

__device__ __forceinline__ float WarpScanStep(
        uint32_t mask, const float &var, int offset, int width) {
    float ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .f32 r0; \n"
        "  .reg .pred p0; \n"
        "  shfl.sync.up.b32 r0|p0, %0, %1, %2, %3; \n"
        "  @p0 add.f32 %0, %0, r0; }\n"
        : "+f"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

__device__ __forceinline__ double WarpScanStep(
        uint32_t mask, const double &var, int offset, int width) {
    double ret = var;
    int shfl_c = (WARP_SIZE - width) << 8;
    asm volatile (
        "{ .reg .f64 r0; \n"
        "  .reg .u32 lo; \n"
        "  .reg .u32 hi; \n"
        "  .reg .pred p0; \n"
        "  mov.b64 {lo, hi}, %0; \n"
        "  shfl.sync.up.b32 lo|p0, lo, %1, %2, %3; \n"
        "  shfl.sync.up.b32 hi|p0, hi, %1, %2, %3; \n"
        "  mov.b64 r0, {lo, hi}; \n"
        "  @p0 add.f64 %0, %0, r0; }\n"
        : "+d"(ret)
        : "r"(offset), "r"(shfl_c), "r"(mask));
    return ret;
}

// ------------------------------------------------------------
// struct TilePrefix
// ------------------------------------------------------------

/*
 * tile prefix workspace status:
 *     INVALID: no data
 *     TILE_SUM: tile reduce sum
 *     TILE_PREFIX: tile inclusive prefix sum
 */
typedef enum {
    INVALID     = 0,
    TILE_SUM    = 1,
    TILE_PREFIX = 2,
} TileStatus;

namespace internal {

template <int SIZE>
struct StatType;

template <>
struct StatType<1> {
    using T = uint8_t;
};

template <>
struct StatType<2> {
    using T = uint16_t;
};

template <>
struct StatType<4> {
    using T = uint32_t;
};

template <>
struct StatType<8> {
    using T = uint64_t;
};

}  // namespace internal

/*
 * TilePrefix is 2*sizeof(T) aligned to make sure that the stat and data are
 * load and store in one ldg/stg instruction and one memory transaction
 */
template <typename DataT>
struct alignas(2 * sizeof(DataT)) TilePrefix {
    using StatT = typename internal::StatType<sizeof(DataT)>::T;
    StatT stat;
    DataT data;
    __host__ __device__ TilePrefix() {}
    __host__ __device__ TilePrefix(TileStatus s, DataT d) : stat(s), data(d) {}
};

// ------------------------------------------------------------
// scan mode
// ------------------------------------------------------------
typedef enum {
    INCLUSIVE = 0,
    EXCLUSIVE = 1,
} ScanMode;

// ------------------------------------------------------------
// warp prefixsum
//
// ATTENTION:
// all threads of the warp should be actively participating in
// this function
// ------------------------------------------------------------
template <ScanMode MODE,
          typename T>
__device__ __forceinline__
T WarpPrefixSum(const T &x) {
    T ret = x;
    #pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        ret = WarpScanStep(0xffffffff, ret, i, WARP_SIZE);
    }

    if (MODE == EXCLUSIVE) {
        ret -= x;
    }
    return ret;
}

// ------------------------------------------------------------
// global prefix scan for 1D array
// ------------------------------------------------------------
constexpr int TILE_PREFIX_PADDING = WARP_SIZE;

// all threads of the warp should be actively participating in this function
template <ScanMode MODE,
          int BLOCK,
          typename T,            // type of input/output and accumulate
          typename TilePrefixT>  // tilePrefix type for type T
__device__ __forceinline__
T GlobalPrefixSum(const T &x,
                  uint32_t tileId,
                  uint32_t warpId,
                  uint32_t laneId,
                  TilePrefixT *tilePrefix) {
    static_assert(BLOCK >= WARP_SIZE && BLOCK % WARP_SIZE == 0,
                  "invlaid template parameter `BLOCK`");

    __shared__ T smem[CTA_WARP_MAX];

    // warp scan
    T threadPrefix = WarpPrefixSum<INCLUSIVE>(x);

    if (laneId == WARP_SIZE - 1) {
        smem[warpId] = threadPrefix;
    }
    if (MODE == EXCLUSIVE) {
        threadPrefix -= x;
    }
    __syncthreads();

    if (warpId == 0) {
        // inter-warp scan
        T tileSum = 0;
        T warpExclusivePrefix = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / WARP_SIZE; ++i) {
            if (laneId == i) {
                warpExclusivePrefix = tileSum;
            }
            tileSum += smem[i];
        }

        TilePrefixT *updatePtr = tilePrefix + TILE_PREFIX_PADDING + tileId;

        // adaptive look-back: store tile sum
        if (laneId == WARP_SIZE - 1) {
            TilePrefixT tileSumStg(TILE_SUM, tileSum);
            Stg<CG>(tileSumStg, updatePtr);
        }

        // adaptive look-back: look-back scan
        const TilePrefixT *lookbackPtr = updatePtr - 1 - laneId;
        TilePrefixT tilePrefixLdg;
        tilePrefixLdg.data = 0;
        T tilePrefixAcc = 0;

        do {
            tilePrefixAcc += tilePrefixLdg.data;
            do {
                // make sure Ldg<CG> inside the while-loop
                __threadfence_block();
                Ldg<CG>(&tilePrefixLdg, lookbackPtr);
            } while (__any_sync(0xffffffff, tilePrefixLdg.stat == INVALID));
            lookbackPtr -= WARP_SIZE;
        } while (__all_sync(0xffffffff, tilePrefixLdg.stat != TILE_PREFIX));

        // accumulate tilePrefix with TILE_PREFIX status
        uint32_t tilePrefixMask =
            __ballot_sync(0xffffffff, tilePrefixLdg.stat == TILE_PREFIX);
        uint32_t firstLane = Ffs(tilePrefixMask);
        if (laneId < firstLane) {
            tilePrefixAcc += tilePrefixLdg.data;
        }

        // calculate tilePrefix via warp reduce
        #pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            tilePrefixAcc += ShflBfly(0xffffffff, tilePrefixAcc, i, WARP_SIZE);
        }

        // adaptive look-back: store tile inclusive prefixsum
        if (laneId == WARP_SIZE - 1) {
            TilePrefixT tilePrefixStg(TILE_PREFIX, tilePrefixAcc + tileSum);
            Stg<CG>(tilePrefixStg, updatePtr);
        }

        smem[laneId] = tilePrefixAcc + warpExclusivePrefix;
    }

    __syncthreads();

    return threadPrefix + smem[warpId];
}

// ------------------------------------------------------------
// tile prefix initialization for 1D prefix scan
// ------------------------------------------------------------
template <int BLOCK, int PADDING, typename TilePrefixT>
__global__ void TilePrefixInitKernel(TilePrefixT *tilePrefix, uint32_t n) {
    uint32_t idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx < n) {
        tilePrefix[idx] = idx < PADDING ?
                          TilePrefixT(TILE_PREFIX, 0) :
                          TilePrefixT(INVALID, 0);
    }
}

template <typename TilePrefixT>
void TilePrefixInit(
        TilePrefixT *tilePrefix,
        size_t n,  // number of TilePrefixT elements to be initialized
        cudaStream_t stream) {
    const int BLOCK = 128;
    const int PADDING = TILE_PREFIX_PADDING;
    int grid = UIntDivRU<uint32_t>(n, BLOCK);
    TilePrefixInitKernel<BLOCK, PADDING><<<grid, BLOCK, 0, stream>>>(
        tilePrefix, n);
}

inline size_t TilePrefixSize(size_t nTiles) {
    return nTiles + TILE_PREFIX_PADDING;
}

}  // namespace prefix_sum

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_PREFIX_SUM_HPP_

