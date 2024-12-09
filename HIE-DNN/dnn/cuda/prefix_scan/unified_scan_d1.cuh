/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unified_scan_d1.cuh
 */

#ifndef DNN_CUDA_PREFIX_SCAN_UNIFIED_SCAN_D1_CUH_
#define DNN_CUDA_PREFIX_SCAN_UNIFIED_SCAN_D1_CUH_

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <utils.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/prefix_sum.hpp>
#include <cuda/intrinsic/global_memory.hpp>
#include <cuda/intrinsic/warp_shuffle.hpp>

#include "scan_utils.cuh"

namespace hiednn {

namespace cuda {

namespace unified_scan_d1 {

using prefix_sum::WarpScanStep;

// a row of 2D array is scanned in a thread block
template <int BLOCK,        // thread block size
          int UNROLL,       // items loaded each thread
          bool REVERSE,     // true for reverse mode (suffix scan)
          typename IdxT,    // type of index or offset
          typename DataT>   // type of input/output array
struct AlignedDataLoader {
    static const uint32_t TILE_SIZE = BLOCK * UNROLL;

    /*
     * data alignment must not more than TILE_SIZE,
     * or shared memory ld/st will be out of bound
     */
    static_assert(MEM_ALIGN_BYTE / sizeof(DataT) <= TILE_SIZE,
                  "unified_scan_d1::DataLoader: invalid TILE_SIZE");

    /*
     * firstTileSize:
     *     size of the first tile.
     *     tiles are MEM_ALIGN_BYTE aligned except the first tile
     * lastTileSize:
     *     size of the last tile.
     * nTiles:
     *     number of tiles with size TILE_SIZE
     *
     * for example (REVERSE = false):
     * firstTileSize = 3, lastTileSize = 5, nTiles = 2
     *
     * first aligned address
     *     |
     *     | <--TILE_SIZE-->| <--TILE_SIZE-->|
     * |---|----------------|----------------|-----|
     *   ^                                      ^
     * firstTileSize                    lastTileSize
     *
     * and for REVERSE = true:
     * firstTileSize = 5, lastTileSize = 3, nTiles = 2
     *
     *                            first aligned address
     *                                       |
     *     | <--TILE_SIZE-->| <--TILE_SIZE-->|
     * |---|----------------|----------------|-----|
     *   ^                                      ^
     * lastTileSize                    firstTileSize
     */
    uint32_t firstTileSize;
    uint32_t lastTileSize;
    IdxT nTiles;

    const DataT *xPtr;
    DataT *yPtr;

    __device__ __forceinline__
    AlignedDataLoader(const DataT *x,          // pointer to input array
                      DataT *y,                // pointer to output array
                      const IdxT &n,           // number of columns of 2D array
                      const uint32_t &mIdx) {  // row index
        const uint32_t DATA_ALIGN = MEM_ALIGN_BYTE / sizeof(DataT);
        static_assert(TILE_SIZE % DATA_ALIGN == 0,
                      "unified_scan_d1::AlignedDataLoader: invalid TILE_SIZE");

        IdxT rowOffset = (REVERSE ? mIdx + 1 : mIdx) * n;
        uint32_t subTile = static_cast<uint32_t>(rowOffset) % DATA_ALIGN;

        xPtr = REVERSE ?
               x + rowOffset - 1 - threadIdx.x :
               x + rowOffset + threadIdx.x;
        yPtr = REVERSE ?
               y + rowOffset - 1 - threadIdx.x :
               y + rowOffset + threadIdx.x;

        if (subTile == 0) {
            // row aligned
            firstTileSize = 0;
            nTiles = n / TILE_SIZE;
            lastTileSize = n % TILE_SIZE;
        } else {
            // row unaligned
            if (REVERSE) {
                firstTileSize = TILE_SIZE - DATA_ALIGN + subTile <= n ?
                                TILE_SIZE - DATA_ALIGN + subTile : n;
            } else {
                firstTileSize = TILE_SIZE - subTile <= n ?
                                TILE_SIZE - subTile : n;
            }

            nTiles = (n - firstTileSize) / TILE_SIZE;
            lastTileSize = (n - firstTileSize) % TILE_SIZE;
        }
    }

    template <typename T>
    __device__ __forceinline__
    void GotoNextTile(T *ptr, uint32_t tileSize = TILE_SIZE) {
        if (REVERSE) {
            (*ptr) -= tileSize;
        } else {
            (*ptr) += tileSize;
        }
    }

    // load a tile with size @tileSize to shared memory
    template <typename SmemT>
    __device__ __forceinline__
    void Load(SmemT *smem, uint32_t tileSize) {
        uint32_t ldgIdx = threadIdx.x;
        uint32_t ldgCount = tileSize > ldgIdx ?
                            UIntDivRU<uint32_t>(tileSize - ldgIdx, BLOCK) : 0;
        DataT reg[UNROLL];
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            reg[i] = DataT(0);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < ldgCount) {
                Ldg<NC, LTCMAX>(&reg[i], REVERSE ?
                                xPtr - i * BLOCK :
                                xPtr + i * BLOCK);
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            smem[threadIdx.x + i * BLOCK] = static_cast<SmemT>(reg[i]);
        }

        GotoNextTile(&xPtr, tileSize);
    }

    // load a tile with size TILE_SIZE to shared memory
    template <typename SmemT>
    __device__ __forceinline__
    void Load(SmemT *smem) {
        DataT reg[UNROLL];
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg<NC, LTCMAX>(&reg[i],
                REVERSE ? xPtr - i * BLOCK : xPtr + i * BLOCK);
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            smem[threadIdx.x + i * BLOCK] = static_cast<SmemT>(reg[i]);
        }

        GotoNextTile(&xPtr);
    }

    // store a tile with size @tileSize to global memory
    template <typename SmemT>
    __device__ __forceinline__
    void Store(const SmemT *smem, uint32_t tileSize) {
        SmemT reg[UNROLL];
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            reg[i] = smem[threadIdx.x + i * BLOCK];
        }

        uint32_t stgIdx = threadIdx.x;
        uint32_t stgCount = tileSize > stgIdx ?
                            UIntDivRU<uint32_t>(tileSize - stgIdx, BLOCK) : 0;
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < stgCount) {
                yPtr[i * (REVERSE ? -BLOCK : BLOCK)] = reg[i];
            }
        }

        GotoNextTile(&yPtr, tileSize);
    }

    // store a tile with size TILE_SIZE to global memory
    template <typename SmemT>
    __device__ __forceinline__
    void Store(const SmemT *smem) {
        SmemT reg[UNROLL];
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            reg[i] = smem[threadIdx.x + i * BLOCK];
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yPtr[i * (REVERSE ? -BLOCK : BLOCK)] = reg[i];
        }

        GotoNextTile(&yPtr);
    }
};

// just load 1 tile
template <int BLOCK,         // thread block size
          int TILE_M,        // tile height
          int TILE_N,        // tile width
          int UNROLL,        // items loaded each thread
          bool REVERSE,      // true for reverse mode (suffix scan)
          typename IdxT,     // type of index or offset
          typename XSmemT,   // xSmem datatype
          typename DataT>    // type of input/output array
struct DataLoader {
    static_assert(BLOCK >= WARP_SIZE && BLOCK % WARP_SIZE == 0,
                  "unified_scan_d1::DataLoader: invalid BLOCK");
    static_assert(BLOCK % TILE_N == 0 || TILE_N % BLOCK == 0,
                  "unified_scan_d1::DataLoader: invalid BLOCK and TILE_N");
    static_assert(BLOCK * UNROLL == TILE_M * TILE_N,
                  "unified_scan_d1::DataLoader: invalid TILE configuration");

    /*
     * shared memory map:
     * padding 1 item each row to avoid bank conflict for thread reduce,
     * for example:
     *
     * UNROLL = 8, XSmemT = float:
     *
     *     |<-----------32*4Byte---------->|4Byte|
     *     |<--8-->|<--8-->|<--8-->|<--8-->|<-1->|
     *     |-------|-------|-------|-------|-----|
     *     |       |       |       |       |     |
     *     |-------|-------|-------|-------|-----|
     *     |       |       |       |       |     |
     *     |-------|-------|-------|-------|-----|
     *     |  ...  |  ...  |  ...  |  ...  | ... |
     *
     * UNROLL = 8, XSmemT = double:
     *
     *     |<-----------32*4Byte---------->|8Byte|
     *     |<------8------>|<------8------>|<-1->|
     *     |---------------|---------------|-----|
     *     |               |               |     |
     *     |---------------|---------------|-----|
     *     |               |               |     |
     *     |---------------|---------------|-----|
     *     |      ...      |      ...      | ... |
     *
     * ITEM_COLS: items stored in each row of shared memory
     * SMEM_SIZE: shared memory size (size in items)
     */
    static const int ITEM_COLS = SMEM_BANKS_ROW / sizeof(XSmemT);
    static const int SMEM_COLS = ITEM_COLS + 1;
    static const int SMEM_SIZE = SMEM_COLS * (BLOCK / (ITEM_COLS / UNROLL));

    static_assert(UNROLL <= ITEM_COLS && IsPowOf2(UNROLL),
                  "unified_scan_d1::DataLoader: invalid UNROLL");

    /*
     * for TILE_N = 512, TILE_M = 4, BLOCK = 256:
     *
     *              |<---- N_LDG=2, N_LDG_STEP=BLOCK ---->|
     *              |<--------------512 items------------>|
     *              |<---256 thread--->|
     *          ----|------------------|------------------|
     *           |  |//////////////////|                  |
     *           |  |------------------|------------------|
     *      M_LDG=4 |                  |                  |
     * M_LDG_STEP=1 |------------------|------------------|
     *           |  |                  |                  |
     *           |  |------------------|------------------|
     *           |  |                  |                  |
     *          ----|------------------|------------------|
     *
     * for TILE_N = 128, TILE_M = 4, BLOCK = 256:
     *
     *              |<---- N_LDG=1, N_LDG_STEP=TILE_N --->|
     *              |<--------------128 items------------>|
     *              |<------------128 thread------------->|
     *          ----|-------------------------------------|---
     *           |  |/////////////////////////////////////| |
     *           |  |-------------------------------------| 2*128 thread
     *      M_LDG=4 |/////////////////////////////////////| |
     * M_LDG_STEP=2 |-------------------------------------|---
     *           |  |                                     |
     *           |  |-------------------------------------|
     *           |  |                                     |
     *          ----|-------------------------------------|
     */
    static const int N_LDG = TILE_N > BLOCK ? TILE_N / BLOCK : 1;
    static const int M_LDG = TILE_N > BLOCK ?
                             TILE_M : TILE_M / (BLOCK / TILE_N);

    static const int N_LDG_STEP = TILE_N > BLOCK ? BLOCK : TILE_N;
    static const int M_LDG_STEP = TILE_N > BLOCK ? 1 : BLOCK / TILE_N;

    uint32_t mTid;
    uint32_t nTid;

    const DataT *__restrict__ xPtr;
    DataT *yPtr;

    XSmemT *smemPtr;
    const uint32_t SMEM_STEP = BLOCK / ITEM_COLS * SMEM_COLS;

    static __device__ __forceinline__
    XSmemT *GetXSmemScanPtr(XSmemT *xSmem) {
        const uint32_t THREAD_COLS = ITEM_COLS / UNROLL;
        return xSmem + threadIdx.x / THREAD_COLS * SMEM_COLS +
                       threadIdx.x % THREAD_COLS * UNROLL;
    }

    __device__ __forceinline__
    DataLoader(const DataT *x,            // pointer to input array
               DataT *y,                  // pointer to output array
               const IdxT &n,             // number of columns of 2D array
               const uint32_t &blockId,   // thread block id
               XSmemT *xSmem) {           // shared memory pointer for input
        mTid = BLOCK > TILE_N ? threadIdx.x / TILE_N : 0;
        nTid = BLOCK > TILE_N ? threadIdx.x % TILE_N : threadIdx.x;

        // global memory ld/st pointer
        xPtr = REVERSE ? x + (blockId * TILE_M + mTid) * n + n - 1 - nTid :
                         x + (blockId * TILE_M + mTid) * n + nTid;
        yPtr = REVERSE ? y + (blockId * TILE_M + mTid) * n + n - 1 - nTid :
                         y + (blockId * TILE_M + mTid) * n + nTid;

        // shared memory st pointer
        smemPtr = xSmem + threadIdx.x / ITEM_COLS * SMEM_COLS +
                          threadIdx.x % ITEM_COLS;
    }

    template <bool FULL_TILE>
    __device__ __forceinline__
    void Load(const IdxT &m, const IdxT &n) {
        DataT reg[M_LDG][N_LDG];
        #pragma unroll
        for (int i = 0; i < M_LDG; ++i) {
            if (!FULL_TILE && mTid + i * M_LDG_STEP >= m) {
                break;
            }

            DataT ldgReg[N_LDG];
            #pragma unroll
            for (int j = 0; j < N_LDG; ++j) {
                ldgReg[j] = DataT(0);
            }
            #pragma unroll
            for (int j = 0; j < N_LDG; ++j) {
                if (nTid + j * N_LDG_STEP < n) {
                    ldgReg[j] = REVERSE ?
                        xPtr[i * M_LDG_STEP * n - j * N_LDG_STEP] :
                        xPtr[i * M_LDG_STEP * n + j * N_LDG_STEP];
                }
            }
            #pragma unroll
            for (int j = 0; j < N_LDG; ++j) {
                reg[i][j] = static_cast<DataT>(ldgReg[j]);
            }
        }

        #pragma unroll
        for (int i = 0; i < M_LDG; ++i) {
            for (int j = 0; j < N_LDG; ++j) {
                smemPtr[(i * N_LDG + j) * SMEM_STEP] = reg[i][j];
            }
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__
    void Store(const IdxT &m, const IdxT &n) {
        XSmemT reg[M_LDG][N_LDG];
        #pragma unroll
        for (int i = 0; i < M_LDG; ++i) {
            for (int j = 0; j < N_LDG; ++j) {
                reg[i][j] = smemPtr[(i * N_LDG + j) * SMEM_STEP];
            }
        }

        #pragma unroll
        for (int i = 0; i < M_LDG; ++i) {
            if (!FULL_TILE && mTid + i * M_LDG_STEP >= m) {
                break;
            }

            #pragma unroll
            for (int j = 0; j < N_LDG; ++j) {
                if (nTid + j * N_LDG_STEP < n) {
                    if (REVERSE) {
                        yPtr[i * M_LDG_STEP * n - j * N_LDG_STEP] = reg[i][j];
                    } else {
                        yPtr[i * M_LDG_STEP * n + j * N_LDG_STEP] = reg[i][j];
                    }
                }
            }
        }
    }
};

template <int BLOCK_X,
          int BLOCK_Y,
          int UNROLL,
          bool EXCLUSIVE,
          typename CompT>
struct Scanner {
    static_assert(BLOCK_X % WARP_SIZE == 0 &&
                  IsPowOf2(BLOCK_X / WARP_SIZE) ||
                  BLOCK_X < WARP_SIZE && IsPowOf2(BLOCK_X),
                  "unified_scan_d1::Scanner: invalid BLOCK_X");

    uint32_t laneId;
    uint32_t warpId;

    CompT threadSum;           // sum of items scaned in a thread
    CompT exclusivePrefix;     // global exclusive prefix sum for a thread
    CompT tileAcc;             // tile prefix sum accumulator

    CompT xReg[UNROLL];  // temporary register buffer

    // constructor
    __device__ __forceinline__
    Scanner() : tileAcc(0),
                warpId(threadIdx.x / WARP_SIZE),
                laneId(threadIdx.x % WARP_SIZE) {}

    /*
     * get the sum of items scaned in a thread
     * @xSmem(input):
     *     shared memory pointer, associated with the loaded
     *     items for each thread.
     */
    template <typename SmemT>
    __device__ __forceinline__
    void ThreadReduce(const SmemT *xSmem) {
        threadSum = 0;
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xReg[i] = xSmem[i];
            threadSum += xReg[i];
        }
    }

    /*
     * helper function, get the warp-range inclusive prefix sum of a thread
     * @warpInclusivePrefix(input/output):
     *     pointer to a register, store the inclusive prefix sum of
     *     input *warpInclusivePrefix value as an output
     */
    template <int WIDTH>
    __device__ __forceinline__
    CompT WarpInclusiveScan(
            uint32_t mask,
            const CompT &warpInclusivePrefix) const {
        CompT ret = warpInclusivePrefix;
        #pragma unroll
        for (int i = 1; i < WIDTH; i *= 2) {
            ret = WarpScanStep(mask, ret, i, WIDTH);
        }
        return ret;
    }

    /*
     * warp scan, update @exclusivePrefix to the warp-range exclusive prfix sum
     * only work for BLOCK_X > 1.
     *
     * @warpPrefixSmem(output):
     *     store the sum of items scanned in a warp,
     *     only used for BLOCK_X > WARP_SIZE,
     */
    template <bool ROUND_SCAN = true>
    __device__ __forceinline__
    void WarpScan(CompT *warpPrefixSmem) {
        if (BLOCK_X > 1) {
            const int WIDTH = BLOCK_X < WARP_SIZE ? BLOCK_X : WARP_SIZE;

            CompT warpInclusivePrefix = threadSum;
            warpInclusivePrefix = WarpInclusiveScan<WIDTH>(
                0xffffffff, warpInclusivePrefix);

            exclusivePrefix = warpInclusivePrefix - threadSum;

            // for BLOCK_X > WARP_SIZE, store the warp-sum to shared memory for
            // the followd InterWarpScan
            if (BLOCK_X > WARP_SIZE && laneId == WARP_SIZE - 1) {
                warpPrefixSmem[warpId] = warpInclusivePrefix;
            }

            // for BLOCK_X <= WARP_SIZE, update the tile accumulator
            if (ROUND_SCAN && BLOCK_X <= WARP_SIZE) {
                exclusivePrefix += tileAcc;

                tileAcc += ShflIdx(0xffffffff,
                                   warpInclusivePrefix,
                                   BLOCK_X - 1,
                                   BLOCK_X);
            }
        } else {
            exclusivePrefix = 0;
        }
    }

    /*
     * block scan, get the block-range (row-global) exclusive prefix sum
     * of a warp and store it to shared memory @warpPrefixSmem
     *
     * only work for BLOCK_X > WARP_SIZE, and each row of tile is
     * scanned by a warp
     */
    template <bool ROUND_SCAN = true>
    __device__ __forceinline__
    void InterWarpScan(CompT *warpPrefixSmem) {
        if (BLOCK_X > WARP_SIZE) {
            bool scanExec = BLOCK_Y > 1 ?
                            warpId < BLOCK_Y && laneId < BLOCK_X / WARP_SIZE :
                            threadIdx.x < BLOCK_X / WARP_SIZE;

            if (scanExec) {
                CompT *warpPrefixPtr = BLOCK_Y > 1 ?
                    warpPrefixSmem + warpId * (BLOCK_X / WARP_SIZE) + laneId :
                    warpPrefixSmem + laneId;

                CompT blockInclusivePrefix = *warpPrefixPtr;
                CompT warpSum = blockInclusivePrefix;

                const int WIDTH = BLOCK_X / WARP_SIZE;
                const uint32_t mask = (1lu << WIDTH) - 1;
                blockInclusivePrefix = WarpInclusiveScan<WIDTH>(
                    mask, blockInclusivePrefix);

                if (ROUND_SCAN) {
                    *warpPrefixPtr = blockInclusivePrefix - warpSum + tileAcc;

                    tileAcc += ShflIdx((1lu << (BLOCK_X / WARP_SIZE)) - 1,
                                       blockInclusivePrefix,
                                       BLOCK_X / WARP_SIZE - 1,
                                       BLOCK_X / WARP_SIZE);
                } else {
                    *warpPrefixPtr = blockInclusivePrefix - warpSum;
                }
            }
        }
    }

    /*
     * get the global prefix sum of each item.
     * @warpPrefixSmem(input):
     *     shared memory pointer, associated with the global exclusive
     *     prefix sum of the warp
     * @xSmem(output):
     *     shared memory pointer, associated with the data of output array
     *     for each thread
     */
    template <typename SmemT>
    __device__ __forceinline__
    void UpdateGlobalPrefix(const CompT *warpPrefixSmem,  SmemT *xSmem) {
        if (BLOCK_X > WARP_SIZE) {
            exclusivePrefix += warpPrefixSmem[warpId];
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xSmem[i] = EXCLUSIVE ? exclusivePrefix : xReg[i] + exclusivePrefix;
            exclusivePrefix += xReg[i];
        }
    }
};

template <int BLOCK_X,
          int BLOCK_Y,
          typename CompT,
          typename ScannerT,
          typename SmemT>
__device__ __forceinline__
void BlockScan(ScannerT *scannerPtr, SmemT *xSmem) {
    /*
     * allocate a shared memory buffer for inter-warp scan.
     * only used for BLOCK_X > WARP_SIZE
     * for BLOCK_X <= WARP_SIZE, just allocate 1 item in warpPrefixSmem
     * to avoid compilation error, and the warpPrefixSmem will not be
     * allocated actually. (warpPrefixSmem is never referenced in the
     * kernel and pruned by compiler)
     */
    const int WARP_PREFIX_SMEM_SIZE = BLOCK_X > WARP_SIZE ?
                                      BLOCK_Y * (BLOCK_X / WARP_SIZE) : 1;
    __shared__ CompT warpPrefixSmem[WARP_PREFIX_SMEM_SIZE];

    auto &scanner = *scannerPtr;

    scanner.ThreadReduce(xSmem);
    scanner.WarpScan(warpPrefixSmem);

    if (BLOCK_X > WARP_SIZE) {
        __syncthreads();
        scanner.InterWarpScan(warpPrefixSmem);
        __syncthreads();
    }

    scanner.UpdateGlobalPrefix(warpPrefixSmem, xSmem);
}

template <int BLOCK,                // thread block size
          int UNROLL,               // number of items scaned by each thread
          bool EXCLUSIVE,           // true for exclusive mode
          bool REVERSE,             // true for reverse mode
          typename IdxT,            // type of offset
          typename CompT,           // compute precision
          typename DataT>           // type of input/output array
__global__ void Unified1DTilePrefixSumKernel(
        const DataT *x,             // input array pointer
        DataT *y,                   // output array pointer
        IdxT n) {                   // number of columns of 2D array
    // UNROLL must be an odd number to avoid shared memory bank conflict
    static_assert(UNROLL % 2 != 0,
                  "Unified1DTilePrefixSumKernel: invalid UNROLL");

    __shared__ CompT xSmem[BLOCK * UNROLL];

    using DataLoaderT = AlignedDataLoader<BLOCK,
                                          UNROLL,
                                          REVERSE,
                                          IdxT,
                                          DataT>;
    using ScannerT = Scanner<BLOCK, 1, UNROLL, EXCLUSIVE, CompT>;

    DataLoaderT dataLoader(x, y, n, blockIdx.x);
    ScannerT scanner;

    CompT *xScanSmem = xSmem + threadIdx.x * UNROLL;

    // scan the unaligned tile (first tile)
    if (dataLoader.firstTileSize > 0) {
        dataLoader.Load(xSmem, dataLoader.firstTileSize);
        __syncthreads();

        BlockScan<BLOCK, 1, CompT>(&scanner, xScanSmem);
        __syncthreads();

        dataLoader.Store(xSmem, dataLoader.firstTileSize);
        __syncthreads();
    }

    // scan aligned tile with size BLOCK*UNROLL
    for (IdxT tileIter = dataLoader.nTiles; tileIter > 0; --tileIter) {
        dataLoader.Load(xSmem);
        __syncthreads();

        BlockScan<BLOCK, 1, CompT>(&scanner, xScanSmem);
        __syncthreads();

        dataLoader.Store(xSmem);
        __syncthreads();
    }

    // scan the last tile
    if (dataLoader.lastTileSize > 0) {
        dataLoader.Load(xSmem, dataLoader.lastTileSize);
        __syncthreads();

        BlockScan<BLOCK, 1, CompT>(&scanner, xScanSmem);
        __syncthreads();

        dataLoader.Store(xSmem, dataLoader.lastTileSize);
    }
}

template <int BLOCK,       // thread block size
          int TILE_M,      // tile height
          int TILE_N,      // tile width
          int UNROLL,      // number of items scaned by each thread
          bool EXCLUSIVE,  // true for exclusive mode
          bool REVERSE,    // true for reverse mode
          typename IdxT,   // type of offset
          typename CompT,  // compute precision
          typename DataT>  // type of input/output array
__global__ void Unified2DTilePrefixSumKernel(
        const DataT *x,    // input array pointer
        DataT *y,          // output array pointer
        IdxT m,            // number of rows of 2D array
        IdxT n) {          // number of columns of 2D array
    const uint32_t BLOCK_X = TILE_N / UNROLL;
    const uint32_t BLOCK_Y = BLOCK / BLOCK_X;
    static_assert(BLOCK_X <= WARP_SIZE,
                  "Unified2DTilePrefixSumKernel: invalid tile config");

    using XSmemT = CompT;
    using DataLoaderT = DataLoader<BLOCK, TILE_M, TILE_N, UNROLL, REVERSE,
                                   IdxT, XSmemT, DataT>;
    using ScannerT = Scanner<BLOCK_X, BLOCK_Y, UNROLL, EXCLUSIVE, CompT>;

    __shared__ XSmemT xSmem[DataLoaderT::SMEM_SIZE];

    DataLoaderT dataLoader(x, y, n, blockIdx.x, xSmem);
    ScannerT scanner;
    XSmemT *xSmemScanPtr = dataLoader.GetXSmemScanPtr(xSmem);

    if (m - blockIdx.x * TILE_M < TILE_M) {
        // full m-tile
        dataLoader.Load<true>(m, n);
        __syncthreads();

        scanner.ThreadReduce(xSmemScanPtr);
        scanner.WarpScan<false>(nullptr);
        scanner.UpdateGlobalPrefix(nullptr, xSmemScanPtr);
        __syncthreads();

        dataLoader.Store<true>(m, n);
    } else {
        dataLoader.Load<false>(m, n);
        __syncthreads();

        scanner.ThreadReduce(xSmemScanPtr);
        scanner.WarpScan<false>(nullptr);
        scanner.UpdateGlobalPrefix(nullptr, xSmemScanPtr);
        __syncthreads();

        dataLoader.Store<false>(m, n);
    }
}

template <bool EXCLUSIVE,
          bool REVERSE,
          typename IdxT,
          typename CompT,
          typename DataT>
hiednnStatus_t Unified1DTilePrefixSum(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        int64_t m,
        int64_t n) {
    void (*kernel)(const DataT *, DataT *, IdxT) = nullptr;
    int64_t grid = m;
    int64_t block;

    if (n > 256 * 9) {
        // BLOCK = 512, UNROLL = 9
        block = 512;
        kernel = Unified1DTilePrefixSumKernel<512, 9, EXCLUSIVE, REVERSE,
                                              IdxT, CompT, DataT>;
    } else if (n > 128 * 9) {
        // BLOCK = 256, UNROLL = 9
        block = 256;
        kernel = Unified1DTilePrefixSumKernel<256, 9, EXCLUSIVE, REVERSE,
                                              IdxT, CompT, DataT>;
    } else if (n > 128 * 5) {
        // BLOCK = 128, UNROLL = 9
        block = 128;
        kernel = Unified1DTilePrefixSumKernel<128, 9, EXCLUSIVE, REVERSE,
                                              IdxT, CompT, DataT>;
    } else {
        // BLOCK = 128, UNROLL = 5
        block = 128;
        kernel = Unified1DTilePrefixSumKernel<128, 5, EXCLUSIVE, REVERSE,
                                              IdxT, CompT, DataT>;
    }

    kernel<<<grid, block, 0, handle.stream>>>(x, y, n);

    return HIEDNN_STATUS_SUCCESS;
}

template <bool EXCLUSIVE,
          bool REVERSE,
          typename IdxT,
          typename CompT,
          typename DataT>
hiednnStatus_t Unified2DTilePrefixSum(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        int64_t m,
        int64_t n) {
    if (n > 512) {
        return HIEDNN_STATUS_INTERNAL_ERROR;
    }

    void (*kernel)(const DataT *, DataT *, IdxT, IdxT) = nullptr;
    int64_t grid;
    int64_t block;

    if (n > 256) {
        // BLOCK = 256, UNROLL = 16, TILE_N = 512, TILE_M = 8
        block = 256;
        grid = UIntDivRU<int64_t>(m, 8);
        kernel = Unified2DTilePrefixSumKernel<256, 8, 512, 16, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else if (n > 128) {
        // BLOCK = 256, UNROLL = 8, TILE_N = 256, TILE_M = 8
        block = 256;
        grid = UIntDivRU<int64_t>(m, 8);
        kernel = Unified2DTilePrefixSumKernel<256, 8, 256, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else if (n > 64) {
        // BLOCK = 256, UNROLL = 8, TILE_N = 128, TILE_M = 16
        block = 256;
        grid = UIntDivRU<int64_t>(m, 16);
        kernel = Unified2DTilePrefixSumKernel<256, 16, 128, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else if (n > 32) {
        // BLOCK = 256, UNROLL = 8, TILE_N = 64, TILE_M = 32
        block = 256;
        grid = UIntDivRU<int64_t>(m, 32);
        kernel = Unified2DTilePrefixSumKernel<256, 32, 64, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else if (n > 16) {
        // BLOCK = 128, UNROLL = 8, TILE_N = 32, TILE_M = 32
        block = 128;
        grid = UIntDivRU<int64_t>(m, 32);
        kernel = Unified2DTilePrefixSumKernel<128, 32, 32, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else if (n > 8) {
        // BLOCK = 128, UNROLL = 8, TILE_N = 16, TILE_M = 64
        block = 128;
        grid = UIntDivRU<int64_t>(m, 64);
        kernel = Unified2DTilePrefixSumKernel<128, 64, 16, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    } else {
        // BLOCK = 128, UNROLL = 8, TILE_N = 8, TILE_M = 128
        block = 128;
        grid = UIntDivRU<int64_t>(m, 128);
        kernel = Unified2DTilePrefixSumKernel<128, 128, 8, 8, EXCLUSIVE,
                                              REVERSE, IdxT, CompT>;
    }

    kernel<<<grid, block, 0, handle.stream>>>(x, y, m, n);

    return HIEDNN_STATUS_SUCCESS;
}

// scan contiguous data of 2D array
template <bool EXCLUSIVE,  // true for exclusive mode, false for includsive mode
          bool REVERSE,    // true for prefix scan, false for suffix scan
          typename CompT,  // compute precision
          typename DataT>  // input/output data type
hiednnStatus_t UnifiedPrefixSum(
        const HiednnCudaHandle &handle,  // hiednn CUDA handle
        const DataT *x,                  // pointer to input array
        DataT *y,                        // pointer to output array
        int64_t m,                       // number of rows of 2D array
        int64_t n) {                     // number of columns of 2D array
    // type of index or offset, associated with max supported array size
    using IdxT = uint32_t;

    if (n > 512) {
        return Unified1DTilePrefixSum<
                   EXCLUSIVE, REVERSE, IdxT, CompT>(
                   handle, x, y, m, n);
    } else {
        return Unified2DTilePrefixSum<
                   EXCLUSIVE, REVERSE, IdxT, CompT>(
                   handle, x, y, m, n);
    }
}

}  // namespace unified_scan_d1

// return true for tiled scan, false for unified scan
bool TiledScanD1(const HiednnCudaHandle &handle,
                 const int64_t m,
                 const int64_t n) {
    return m <= handle.deviceProp.multiProcessorCount * 8;
}

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_PREFIX_SCAN_UNIFIED_SCAN_D1_CUH_


