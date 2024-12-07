/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tiled_scan_d1.cuh
 */

#ifndef DNN_CUDA_PREFIX_SCAN_TILED_SCAN_D1_CUH_
#define DNN_CUDA_PREFIX_SCAN_TILED_SCAN_D1_CUH_

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <utils.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/prefix_sum.hpp>
#include <cuda/intrinsic/global_memory.hpp>

#include "scan_utils.cuh"

namespace hiednn {

namespace cuda {

namespace tiled_scan_d1 {

using prefix_sum::TileStatus;
using prefix_sum::TilePrefix;
using prefix_sum::ScanMode;
using prefix_sum::GlobalPrefixSum;

// ------------------------------------------------------------
// TilePrefix init
// ------------------------------------------------------------

// BLOCK: thrad block size
template <int NDIM, int BLOCK, int PADDING, typename TilePrefixT>
__global__ void TilePrefixInitKernel(
        TilePrefixT *tilePrefix,
        int lineSize,
        int padding) {
    int lineIdx = blockIdx.x * BLOCK + threadIdx.x;

    if (lineIdx < lineSize) {
        TilePrefixT stgVal = lineIdx < (PADDING < 0 ? padding : PADDING) ?
                             TilePrefixT(TileStatus::TILE_PREFIX, 0) :
                             TilePrefixT(TileStatus::INVALID, 0);

        TilePrefixT *stgPtr = NDIM == 2 ?
                              tilePrefix + blockIdx.y * lineSize + lineIdx:
                              tilePrefix + lineIdx;
        *stgPtr = stgVal;
    }
}

/*
 * initialize tilePrefix (workspace) before prefix sum kernel launch.
 * PADDING: fixed padding size, set PADDING to -1 for dynamic padding
 * NDIM: dimension of tilePrefix, can only be 1 or 2
 *
 * @lineSize: length of line (include padding)
 */
template <int NDIM, int PADDING, typename TilePrefixT>
void TilePrefixInit(cudaStream_t stream,
                    TilePrefixT *tilePrefix,
                    int lineSize,
                    int nLines = 1,
                    int padding = 0) {
    static_assert(NDIM == 1 || NDIM == 2, "TilePrefixInit: invalid NDIM");

    const int BLOCK = 128;

    int gridX = UIntDivRU(lineSize, BLOCK);
    int gridY = NDIM == 2 ? nLines : 1;
    dim3 grid(gridX, gridY);

    TilePrefixInitKernel<NDIM, BLOCK, PADDING, TilePrefixT>
        <<<grid, BLOCK, 0, stream>>>(tilePrefix, lineSize, padding);
}

// ------------------------------------------------------------
// Tiled PrefixScanD1
// ------------------------------------------------------------

template <int TILE_SIZE, bool REVERSE, typename IdxT, typename DataT>
__device__ __forceinline__
void Init1DTile(const DataT *x,
                DataT *y,
                const IdxT &n,
                const uint32_t &nTiles,
                const uint32_t &tileId,
                uint32_t *tileSize,
                const DataT **xTilePtr,
                DataT **yTilePtr) {
    if (REVERSE) {
        if (tileId == 0) {
            *tileSize = n - (nTiles - 1) * TILE_SIZE;
            *xTilePtr = x + n - 1;
            *yTilePtr = y + n - 1;
        } else {
            *tileSize = TILE_SIZE;
            *xTilePtr = x + (nTiles - tileId) * TILE_SIZE - 1;
            *yTilePtr = y + (nTiles - tileId) * TILE_SIZE - 1;
        }
    } else {
        *tileSize = n - tileId * TILE_SIZE < TILE_SIZE ?
                    n - tileId * TILE_SIZE : TILE_SIZE;
        *xTilePtr = x + tileId * TILE_SIZE;
        *yTilePtr = y + tileId * TILE_SIZE;
    }
}

template <int TILE_SIZE, bool REVERSE, typename IdxT, typename DataT>
__device__ __forceinline__
void Init2DTile(const DataT *x,
                DataT *y,
                const IdxT &n,
                const uint32_t &tileId,
                const uint32_t &rowId,
                uint32_t *tileSize,
                const DataT **xTilePtr,
                DataT **yTilePtr) {
    const uint32_t DATA_ALIGN = MEM_ALIGN_BYTE / sizeof(DataT);
    static_assert(TILE_SIZE % DATA_ALIGN == 0,
                  "tiled_scan_d1::Init2DTile: invalid TILE_SIZE");

    /*
     * tileGlobalOffset:
     *     offset of the 1'st item of the tile, for REVERSE mode,
     *     1'st item indicate the last item of the tile in memory
     */
    IdxT rowOffset = (REVERSE ? rowId + 1 : rowId) * n;
    IdxT tileGlobalOffset = REVERSE ? rowOffset - 1 : rowOffset;

    uint32_t firstTileSize = TILE_SIZE - (REVERSE ?
         (DATA_ALIGN - (rowOffset % DATA_ALIGN)) % DATA_ALIGN :
         rowOffset % DATA_ALIGN);

    if (tileId > 0) {
        IdxT tileOffset = firstTileSize + (tileId - 1) * TILE_SIZE;
        *tileSize = tileOffset >= n ? 0 :
            n - tileOffset < TILE_SIZE ?
            n - tileOffset : TILE_SIZE;
        tileGlobalOffset = REVERSE ?
            tileGlobalOffset - tileOffset :
            tileGlobalOffset + tileOffset;
    } else {
        // set first tile
        *tileSize = firstTileSize > n ? n : firstTileSize;
    }

    *xTilePtr = x + tileGlobalOffset;
    *yTilePtr = y + tileGlobalOffset;
}

// data loader for scan contiguous data
template <int DIM,         // 1 for 1D array, 2 for 2D array
          int BLOCK,       // thread block size
          int UNROLL,      // items loaded each thread
          bool REVERSE,    // true for reverse mode (suffix scan)
          typename IdxT,   // type of index or offset
          typename DataT>  // type of input/output array
struct DataLoader {
    // only work for 1D or 2D array
    static_assert(DIM == 1 || DIM == 2,
                  "tiled_scan_d1::DataLoader: invalid DIM");

    static const uint32_t TILE_SIZE = BLOCK * UNROLL;

    /*
     * data alignment must not more than TILE_SIZE,
     * or shared memory ld/st will be out of bound
     */
    static_assert(MEM_ALIGN_BYTE / sizeof(DataT) <= TILE_SIZE,
                  "tiled_scan_d1::DataLoader: invalid TILE_SIZE");

    uint32_t tileSize;

    /*
     * xTilePtr, yTilePtr:
     *     tile pointer, point to the first item of tile,
     *     for reverse mode, point to the last item of tile in memory
     */
    const DataT *xTilePtr;
    DataT       *yTilePtr;

    __device__ __forceinline__
    DataLoader(const DataT *x,
               DataT *y,
               const IdxT &n,
               const uint32_t &nTiles,
               const uint32_t &tileId,
               const uint32_t &rowId) {
        if (DIM == 1) {
            Init1DTile<TILE_SIZE, REVERSE, IdxT, DataT>(
                x, y, n, nTiles, tileId, &tileSize, &xTilePtr, &yTilePtr);
        } else {
            Init2DTile<TILE_SIZE, REVERSE, IdxT, DataT>(
                x, y, n, tileId, rowId, &tileSize, &xTilePtr, &yTilePtr);
        }
    }

    /*
     * Load() brief:
     *
     * for UNROLL is 2 and BLOCK is 4:
     *
     * x:      |x0 |x1 |x2 |x3 |x4 |x5 |x6 |x7 |
     * thread: |t0 |t1 |t2 |t3 |t0 |t1 |t2 |t3 |
     * thread 0: xReg = {x0, x4}
     * thread 1: xReg = {x1, x5}
     *
     * for reverse mode, the xReg is not reversed:
     * x:      |x0 |x1 |x2 |x3 |x4 |x5 |x6 |x7 |
     * thread: |t3 |t2 |t1 |t0 |t3 |t2 |t1 |t0 |
     * thread 0: xReg = {x7, x3}
     * thread 1: xReg = {x6, x2}
     */
    template <typename RegT>
    __device__ __forceinline__
    void Load(RegT (&xReg)[UNROLL]) const {
        const DataT *xLdgPtr = REVERSE ? xTilePtr - threadIdx.x :
                                         xTilePtr + threadIdx.x;
        DataT ldgReg[UNROLL];

        if (tileSize < TILE_SIZE) {
            uint32_t idx = threadIdx.x;
            uint32_t ldgCount = tileSize > idx ?
                                UIntDivRU<uint32_t>(tileSize - idx, BLOCK) : 0;
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                ldgReg[i] = DataT(0);
            }
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                if (i < ldgCount) {
                    Ldg<NC, LTCMAX>(&ldgReg[i], REVERSE ? xLdgPtr - i * BLOCK :
                                                          xLdgPtr + i * BLOCK);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                Ldg<NC, LTCMAX>(&ldgReg[i], REVERSE ? xLdgPtr - i * BLOCK :
                                                      xLdgPtr + i * BLOCK);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xReg[i] = static_cast<RegT>(ldgReg[i]);
        }
    }

    template <typename SmemT>
    __device__ __forceinline__
    void Store(const SmemT *ySmem) const {
        DataT *yStgPtr = REVERSE ?
                          yTilePtr - threadIdx.x :
                          yTilePtr + threadIdx.x;
        SmemT reg[UNROLL];

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            reg[i] = ySmem[threadIdx.x + i * BLOCK];
        }

        if (tileSize < TILE_SIZE) {
            uint32_t idx = threadIdx.x;
            uint32_t stgCount = tileSize > idx ?
                                UIntDivRU<uint32_t>(tileSize - idx, BLOCK) : 0;
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                if (i < stgCount) {
                    yStgPtr[(REVERSE ? -i : i) * BLOCK] = reg[i];
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                yStgPtr[(REVERSE ? -i : i) * BLOCK] = reg[i];
            }
        }
    }
};

template <int DIM,                  // 1 for 1D array, 2 for 2D array
          int BLOCK,                // thread block size
          int UNROLL,               // number of items scaned by each thread
          bool EXCLUSIVE,           // true for exclusive mode
          bool REVERSE,             // true for reverse mode
          typename IdxT,            // type of offset
          typename CompT,           // compute precision
          int TILE_PREFIX_PADDING,  // tilePrefix padding
          typename DataT,           // type of input/output array
          typename TilePrefixT>
__global__ void TiledPrefixSumKernel(
        const DataT *x,             // input array pointer
        DataT *y,                   // output array pointer
        TilePrefixT *tilePrefix,    // workspace for inter-tile scan
        IdxT n,                     // size of 1D array,
                                    // or number of columns for 2D array
        uint32_t tilePrefixSize) {  // line size of tilePrefix (include padding)
    __shared__ CompT xSmem[BLOCK * UNROLL];

    using DataLoaderT = DataLoader<DIM, BLOCK, UNROLL, REVERSE, IdxT, DataT>;
    DataLoaderT tileLoader(x, y, n, gridDim.x, blockIdx.x, blockIdx.y);

    if (DIM == 2 && tileLoader.tileSize == 0) {
        return;
    }

    // register buffer
    CompT xReg[UNROLL];

    // load xTile from gmem to register
    tileLoader.Load(xReg);

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xSmem[threadIdx.x + i * BLOCK] = xReg[i];
    }

    __syncthreads();

    // thread reduce
    CompT threadAcc = 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xReg[i] = xSmem[threadIdx.x * UNROLL + i];
        threadAcc += xReg[i];
    }

    // global scan
    uint32_t warpId = threadIdx.x / WARP_SIZE;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    TilePrefixT *tilePrefixPtr = DIM == 1 ?
                                 tilePrefix :
                                 tilePrefix + blockIdx.y * tilePrefixSize;
    CompT threadExclusivePrefix = GlobalPrefixSum<ScanMode::EXCLUSIVE, BLOCK>(
        threadAcc, blockIdx.x, warpId, laneId, tilePrefixPtr);

    // update the global prefix sum
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (EXCLUSIVE) {
            xSmem[threadIdx.x * UNROLL + i] = threadExclusivePrefix;
            threadExclusivePrefix += xReg[i];
        } else {
            threadExclusivePrefix += xReg[i];
            xSmem[threadIdx.x * UNROLL + i] = threadExclusivePrefix;
        }
    }

    __syncthreads();

    // writeback
    tileLoader.Store(xSmem);
}

// scan contiguous data of 2D array
template <bool EXCLUSIVE,  // true for exclusive mode, false for includsive mode
          bool REVERSE,    // true for prefix scan, false for suffix scan
          typename CompT,  // compute precision
          typename DataT>  // input/output data type
hiednnStatus_t TiledPrefixSum(
        const HiednnCudaHandle &handle,  // hiednn CUDA handle
        const DataT *x,                  // pointer to input array
        DataT *y,                        // pointer to output array
        int64_t m,                       // number of rows of input array
        int64_t n) {                     // number of columns of input array
    // type of index or offset, associated with max supported array size
    using IdxT = uint32_t;

    using TilePrefixT = TilePrefix<CompT>;

    // number of items scanned by a thread
    // UNROLL must be an odd number to avoid shared memory bank conflict
    const int64_t UNROLL = 15;
    // thread block size
    const int64_t BLOCK = 128;
    // tile size
    const int64_t TILE_SIZE = UNROLL * BLOCK;

    const int TILE_PREFIX_PADDING = prefix_sum::TILE_PREFIX_PADDING;

    int64_t gridX = UIntDivRU(n, TILE_SIZE);

    if (m == 1) {
        // scan 1D array
        int64_t tilePrefixSize = gridX + TILE_PREFIX_PADDING;

        // tensor oversize
        if (gridX > handle.deviceProp.maxGridSize[0]) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        TilePrefixT *tilePrefix;
        DeviceWsGuard wsGuard(handle);
        wsGuard.GetWorkspace(&tilePrefix, tilePrefixSize * sizeof(TilePrefixT));
        if (tilePrefix == nullptr) {
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }

        TilePrefixInit<1, TILE_PREFIX_PADDING, TilePrefixT>(
            handle.stream, tilePrefix, tilePrefixSize);

        TiledPrefixSumKernel<1, BLOCK, UNROLL, EXCLUSIVE, REVERSE,
                             IdxT, CompT, TILE_PREFIX_PADDING>
                             <<<gridX, BLOCK, 0, handle.stream>>>(
                             x, y, tilePrefix, n, tilePrefixSize);
    } else {
        // scan 2D array
        int64_t gridY = m;

        if (gridX + 1 > handle.deviceProp.maxGridSize[0] ||
            gridY > handle.deviceProp.maxGridSize[1]) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        dim3 grid(gridX + 1, gridY);

        // each line of tilePrefix is MEM_ALIGN_BYTE aligned,
        // to maximum the inter-threadblock scan performance
        int64_t tilePrefixAlign = MEM_ALIGN_BYTE / sizeof(TilePrefixT);
        int64_t tilePrefixSizeX = tilePrefixAlign *
            UIntDivRU((gridX + TILE_PREFIX_PADDING), tilePrefixAlign);
        int64_t tilePrefixSizeY = m;
        int64_t tilePrefixSize = tilePrefixSizeX * tilePrefixSizeY;

        TilePrefixT *tilePrefix;
        DeviceWsGuard wsGuard(handle);
        wsGuard.GetWorkspace(&tilePrefix, tilePrefixSize * sizeof(TilePrefixT));
        if (tilePrefix == nullptr) {
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }

        TilePrefixInit<2, TILE_PREFIX_PADDING, TilePrefixT>(
            handle.stream, tilePrefix, tilePrefixSizeX, m);

        TiledPrefixSumKernel<2, BLOCK, UNROLL, EXCLUSIVE, REVERSE,
                             IdxT, CompT, TILE_PREFIX_PADDING>
                             <<<grid, BLOCK, 0, handle.stream>>>(
                             x, y, tilePrefix, n, tilePrefixSizeX);
    }

    return HIEDNN_STATUS_SUCCESS;
}

}  // namespace tiled_scan_d1

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_PREFIX_SCAN_TILED_SCAN_D1_CUH_


