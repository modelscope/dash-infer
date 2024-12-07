/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scan_d0.cuh
 */

#ifndef DNN_CUDA_PREFIX_SCAN_SCAN_D0_CUH_
#define DNN_CUDA_PREFIX_SCAN_SCAN_D0_CUH_

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <utils.hpp>
#include <integer_divmod.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/prefix_sum.hpp>
#include <cuda/intrinsic/global_memory.hpp>
#include <cuda/intrinsic/warp_shuffle.hpp>

#include "scan_utils.cuh"

namespace hiednn {

namespace cuda {

namespace scan_d0 {

using prefix_sum::TileStatus;
using prefix_sum::TilePrefix;
using prefix_sum::WarpScanStep;

//-----------------------------------------------
// DataLoader
//-----------------------------------------------

// DataLoader for 3D tile
template <int BLOCK,         // thread block
          int TILE_BATCH,    // batch-dim of tile
          int TILE_M,        // m-dim of tile
          int TILE_N,        // n-dim of tile
          int THREAD_SCAN,   // sequence length scaned in 1 thread
          int THREAD_BATCH,  // number of sequences scaned in 1 thread
          bool REVERSE,      // true for reverse mode (suffix scan)
          typename IdxT,     // datatype of index and offset
          typename CompT,    // compute precision
          typename DataT>    // datatype of input/output tensor
struct DataLoader {
    static const int TILE_MN = TILE_M * TILE_N;

    static_assert(TILE_BATCH > 1 && IsPowOf2(TILE_N) &&
                  BLOCK % TILE_MN == 0 &&  // BLOCK >= TILE_MN
                  THREAD_SCAN % 2 == 0 &&
                  TILE_M % THREAD_SCAN == 0,
                  "scan_d0::DataLoader: invalid BLOCK or tile configuration");

    // m-, n-, batch-dim of thread block
    static const int TB_M = TILE_M / THREAD_SCAN;
    static const int TB_N = TILE_N;
    static const int TB_BATCH = BLOCK / (TB_M * TB_N);

    // LDG/STG unrolling factor
    static const int IO_UNROLL = THREAD_SCAN * THREAD_BATCH;

    const DataT *xLdgPtr;
    DataT       *yStgPtr;

    /*
     * shared memory dimension
     *
     * for TILE_N < WARP_SIZE, THREAD_SCAN is even, pad 1 element each
     * THREAD_SCAN to avoid smemScanPtr lds/sts bank conflict. this may
     * trigger smemIoPtr 2-way bank conflict for sizeof(CompT) >= 4,
     * but no impact on kernel performance.
     *
     * for example, TILE_N = 8, THREAD_SCAN = 2, shared memory map:
     *
     *    |<----------- TILE_N ---------->|
     *    |---|---|---|---|---|---|---|---|--
     *    |///|///|///|///|///|///|///|///| |
     *    |---|---|---|---|---|---|---|---| THREAD_SCAN
     *    |///|///|///|///|///|///|///|///| |
     *    |---|---|---|---|---|---|---|---|--
     *    |   |   |   |   |   |   |   |   | PADDING(column) = 1
     *    |---|---|---|---|---|---|---|---|--
     *    |///|///|///|///|///|///|///|///|
     *    |---|---|---|---|---|---|---|---|
     *    |///|///|///|///|///|///|///|///|
     *    |---|---|---|---|---|---|---|---|
     *    |   |   |   |   |   |   |   |   |
     *    |---|---|---|---|---|---|---|---|
     *    |           ......              |
     */
    static const int SMEM_M = TILE_N < WARP_SIZE ?
                              TB_M * (THREAD_SCAN + 1) : TB_M * THREAD_SCAN;
    static const int SMEM_N = TB_N;
    static const int SMEM_BATCH = TILE_BATCH;
    static const int SMEM_SIZE = SMEM_M * SMEM_N * SMEM_BATCH;
    CompT *smemIoPtr;
    CompT *smemScanPtr;

    IdxT batchIdx;
    bool fullBatchTile;
    bool mnGuard;

    /*
     * let
     * m_tiles = RoundUp(m / TILE_M)
     * n_tiles = RoundUp(n / TILE_N)
     *
     * param:
     * @tileNIdx: blockIdx.x % n_tiles;
     * @tileMIdx: blockIdx.x / n_tiles % m_tiles
     * @tileBatchIdx: blockIdx.x / n_tiles / m_tils
     */
    __device__ __forceinline__
    DataLoader(const DataT *x,
               DataT *y,
               CompT *smem,
               const IdxT &m,
               const IdxT &n,
               const IdxT &batch,
               const IdxT &tileBatchIdx,
               const IdxT &tileMIdx,
               const IdxT &tileNIdx) {
        int tidN = threadIdx.x % TILE_N;
        int tidM = BLOCK > TILE_MN ?
                   (threadIdx.x / TILE_N) % TILE_M :
                   (threadIdx.x / TILE_N);
        int tidBatch = BLOCK > TILE_MN ? threadIdx.x / TILE_MN : 0;

        batchIdx = tileBatchIdx * TILE_BATCH + tidBatch;
        IdxT mIdx = tileMIdx * TILE_M + tidM;
        IdxT nIdx = tileNIdx * TILE_N + tidN;

        IdxT batchMax = batchIdx + (IO_UNROLL - 1) * (BLOCK / TILE_MN);
        fullBatchTile = batchMax < batch;
        mnGuard = mIdx < m && nIdx < n;

        IdxT offset;
        if (REVERSE) {
            offset = (tileBatchIdx * TILE_BATCH + tidBatch) * m * n +
                     (m - 1 - mIdx) * n + nIdx;
        } else {
            offset = (tileBatchIdx * TILE_BATCH + tidBatch) * m * n +
                     mIdx * n + nIdx;
        }

        xLdgPtr = x + offset;
        yStgPtr = y + offset;

        if (TILE_N < WARP_SIZE) {
            // pad 1 element each THREAD_SCAN to avoid
            // smemScanPtr lds/sts bank conflict.
            // refer SMEM_M's comment for more information.
            smemIoPtr = smem + (threadIdx.x / TILE_N +
                        threadIdx.x / TILE_N / THREAD_SCAN) * SMEM_N + tidN;
            smemScanPtr = smem + (threadIdx.x / TILE_N) *
                          (THREAD_SCAN + 1) * TILE_N + tidN;
        } else {
            smemIoPtr = smem + threadIdx.x;
            smemScanPtr = smem + (threadIdx.x / TILE_N) * (TILE_N * THREAD_SCAN)
                          + tidN;
        }
    }

    __device__ __forceinline__
    void Load(CompT (&reg)[IO_UNROLL],
              const IdxT &m,
              const IdxT &n,
              const IdxT &batch) {
        // LDG
        DataT lReg[IO_UNROLL];
        if (mnGuard) {
            if (fullBatchTile) {
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    lReg[i] = xLdgPtr[(BLOCK / TILE_MN) * m * n * i];
                }
            } else {
                IdxT batchCount = batch > batchIdx ?
                    UIntDivRU<IdxT>(batch - batchIdx, BLOCK / TILE_MN) : 0;
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    lReg[i] = DataT(0);
                }
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    if (i < batchCount) {
                        lReg[i] = xLdgPtr[(BLOCK / TILE_MN) * m * n * i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                lReg[i] = 0;
            }
        }

        // STS
        #pragma unroll
        for (int i = 0; i < IO_UNROLL; ++i) {
            smemIoPtr[(BLOCK / TILE_MN) * SMEM_M * SMEM_N * i] =
                static_cast<CompT>(lReg[i]);
        }
        __syncthreads();

        // LDS
        #pragma unroll
        for (int i = 0; i < THREAD_BATCH; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_SCAN; ++j) {
                reg[i * THREAD_SCAN + j] =
                    smemScanPtr[i * SMEM_M * SMEM_N * TB_BATCH + j * SMEM_N];
            }
        }
    }

    __device__ __forceinline__
    void Store(CompT (&reg)[IO_UNROLL],
               const IdxT &m,
               const IdxT &n,
               const IdxT &batch) {
        // STS
        #pragma unroll
        for (int i = 0; i < THREAD_BATCH; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_SCAN; ++j) {
                smemScanPtr[i * SMEM_M * SMEM_N * TB_BATCH + j * SMEM_N] =
                    reg[i * THREAD_SCAN + j];
            }
        }
        __syncthreads();

        if (mnGuard) {
            // LDS
            DataT sReg[IO_UNROLL];
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                sReg[i] = static_cast<DataT>(
                    smemIoPtr[(BLOCK / TILE_MN) * SMEM_M * SMEM_N * i]);
            }

            // STG
            if (fullBatchTile) {
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    yStgPtr[(BLOCK / TILE_MN) * m * n * i] = sReg[i];
                }
            } else {
                IdxT batchCount = batch > batchIdx ?
                    UIntDivRU<IdxT>(batch - batchIdx, BLOCK / TILE_MN) : 0;
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    if (i < batchCount) {
                        yStgPtr[(BLOCK / TILE_MN) * m * n * i] = sReg[i];
                    }
                }
            }
        }
    }
};

// DataLoader for 2D tile
template <int BLOCK,
          int TILE_M,
          int TILE_N,
          int THREAD_SCAN,
          int THREAD_BATCH,
          bool REVERSE,
          typename IdxT,
          typename CompT,
          typename DataT>
struct DataLoader<BLOCK, 1, TILE_M, TILE_N, THREAD_SCAN, THREAD_BATCH,
                  REVERSE, IdxT, CompT, DataT> {
    static_assert(THREAD_BATCH == 1 &&
                  TILE_N % 2 == 0 &&
                  BLOCK % TILE_N == 0,  // BLOCK >= TILE_N
                  "scan_d0::DataLoader: invalid BLOCK or tile configuration");

    // to avoid shared memory bank conflicg,
    // THREAD_SCAN should be odd for TILE_N < WARP_SIZE
    static_assert(TILE_N >= WARP_SIZE || THREAD_SCAN % 2 == 1,
                  "scan_d0::DataLoader: invalid TILE_N, THREAD_SCAN");

    // LDG/STG unrolling factor
    static const int IO_UNROLL = THREAD_SCAN;

    const DataT *xLdgPtr;
    DataT       *yStgPtr;

    // SMEM_SIZE = 1 for BLOCK==TILE_N to avoid compilation error
    static const int SMEM_SIZE = BLOCK == TILE_N ? 1 : TILE_M * TILE_N;
    CompT *smemIoPtr;
    CompT *smemScanPtr;

    IdxT mIdx;
    bool fullMTile;
    bool nGuard;

    __device__ __forceinline__
    DataLoader(const DataT *x,
               DataT *y,
               CompT *smem,
               const IdxT &m,
               const IdxT &n,
               const IdxT &batch,
               const IdxT &tileBatchIdx,
               const IdxT &tileMIdx,
               const IdxT &tileNIdx) {
        int tidM = BLOCK > TILE_N ? threadIdx.x / TILE_N : 0;
        int tidN = BLOCK > TILE_N ? threadIdx.x % TILE_N : threadIdx.x;

        mIdx = tileMIdx * TILE_M + tidM;
        IdxT nIdx = tileNIdx * TILE_N + tidN;

        IdxT mMax = mIdx + (IO_UNROLL - 1) * (BLOCK / TILE_N);
        fullMTile = mMax < m;
        nGuard = nIdx < n;

        IdxT offset;
        if (REVERSE) {
            offset = tileBatchIdx * m * n + (m - 1 - mIdx) * n + nIdx;
        } else {
            offset = tileBatchIdx * m * n + mIdx * n + nIdx;
        }

        xLdgPtr = x + offset;
        yStgPtr = y + offset;

        smemIoPtr = smem + threadIdx.x;
        smemScanPtr = smem + tidM * (TILE_N * IO_UNROLL) + tidN;
    }

    __device__ __forceinline__
    void Load(CompT (&reg)[IO_UNROLL],
              const IdxT &m,
              const IdxT &n,
              const IdxT &batch) {
        // LDG
        DataT ldgReg[IO_UNROLL];
        if (nGuard) {
            if (fullMTile) {
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    if (REVERSE) {
                        ldgReg[i] = *(xLdgPtr - i * (BLOCK / TILE_N) * n);
                    } else {
                        ldgReg[i] = *(xLdgPtr + i * (BLOCK / TILE_N) * n);
                    }
                }
            } else {
                IdxT mCount = m > mIdx ?
                              UIntDivRU<IdxT>(m - mIdx, BLOCK / TILE_N) : 0;
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    ldgReg[i] = DataT(0);
                }
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    if (i < mCount) {
                        ldgReg[i] = REVERSE ?
                                    *(xLdgPtr - i * (BLOCK / TILE_N) * n) :
                                    *(xLdgPtr + i * (BLOCK / TILE_N) * n);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                ldgReg[i] = 0;
            }
        }

        if (BLOCK == TILE_N) {
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                reg[i] = static_cast<CompT>(ldgReg[i]);
            }
        } else {
            // STS
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                smemIoPtr[i * BLOCK] = static_cast<CompT>(ldgReg[i]);
            }
            __syncthreads();
            // LDS
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                reg[i] = smemScanPtr[i * TILE_N];
            }
        }
    }

    __device__ __forceinline__
    void StoreRegToGmem(DataT (&stgReg)[IO_UNROLL],
                        const IdxT &m,
                        const IdxT &n) {
        if (fullMTile) {
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                if (REVERSE) {
                    *(yStgPtr - i * (BLOCK / TILE_N) * n) = stgReg[i];
                } else {
                    *(yStgPtr + i * (BLOCK / TILE_N) * n) = stgReg[i];
                }
            }
        } else {
            IdxT mCount = m > mIdx ?
                          UIntDivRU<IdxT>(m - mIdx, BLOCK / TILE_N) : 0;
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                if (i < mCount) {
                    if (REVERSE) {
                        *(yStgPtr - i * (BLOCK / TILE_N) * n) = stgReg[i];
                    } else {
                        *(yStgPtr + i * (BLOCK / TILE_N) * n) = stgReg[i];
                    }
                }
            }
        }
    }

    __device__ __forceinline__
    void StoreViaSmem(CompT (&reg)[IO_UNROLL],
                      const IdxT &m,
                      const IdxT &n) {
        // STS
        #pragma unroll
        for (int i = 0; i < IO_UNROLL; ++i) {
            smemScanPtr[i * TILE_N] = reg[i];
        }
        __syncthreads();

        if (nGuard) {
            // LDS
            DataT stgReg[IO_UNROLL];
            #pragma unroll
            for (int i = 0; i < IO_UNROLL; ++i) {
                stgReg[i] = static_cast<DataT>(smemIoPtr[i * BLOCK]);
            }

            // STG
            StoreRegToGmem(stgReg, m, n);
        }
    }

    __device__ __forceinline__
    void Store(CompT (&reg)[IO_UNROLL],
               const IdxT &m,
               const IdxT &n,
               const IdxT &batch) {
        if (BLOCK == TILE_N) {
            if (nGuard) {
                DataT stgReg[IO_UNROLL];
                #pragma unroll
                for (int i = 0; i < IO_UNROLL; ++i) {
                    stgReg[i] = static_cast<DataT>(stgReg[i]);
                }
                StoreRegToGmem(stgReg, m, n);
            }
        } else {
            StoreViaSmem(reg, m, n);
        }
    }
};

//-----------------------------------------------
// Scanner
//-----------------------------------------------
template <int THREAD_SCAN,
          int THREAD_BATCH,
          bool EXCLUSIVE,
          typename CompT>
struct ScannerBase {
    CompT data[THREAD_SCAN * THREAD_BATCH];
    CompT threadAcc[THREAD_BATCH];

    __device__ __forceinline__ void ThreadScan() {
        #pragma unroll
        for (int i = 0; i < THREAD_BATCH; ++i) {
            threadAcc[i] = 0;
            #pragma unroll
            for (int j = 0; j < THREAD_SCAN; ++j) {
                if (EXCLUSIVE) {
                    CompT tmp = data[i * THREAD_SCAN + j];
                    data[i * THREAD_SCAN + j] = threadAcc[i];
                    threadAcc[i] += tmp;
                } else {
                    threadAcc[i] += data[i * THREAD_SCAN + j];
                    data[i * THREAD_SCAN + j] = threadAcc[i];
                }
            }
        }
    }
};

/*
 * Scanner only work for:
 *
 * for 3D tile (TILE_BATCH > 1):
 *     TILED_SCAN = false and
 *     THREAD_BATCH = 1, multi-thread_scan = [true|false] or
 *     THREAD_BATCH > 1, multi-thread_scan = false
 * for 2D tile (TILE_BATCH = 1):
 *     TILED_SCAN = [true|false] and
 *     THREAD_BATCH = 1, multi-thread_scan = [true|false]
 *
 * so the restrictions are:
 *
 * for THREAD_BATCH > 1:
 *     multi-thread_scan = false and
 *     TILED_SCAN = false.
 * for THREAD_BATCH = 1:
 *     multi-thread_scan = [true|false] and
 *     TILE_BATCH > 1, TILED_SCAN = false or
 *     TILE_BATCH = 1, TILED_SCAN = [true|false]
 *
 * comment:
 * multi-thread_scan:
 *     true for a sequence of tile was scanned by multi thread,
 *     false for the sequence was scanned by 1 thread.
 *     for multi-thread_scan = true, the output of ThreadScan()
 *     should be merged via shared memory or warp shuffle.
 */
template <int BLOCK,         // thread block
          int TILE_BATCH,    // batch-dim of tile
          int TILE_M,        // m-dim of tile
          int TILE_N,        // n-dim of tile
          int THREAD_SCAN,   // sequence length scaned in 1 thread
          int THREAD_BATCH,  // number of sequences scaned in 1 thread
          bool EXCLUSIVE,    // true for exclusive mode
          bool TILED_SCAN,   // true for tiled scan
          typename IdxT,     // datatype of index and offset
          typename CompT,    // compute precision
          typename TilePrefixT>
struct Scanner : public ScannerBase<THREAD_SCAN,
                                    THREAD_BATCH,
                                    EXCLUSIVE,
                                    CompT> {
    // Scanner for:
    //     THREAD_BATCH > 1 and
    //     multi-thread_scan = false and
    //     TILED_SCAN = false
    static_assert(
        THREAD_BATCH > 1 && TILE_M == THREAD_SCAN && TILED_SCAN == false,
        "scan_d0::Scanner: invalid tile configuration");

    __device__ __forceinline__ void Scan(
            const IdxT &tileBatchIdx,
            const IdxT &tileMIdx,
            const IdxT &tileNIdx,
            TilePrefixT *tilePrefix,
            const IdxT &tilePrefixMN,
            const IdxT &tilePrefixN) {
        ScannerBase<THREAD_SCAN, THREAD_BATCH, EXCLUSIVE, CompT>::ThreadScan();
    }
};

template <int BLOCK,
          int TILE_BATCH,
          int TILE_M,
          int TILE_N,
          int THREAD_SCAN,
          bool EXCLUSIVE,
          bool TILED_SCAN,
          typename IdxT,
          typename CompT,
          typename TilePrefixT>
struct Scanner<BLOCK, TILE_BATCH, TILE_M, TILE_N, THREAD_SCAN, 1,
               EXCLUSIVE, TILED_SCAN, IdxT, CompT, TilePrefixT>
        : public ScannerBase<THREAD_SCAN, 1, EXCLUSIVE, CompT> {
    // Scanner for
    //     THREAD_BATCH = 1 and
    //     multi-thread_scan = [true|false] and
    //     TILE_BATCH > 1, TILED_SCAN = false or
    //     TILE_BATCH = 1, TILED_SCAN = [true|false]
    static_assert(TILE_BATCH == 1 || TILED_SCAN == false,
                  "scan_d0::Scanner: invalid tile configuration");

    // batrch-, m-, n-dim of thread block
    static const int TB_BATCH = TILE_BATCH;
    static const int TB_M = TILE_M / THREAD_SCAN;
    static const int TB_N = TILE_N;

    using Base = ScannerBase<THREAD_SCAN, 1, EXCLUSIVE, CompT>;

    // input: tileAcc, sum of TILE_M sequence
    // output: tileAcc, exclusive prefix of the tile
    __device__ __forceinline__ void InterBlockScan(
            CompT *tileAcc,
            const IdxT &batchIdx,
            const IdxT &tileMIdx,
            const IdxT &nIdx,
            TilePrefixT *tilePrefix,
            const IdxT &tilePrefixMN,
            const IdxT &tilePrefixN) {
        TilePrefixT *tilePrefixLdgPtr =
            tilePrefix + batchIdx * tilePrefixMN +
            tileMIdx * tilePrefixN + nIdx;
        TilePrefixT *tilePrefixStgPtr = tilePrefixLdgPtr + tilePrefixN;

        // store tileAcc as TILE_SUM to global memory (L2 cache)
        TilePrefixT stgTileSum(TileStatus::TILE_SUM, *tileAcc);
        Stg<CG>(stgTileSum, tilePrefixStgPtr);

        TilePrefixT ldgPrefix;
        CompT tilePrefixAcc = 0;

        // inter-block prefix accumulate
        do {
            do {
                // make sure Ldg<CG> inside the while-loop
                __threadfence_block();
                Ldg<CG>(&ldgPrefix, tilePrefixLdgPtr);
            } while (ldgPrefix.stat == TileStatus::INVALID);

            tilePrefixAcc += ldgPrefix.data;
            tilePrefixLdgPtr -= tilePrefixN;
        } while (ldgPrefix.stat != TileStatus::TILE_PREFIX);

        // store inclusive tile prefix sum to global memory (L2 cache)
        TilePrefixT stgTilePrefix(TileStatus::TILE_PREFIX,
                                  *tileAcc + tilePrefixAcc);
        Stg<CG>(stgTilePrefix, tilePrefixStgPtr);

        *tileAcc = tilePrefixAcc;
    }

    // input: threadAcc[0], sum of THREAD_SCAN sequence
    // output: threadAcc[0], exclusive prefix of the thread
    __device__ __forceinline__ void SerialGlobalScan(
            const IdxT &tileBatchIdx,
            const IdxT &tileMIdx,
            const IdxT &tileNIdx,
            TilePrefixT *tilePrefix,
            const IdxT &tilePrefixMN,
            const IdxT &tilePrefixN) {
        __shared__ CompT smem[BLOCK];

        smem[threadIdx.x] = Base::threadAcc[0];
        __syncthreads();

        if (threadIdx.x < TB_N * TB_BATCH) {
            CompT tileAcc = 0;
            CompT threadPrefix[TB_M];
            CompT *smemPtr = smem + (threadIdx.x / TB_N) * (TB_M * TB_N) +
                              (threadIdx.x % TB_N);
            #pragma unroll
            for (int i = 0; i < TB_M; ++i) {
                threadPrefix[i] = smemPtr[i * TB_N];
            }

            if (TILED_SCAN) {
                #pragma unroll
                for (int i = 0; i < TB_M; ++i) {
                    tileAcc += threadPrefix[i];
                }

                IdxT batchIdx = TILE_BATCH > 1 ?
                    tileBatchIdx * TILE_BATCH + threadIdx.x / TB_N :
                    tileBatchIdx;
                IdxT nIdx = TILE_BATCH > 1 ?
                    tileNIdx * TILE_N + (threadIdx.x % TB_N) :
                    tileNIdx * TILE_N + threadIdx.x;
                InterBlockScan(&tileAcc, batchIdx, tileMIdx, nIdx,
                               tilePrefix, tilePrefixMN, tilePrefixN);
            }

            #pragma unroll
            for (int i = 0; i < TB_M; ++i) {
                smemPtr[i * TB_N] = tileAcc;
                tileAcc += threadPrefix[i];
            }
        }
        __syncthreads();

        // get thread prefix
        Base::threadAcc[0] = smem[threadIdx.x];
    }

    // ParallelGlobalScan restriction:
    // TB_M and TB_N should be power of 2,
    // TB_M * TB_N >= WARP_SIZE and TB_N < WARP_SIZE,
    __device__ __forceinline__ void ParallelGlobalScan(
            const IdxT &tileBatchIdx,
            const IdxT &tileMIdx,
            const IdxT &tileNIdx,
            TilePrefixT *tilePrefix,
            const IdxT &tilePrefixMN,
            const IdxT &tilePrefixN) {
        /*
         * SCAN_UNROLL: number of threadAcc scand by 1 thread
         *
         * to avoid all threads of thread block STS/LDS threadAcc bank
         * conflict, pad1 = WARP_SIZE / TB_N;
         * to avoid inter-thread scan LDS/STS threadAcc bank conflict,
         * pad2 = TB_M / SCAN_UNROLL;
         * pad1 = pad2, WARP_SIZE / TB_N = TB_M / SCAN_UNROLL,
         * SCAN_UNROLL = TB_M * TB_N / WARP_SIZE, 1 warp for each TB_M * TB_N
         * tile inter-thread scan.
         */
        const int SCAN_UNROLL = TB_M * TB_N / WARP_SIZE > 1 ?
                                TB_M * TB_N / WARP_SIZE : 1;

        /*
         * for all threads of thread block, threadAcc STS/LDS map:
         *
         * for example TB_N = 8, TB_M = 16, SCAN_UNROLL = 4:
         *
         *          |       ------------------------------------------
         *          |        ^   |t0 |t1 |t2 |t3 |t4 |t5 |t6 |t7 |  ^
         *          |        |   ---------------------------------  |
         *          |        |   |t8 |t9 |t10|t11|t12|t13|t14|t15|  |
         *          | SCAN_UNROLL---------------------------------  |
         *          |        |   |t16|t17|t18|t19|t20|t21|t22|t23|  |
         *   COLUMN |        |   ---------------------------------  |
         *   MAJOR  |        v   |t24|t25|t26|t27|t28|t29|t30|t31| TB_M + pad
         *          |       --------------------------------------  |
         *          |            |           padding             |  |
         *          |            ---------------------------------  |
         *          |            |t32|t33|t34|t35|t36|t37|t38|t39|  |
         *          |            ---------------------------------  |
         *          v            |            ......             |  v
         *                       |                               |----
         *
         * for sizeof(CompT) >= 4, to avoid smem bank conflict,
         * parameters should follow: TB_M * (TB_N ^ 2) >= WARP_SIZE ^ 2
         */
        const int SMEM_M = SCAN_UNROLL > 1 ? TB_M + TB_M / SCAN_UNROLL : TB_M;
        const int SMEM_N = TB_N;
        __shared__ CompT smem[SMEM_M * SMEM_N * TILE_BATCH];

        CompT *l1SmemPtr;

        if (TILE_BATCH > 1) {
            int nTid = threadIdx.x % TB_N;
            int mTid = threadIdx.x / TB_N % TB_M;
            int batchTid = threadIdx.x / TB_N / TB_M;
            l1SmemPtr = smem + batchTid * SMEM_M * SMEM_N + nTid * SMEM_M +
                        (SCAN_UNROLL > 1 ? mTid + mTid / SCAN_UNROLL : mTid);
        } else {
            int nTid = threadIdx.x % TB_N;
            int mTid = threadIdx.x / TB_N;
            l1SmemPtr = smem + nTid * SMEM_M +
                        (SCAN_UNROLL > 1 ? mTid + mTid / SCAN_UNROLL : mTid);
        }

        *l1SmemPtr = Base::threadAcc[0];
        __syncthreads();

        int warpId = threadIdx.x / WARP_SIZE;
        int laneId = threadIdx.x % WARP_SIZE;
        if (warpId < TILE_BATCH) {
            const int SCAN_THREAD_M = TB_M / SCAN_UNROLL > 1 ?
                                      TB_M / SCAN_UNROLL : 1;

            // serial scan (l1 scan)
            CompT *scanL1SmemPtr = smem +
                threadIdx.x / SCAN_THREAD_M * SMEM_M +
                threadIdx.x % SCAN_THREAD_M *
                (SCAN_UNROLL > 1 ? SCAN_UNROLL + 1 : SCAN_UNROLL);
            CompT l1Prefix[SCAN_UNROLL];
            CompT l1Acc = 0;
            #pragma unroll
            for (int i = 0; i < SCAN_UNROLL; ++i) {
                l1Prefix[i] = scanL1SmemPtr[i];
                l1Acc += l1Prefix[i];
            }

            // parallel warp shuffle scan (l2 scan)
            CompT l2Prefix = l1Acc;
            #pragma unroll
            for (int i = 1; i < SCAN_THREAD_M; i *= 2) {
                l1Acc = WarpScanStep(0xffffffff, l1Acc, i, SCAN_THREAD_M);
            }
            l2Prefix = l1Acc - l2Prefix;  // exclusive prefix

            if (TILED_SCAN) {
                // inter-block scan (l3 scan)
                l1Acc = ShflIdx(0xffffffff,
                                l1Acc,
                                laneId * SCAN_THREAD_M + (SCAN_THREAD_M - 1),
                                WARP_SIZE);

                if (laneId < TB_N) {
                    IdxT batchIdx = TILE_BATCH > 1 ?
                        tileBatchIdx * TILE_BATCH + warpId : tileBatchIdx;
                    IdxT nIdx = tileNIdx * TILE_N + laneId;
                    InterBlockScan(&l1Acc, batchIdx, tileMIdx, nIdx,
                                   tilePrefix, tilePrefixMN, tilePrefixN);
                }

                l2Prefix += ShflIdx(0xffffffff,
                                    l1Acc,
                                    laneId / SCAN_THREAD_M,
                                    WARP_SIZE);
            }

            #pragma unroll
            for (int i = 0; i < SCAN_UNROLL; ++i) {
                scanL1SmemPtr[i] = l2Prefix;
                l2Prefix += l1Prefix[i];
            }
        }
        __syncthreads();

        Base::threadAcc[0] = *l1SmemPtr;
    }

    __device__ __forceinline__ void ThreadScanUpdate() {
        for (int i = 0; i < THREAD_SCAN; ++i) {
            Base::data[i] += Base::threadAcc[0];
        }
    }

    __device__ __forceinline__ void Scan(
            const IdxT &tileBatchIdx,
            const IdxT &tileMIdx,
            const IdxT &tileNIdx,
            TilePrefixT *tilePrefix,
            const IdxT &tilePrefixMN,
            const IdxT &tilePrefixN) {
        Base::ThreadScan();

        if (TILED_SCAN || TILE_M > THREAD_SCAN) {
            if (TB_N < WARP_SIZE && (TB_M * TB_N) >= WARP_SIZE && TB_M > 8) {
                ParallelGlobalScan(tileBatchIdx, tileMIdx, tileNIdx,
                                   tilePrefix, tilePrefixMN, tilePrefixN);
            } else {
                SerialGlobalScan(tileBatchIdx, tileMIdx, tileNIdx,
                                 tilePrefix, tilePrefixMN, tilePrefixN);
            }

            ThreadScanUpdate();
        }
    }
};

//-----------------------------------------------
// prefix scan kernel
//-----------------------------------------------
template <int BLOCK,         // thread block
          int TILE_BATCH,    // batch-dim of tile
          int TILE_M,        // m-dim of tile
          int TILE_N,        // n-dim of tile
          int THREAD_SCAN,   // sequence length scaned in 1 thread
          int THREAD_BATCH,  // number of sequences scaned in 1 thread
          bool REVERSE,      // true for reverse mode (suffix scan)
          bool EXCLUSIVE,    // true for exclusive mode
          bool TILED_SCAN,   // true for tiled scan
          typename IdxT,     // datatype of index and offset
          typename CompT,    // compute precision
          typename DataT,    // datatype of input/output tensor
          typename TilePrefixT>
__global__ void PrefixSumKernel(
        const DataT *x,
        DataT *y,
        TilePrefixT *tilePrefix,
        IdxT m,
        IdxT n,
        IdxT batch,
        IdxT tilePrefixMN,
        IdxT tilePrefixN,
        U32DivMod mTileDivMod,
        U32DivMod nTileDivMod) {
    static_assert(BLOCK % WARP_SIZE == 0 &&
                  BLOCK * THREAD_SCAN * THREAD_BATCH ==
                  TILE_BATCH * TILE_M * TILE_N,
                  "scan_d0::PrefixSumKernel: invalid tile configuration");

    using DataLoaderT = DataLoader<BLOCK, TILE_BATCH, TILE_M, TILE_N,
                                   THREAD_SCAN, THREAD_BATCH, REVERSE,
                                   IdxT, CompT, DataT>;
    using ScannerT = Scanner<BLOCK, TILE_BATCH, TILE_M, TILE_N,
                             THREAD_SCAN, THREAD_BATCH, EXCLUSIVE,
                             TILED_SCAN, IdxT, CompT, TilePrefixT>;

    __shared__ CompT dataSmem[DataLoaderT::SMEM_SIZE];

    auto nTileDm = nTileDivMod.DivMod(blockIdx.x);
    auto mTileDm = mTileDivMod.DivMod(nTileDm.div);
    IdxT tileBatchIdx = mTileDm.div;
    IdxT tileMIdx = mTileDm.mod;
    IdxT tileNIdx = nTileDm.mod;

    DataLoaderT dataLoader(
        x, y, dataSmem, m, n, batch, tileBatchIdx, tileMIdx, tileNIdx);
    ScannerT scanner;

    dataLoader.Load(scanner.data, m, n, batch);
    scanner.Scan(tileBatchIdx, tileMIdx, tileNIdx,
                 tilePrefix, tilePrefixMN, tilePrefixN);
    dataLoader.Store(scanner.data, m, n, batch);
}

//-----------------------------------------------
// tile prefix initialization
//-----------------------------------------------
template <int BLOCK, int UNROLL, typename TilePrefixT>
__global__ void TilePrefixInitKernel(
        TilePrefixT *tilePrefix,
        uint32_t tilePrefixN,
        uint32_t tilePrefixSize,
        U32DivMod tilePrefixMNDivMod) {
    uint32_t idx = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    TilePrefixT stgVal[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        stgVal[i] = tilePrefixMNDivMod.Mod(idx + i * BLOCK) < tilePrefixN ?
                    TilePrefixT(TileStatus::TILE_PREFIX, 0) :
                    TilePrefixT(TileStatus::INVALID, 0);
    }

    uint32_t loopCount = tilePrefixSize > idx ?
                         UIntDivRU<uint32_t>(tilePrefixSize - idx, BLOCK) : 0;
    if (UNROLL >= 8 && loopCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            tilePrefix[idx + i * BLOCK] = stgVal[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < loopCount) {
                tilePrefix[idx + i * BLOCK] = stgVal[i];
            }
        }
    }
}

template <typename TilePrefixT>
void TilePrefixInit(TilePrefixT *tilePrefix,
                    uint32_t tilePrefixM,
                    uint32_t tilePrefixN,
                    uint32_t tilePrefixBatch,
                    cudaStream_t stream) {
    static_assert(sizeof(TilePrefixT) <= 16,
                  "scan_d0::TilePrefixInit: invalid TilePrefixT");

    const uint32_t BLOCK = 128;
    const uint32_t UNROLL = 16 / sizeof(TilePrefixT);
    uint32_t tilePrefixSize = tilePrefixM * tilePrefixN * tilePrefixBatch;
    U32DivMod tilePrefixMNDivMod(tilePrefixM * tilePrefixN);
    uint32_t grid = UIntDivRU(tilePrefixSize, BLOCK * UNROLL);

    TilePrefixInitKernel<BLOCK, UNROLL, TilePrefixT>
        <<<grid, BLOCK, 0, stream>>>(
        tilePrefix, tilePrefixN, tilePrefixSize, tilePrefixMNDivMod);
}

//-----------------------------------------------
// kernel launcher
//-----------------------------------------------
template <int BLOCK,
          int TILE_BATCH,
          int TILE_M,
          int TILE_N,
          int THREAD_SCAN,
          int THREAD_BATCH,
          bool REVERSE,
          bool EXCLUSIVE,
          typename IdxT,
          typename CompT,
          typename DataT,
          typename TilePrefixT>
hiednnStatus_t LaunchUnifiedPrefixSumKernel(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        IdxT m,
        IdxT n,
        IdxT batch) {
    IdxT gridX = UIntDivRU<IdxT>(n, TILE_N);
    IdxT gridY = UIntDivRU<IdxT>(m, TILE_M);
    IdxT gridZ = UIntDivRU<IdxT>(batch, TILE_BATCH);
    IdxT grid = gridX * gridY * gridZ;

    U32DivMod mTileDivMod(gridY);
    U32DivMod nTileDivMod(gridX);

    PrefixSumKernel<BLOCK, TILE_BATCH, TILE_M, TILE_N, THREAD_SCAN,
                    THREAD_BATCH, REVERSE, EXCLUSIVE, false,
                    IdxT, CompT, DataT, TilePrefixT>
                   <<<grid, BLOCK, 0, handle.stream>>>(
                   x, y, nullptr, m, n, batch, 0, 0, mTileDivMod, nTileDivMod);

    return HIEDNN_STATUS_SUCCESS;
}

template <int BLOCK,
          int TILE_BATCH,
          int TILE_M,
          int TILE_N,
          int THREAD_SCAN,
          int THREAD_BATCH,
          bool REVERSE,
          bool EXCLUSIVE,
          typename IdxT,
          typename CompT,
          typename DataT,
          typename TilePrefixT>
hiednnStatus_t LaunchTiledPrefixSumKernel(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        IdxT m,
        IdxT n,
        IdxT batch) {
    IdxT gridX = UIntDivRU<IdxT>(n, TILE_N);
    IdxT gridY = UIntDivRU<IdxT>(m, TILE_M);
    IdxT gridZ = UIntDivRU<IdxT>(batch, TILE_BATCH);
    IdxT grid = gridX * gridY * gridZ;

    U32DivMod mTileDivMod(gridY);
    U32DivMod nTileDivMod(gridX);

    // at least 32-byte aligned
    size_t tilePrefixAlign = 32 / sizeof(TilePrefixT);
    size_t tilePrefixN = UIntDivRU<size_t>(gridX * TILE_N, tilePrefixAlign) *
                         tilePrefixAlign;
    size_t tilePrefixM = gridY + 1;
    size_t tilePrefixMN = tilePrefixM * tilePrefixN;
    size_t tilePrefixBatch = UIntDivRU<size_t>(batch, TILE_BATCH) * TILE_BATCH;
    size_t tilePrefixSize = tilePrefixMN * tilePrefixBatch;

    TilePrefixT *tilePrefix;
    DeviceWsGuard wsGuard(handle);
    wsGuard.GetWorkspace(&tilePrefix, tilePrefixSize * sizeof(TilePrefixT));
    if (tilePrefix == nullptr) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    TilePrefixInit<TilePrefixT>(tilePrefix, tilePrefixM, tilePrefixN,
                                  tilePrefixBatch, handle.stream);

    PrefixSumKernel<BLOCK, TILE_BATCH, TILE_M, TILE_N, THREAD_SCAN,
                    THREAD_BATCH, REVERSE, EXCLUSIVE, true,
                    IdxT, CompT, DataT, TilePrefixT>
                   <<<grid, BLOCK, 0, handle.stream>>>(
                   x, y, tilePrefix, m, n, batch, tilePrefixMN, tilePrefixN,
                   mTileDivMod, nTileDivMod);

    return HIEDNN_STATUS_SUCCESS;
}

//-----------------------------------------------
// tile dispatch
//-----------------------------------------------
template <bool EXCLUSIVE,
          bool REVERSE,
          typename IdxT,
          typename CompT,
          typename DataT,
          typename TilePrefixT>
hiednnStatus_t UnifiedPrefixSum(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        IdxT m,
        IdxT n,
        IdxT batch) {
    hiednnStatus_t ret = HIEDNN_STATUS_TENSOR_OVERSIZE;
    const int BLOCK = 512;  // thread block

    if (n <= 2) {
        const int TILE_N = 2;

        if (m <= 4) {
            // TILE_M = 4, TILE_BATCH = 512, THREAD_SCAN = 4, THREAD_BATCH = 2
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 512, 4, TILE_N, 4, 2, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // TILE_M = 8, TILE_BATCH = 256, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 256, 8, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // TILE_M = 16, TILE_BATCH = 128, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 128, 16, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // TILE_M = 32, TILE_BATCH = 64, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 64, 32, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 64) {
            // TILE_M = 64, TILE_BATCH = 32, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 32, 64, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 128) {
            // TILE_M = 128, TILE_BATCH = 16, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 16, 128, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 256) {
            // TILE_M = 256, TILE_BATCH = 8, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 8, 256, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 448) {
            // BLOCK = 128, TILE_M = 448, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 448, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 896) {
            // BLOCK = 256, TILE_M = 896, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 896, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    } else if (n <= 4) {
        const int TILE_N = 4;

        if (m <= 4) {
            // TILE_M = 4, TILE_BATCH = 256, THREAD_SCAN = 4, THREAD_BATCH = 2
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 256, 4, TILE_N, 4, 2, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // TILE_M = 8, TILE_BATCH = 128, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 128, 8, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // TILE_M = 16, TILE_BATCH = 64, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 64, 16, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // TILE_M = 32, TILE_BATCH = 32, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 32, 32, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 64) {
            // TILE_M = 64, TILE_BATCH = 16, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 16, 64, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 128) {
            // TILE_M = 128, TILE_BATCH = 8, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 8, 128, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 224) {
            // BLOCK = 128, TILE_M = 224, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 224, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 448) {
            // BLOCK = 256, TILE_M = 448, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 448, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    } else if (n <= 8) {
        const int TILE_N = 8;

        if (m <= 4) {
            // TILE_M = 4, TILE_BATCH = 128, THREAD_SCAN = 4, THREAD_BATCH = 2
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 128, 4, TILE_N, 4, 2, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // TILE_M = 8, TILE_BATCH = 64, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 64, 8, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // TILE_M = 16, TILE_BATCH = 32, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 32, 16, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // TILE_M = 32, TILE_BATCH = 16, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 16, 32, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 64) {
            // TILE_M = 64, TILE_BATCH = 8, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 8, 64, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 112) {
            // BLOCK = 128, TILE_M = 112, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 112, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 224) {
            // BLOCK = 256, TILE_M = 224, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 224, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    } else if (n <= 16) {
        const int TILE_N = 16;

        if (m <= 4) {
            // TILE_M = 4, TILE_BATCH = 64, THREAD_SCAN = 4, THREAD_BATCH = 2
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 64, 4, TILE_N, 4, 2, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // TILE_M = 8, TILE_BATCH = 32, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 32, 8, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // TILE_M = 16, TILE_BATCH = 16, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 16, 16, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // TILE_M = 32, TILE_BATCH = 8, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 8, 32, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 56) {
            // BLOCK = 128, TILE_M = 56, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 56, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 112) {
            // BLOCK = 256, TILE_M = 112, TILE_BATCH = 1, THREAD_SCAN = 7
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 112, TILE_N, 7, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    } else if (n < 256) {
        const int TILE_N = 32;

        if (m <= 4) {
            // TILE_M = 4, TILE_BATCH = 32, THREAD_SCAN = 4, THREAD_BATCH = 2
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 32, 4, TILE_N, 4, 2, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // TILE_M = 8, TILE_BATCH = 16, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 16, 8, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // TILE_M = 16, TILE_BATCH = 8, THREAD_SCAN = 8, THREAD_BATCH = 1
            ret = LaunchUnifiedPrefixSumKernel<
                      BLOCK, 8, 16, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // BLOCK = 128, TILE_M = 32, TILE_BATCH = 1, THREAD_SCAN = 8
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 32, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 64) {
            // BLOCK = 256, TILE_M = 64, TILE_BATCH = 1, THREAD_SCAN = 8
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 64, TILE_N, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    } else {
        if (m <= 4) {
            // BLOCK = 128, TILE_BATCH = 1, TILE_M = 4, TILE_N = 128,
            // THREAD_SCAN = 4, TILE_BATCH = 1,
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 4, 128, 4, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 8) {
            // BLOCK = 128, TILE_BATCH = 1, TILE_M = 8, TILE_N = 128,
            // THREAD_SCAN = 8, TILE_BATCH = 1,
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 8, 128, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 16) {
            // BLOCK = 128, TILE_BATCH = 1, TILE_M = 8, TILE_N = 64,
            // THREAD_SCAN = 8, TILE_BATCH = 1,
            ret = LaunchUnifiedPrefixSumKernel<
                      128, 1, 16, 64, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 32) {
            // BLOCK = 256, TILE_BATCH = 1, TILE_M = 32, TILE_N = 64,
            // THREAD_SCAN = 8, TILE_BATCH = 1,
            ret = LaunchUnifiedPrefixSumKernel<
                      256, 1, 32, 64, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        } else if (m <= 64) {
            // BLOCK = 512, TILE_BATCH = 1, TILE_M = 64, TILE_N = 64,
            // THREAD_SCAN = 8, TILE_BATCH = 1,
            ret = LaunchUnifiedPrefixSumKernel<
                      512, 1, 64, 64, 8, 1, REVERSE, EXCLUSIVE, IdxT,
                      CompT, DataT, TilePrefixT>(handle, x, y, m, n, batch);
        }
    }

    return ret;
}

template <bool EXCLUSIVE,
          bool REVERSE,
          typename IdxT,
          typename CompT,
          typename DataT,
          typename TilePrefixT>
hiednnStatus_t TiledPrefixSum(
        const HiednnCudaHandle &handle,
        const DataT *x,
        DataT *y,
        IdxT m,
        IdxT n,
        IdxT batch) {
    // tiled scan only work for TILE_BATCH = 1, THREAD_BATCH = 1

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;

    if (n <= 2) {
        // BLOCK = 256, TILE_M = 1152, TILE_N = 2, THREAD_SCAN = 9
        ret = LaunchTiledPrefixSumKernel<256, 1, 1152, 2, 9, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    } else if (n <= 4) {
        // BLOCK = 256, TILE_M = 576, TILE_N = 4, THREAD_SCAN = 9
        ret = LaunchTiledPrefixSumKernel<256, 1, 576, 4, 9, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    } else if (n <= 8) {
        // BLOCK = 256, TILE_M = 288, TILE_N = 8, THREAD_SCAN = 9
        ret = LaunchTiledPrefixSumKernel<256, 1, 288, 8, 9, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    } else if (n <= 16) {
        // BLOCK = 256, TILE_M = 144, TILE_N = 16, THREAD_SCAN = 9
        ret = LaunchTiledPrefixSumKernel<256, 1, 144, 16, 9, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    } else if (n < 128) {
        // BLOCK = 256, TILE_M = 64, TILE_N = 32, THREAD_SCAN = 8
        ret = LaunchTiledPrefixSumKernel<256, 1, 64, 32, 8, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    } else {
        // BLOCK = 512, TILE_M = 64, TILE_N = 64, THREAD_SCAN = 8
        ret = LaunchTiledPrefixSumKernel<512, 1, 64, 64, 8, 1,
            REVERSE, EXCLUSIVE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    }

    return ret;
}

//-----------------------------------------------
// non-contiguous dimension prefix scan interface
//-----------------------------------------------
template <bool EXCLUSIVE,  // true for exclusive mode, false for includsive mode
          bool REVERSE,    // true for prefix scan, false for suffix scan
          typename CompT,  // compute precision
          typename DataT>  // input/output data type
hiednnStatus_t PrefixSum(
        const HiednnCudaHandle &handle,
        const DataT *x,   // pointer to input array
        DataT *y,         // pointer to output array
        int64_t m,        // number of rows of input array
        int64_t n,        // number of columns of input array
        int64_t batch) {  // number of 2D array
    // type of index or offset, associated with max supported array size
    using IdxT = uint32_t;
    using TilePrefixT = TilePrefix<CompT>;

    // try unified prefix scan
    auto ret = UnifiedPrefixSum<
        EXCLUSIVE, REVERSE, IdxT, CompT, DataT, TilePrefixT>(
        handle, x, y, m, n, batch);

    if (ret == HIEDNN_STATUS_TENSOR_OVERSIZE) {
        ret = TiledPrefixSum<
            EXCLUSIVE, REVERSE, IdxT, CompT, DataT, TilePrefixT>(
            handle, x, y, m, n, batch);
    }

    return ret;
}

}  // namespace scan_d0

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_PREFIX_SCAN_SCAN_D0_CUH_


