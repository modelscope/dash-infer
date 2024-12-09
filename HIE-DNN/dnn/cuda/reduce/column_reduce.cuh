/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    column_reduce.cuh
 */

#ifndef DNN_CUDA_REDUCE_COLUMN_REDUCE_CUH_
#define DNN_CUDA_REDUCE_COLUMN_REDUCE_CUH_

#include <utils.hpp>
#include <integer_divmod.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/warp_shuffle.hpp>

#include "reduce_utils.cuh"

namespace hiednn {

namespace cuda {

namespace reduce {

template <int TB_N, typename ReduceFunc, typename ReduceT>
__device__ __forceinline__
ReduceT ColumnWarpReduce(const ReduceT &x) {
    ReduceT ret = x;
    #pragma unroll
    for (int i = WARP_SIZE; i > TB_N; i /= 2) {
        ReduceT y = ShflBfly(0xffffffff, ret, i / 2, i);
        ret = ReduceFunc()(ret, y);
    }
    return ret;
}

template <int TB_M, int TB_N, typename ReduceFunc, typename ReduceT>
__device__ __forceinline__
ReduceT ColumnBlockReduce(const ReduceT &x,
                          const uint32_t &warpId,
                          const uint32_t &laneId,
                          const uint32_t &threadIdM,
                          const uint32_t &threadIdN) {
    const int SMEM_M = TB_N < WARP_SIZE ? TB_M / (WARP_SIZE / TB_N) : TB_M;
    const int SMEM_N = TB_N + 1;  // append 1 item to avoid bank conflict

    __shared__ ReduceT smem[SMEM_M][SMEM_N];
    __shared__ ReduceT retSmem[SMEM_N];

    ReduceT ret = x;

    if (TB_N < WARP_SIZE) {
        if (laneId < TB_N) { smem[warpId][laneId] = ret; }
    } else {
        smem[threadIdM][threadIdN] = ret;
    }

    __syncthreads();

    const int SHFL_WIDTH = SMEM_M < WARP_SIZE ? SMEM_M : WARP_SIZE;

    if (threadIdx.x < SHFL_WIDTH * TB_N) {
        int ldsIdxM = SHFL_WIDTH < WARP_SIZE ?
                      threadIdx.x % SHFL_WIDTH : laneId;
        int ldsIdxN = SHFL_WIDTH < WARP_SIZE ?
                      threadIdx.x / SHFL_WIDTH : warpId;
        ReduceT acc = smem[ldsIdxM][ldsIdxN];

        #pragma unroll
        for (int i = 1; i < SMEM_M / SHFL_WIDTH; ++i) {
            ReduceT tmp = smem[ldsIdxM + i * SHFL_WIDTH][ldsIdxN];
            acc = ReduceFunc()(acc, tmp);
        }

        #pragma unroll
        for (int i = SHFL_WIDTH; i > 1; i /= 2) {
            uint32_t mask = SHFL_WIDTH * TB_N >= 32 ?
                            0xffffffff : (1lu << SHFL_WIDTH * TB_N) - 1;
            ReduceT tmp = ShflBfly(mask, acc, i / 2, i);
            acc = ReduceFunc()(acc, tmp);
        }

        if (ldsIdxM == 0) { retSmem[ldsIdxN] = acc; }
    }

    __syncthreads();

    if (threadIdx.x < TB_N) { ret = retSmem[threadIdx.x]; }
    return ret;
}

template <int TB_M,          // m-dim of logic thread block
          int TB_N,          // n-dim of logic thread block
          int UNROLL,        // m-dim unrolling factor
          bool WITH_IDX,     // reduce with index
          bool MULTI_BLOCK,  // @n dim is reduced by multi thread blocks,
                             // and thread block reduce fixed tile size
          bool USE_WS,       // store output data to @ws
          bool RET_DATA,
          bool RET_IDX,
          typename PreFunc,
          typename ReduceFunc,
          typename PostFunc,
          typename ReduceInitFunc,
          typename ST,
          typename DT,
          typename MathT,    // precision of PreFunc operator
          typename ReduceT,  // precision of ReduceFunc operator
          typename OffsetT,  // type of offset in kernel
          typename IdxT>     // type of returned indices
__global__ void ColumnReduceKernel(
        const ST *x,
        DT *y,
        IdxT *indices,
        ReduceT *ws,
        DT alpha,
        OffsetT m,
        OffsetT n,
        OffsetT xBatchStride,
        OffsetT wsBatchStride,
        U32DivMod mBlockDivMod,
        U32DivMod nBlockDivMod) {
    static_assert(IsPowOf2(TB_M) &&
                  IsPowOf2(TB_N) &&
                  (TB_M * TB_N) % WARP_SIZE == 0,
        "ColumnReduceKernel: invalid thread block configuration");

    // blockId of batch-, m-, n- dim
    uint32_t blockIdB, blockIdM, blockIdN;
    auto nBlockDm = nBlockDivMod.DivMod(blockIdx.x);
    blockIdN = nBlockDm.mod;
    if (MULTI_BLOCK) {
        auto mBlockDm = mBlockDivMod.DivMod(nBlockDm.div);
        blockIdM = mBlockDm.mod;
        blockIdB = mBlockDm.div;
    } else {
        blockIdM = 0;
        blockIdB = nBlockDm.div;
    }

    uint32_t threadIdM = TB_M == 1 ? 0 : threadIdx.x / TB_N;
    uint32_t threadIdN = TB_M == 1 ? threadIdx.x : threadIdx.x % TB_N;

    OffsetT idxB = static_cast<OffsetT>(blockIdB);
    OffsetT idxM = static_cast<OffsetT>(blockIdM) * TB_M * UNROLL + threadIdM;
    OffsetT idxN = static_cast<OffsetT>(blockIdN) * TB_N + threadIdN;

    ReduceT ret;

    // serial reduce
    ThreadReducer<MULTI_BLOCK, UNROLL, TB_M, PreFunc, ReduceFunc,
                  ReduceInitFunc, MathT, ReduceT, OffsetT, IdxT, WITH_IDX>
                 threadReducer;

    if (idxN < n) {
        const ST *xPtr = x + idxB * xBatchStride + idxM * n + idxN;
        bool fullTile = blockIdM * TB_M * UNROLL + TB_M * UNROLL <= m;
        ret = threadReducer.Reduce(xPtr, idxM, TB_M * n, m, fullTile);
    }

    // warp reduce
    if (TB_N < WARP_SIZE) {
        ret = ColumnWarpReduce<TB_N, ReduceFunc>(ret);
    }

    uint32_t warpId = threadIdx.x / WARP_SIZE;
    uint32_t laneId = threadIdx.x % WARP_SIZE;

    // block reduce (inter-warp reduce)
    if (TB_M > 1 && TB_M > WARP_SIZE / TB_N) {
        ret = ColumnBlockReduce<TB_M, TB_N, ReduceFunc, ReduceT>(
                  ret, warpId, laneId, threadIdM, threadIdN);
    }

    // write back
    if (threadIdM == 0 && idxN < n) {
        OffsetT offset = USE_WS ?
                         idxB * wsBatchStride + blockIdM * n + idxN :
                         idxB * n + idxN;
        WriteBack<USE_WS, RET_DATA, RET_IDX, PostFunc>::
            Store(ret, ws, y, indices, alpha, offset);
    }
}

template <typename PreFunc,
          typename ReduceFunc,
          typename PostFunc,
          typename ST,
          typename DT,
          typename MathT,
          typename ReduceT,
          typename OffsetT,
          typename IdxT,
          bool RET_DATA,
          bool RET_IDX>
struct SingleBlockColumnReduce {
    template <int TB_M, int TB_N, int UNROLL>
    static hiednnStatus_t LaunchKernel(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const int64_t &batch,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        int64_t gridB = batch;
        int64_t gridN = UIntDivRU<int64_t>(n, TB_N);

        ColumnReduceKernel<TB_M, TB_N, UNROLL, RET_IDX, false, false,
                           RET_DATA, RET_IDX, PreFunc, ReduceFunc, PostFunc,
                           ReduceFuncInit<ReduceFunc, MathT>,
                           ST, DT, MathT, ReduceT, OffsetT, IdxT>
            <<<gridB * gridN, TB_M * TB_N, 0, handle.stream>>>(
            x, y, indices, nullptr, alpha, m, n, m * n, 0,
            U32DivMod(1), U32DivMod(gridN));

        return HIEDNN_STATUS_SUCCESS;
    }

    static hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const int64_t &batch,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        if (n >= 256) {
            return LaunchKernel<2, 128, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 128) {
            return LaunchKernel<2, 64, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 64) {
            return LaunchKernel<4, 32, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 32) {
            return LaunchKernel<8, 16, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 16) {
            return LaunchKernel<16, 8, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        } else {
            return LaunchKernel<32, 4, 8>(
                handle, m, n, batch, alpha, x, y, indices);
        }
    }
};

template <typename PreFunc,
          typename ReduceFunc,
          typename PostFunc,
          typename ST,
          typename DT,
          typename MathT,
          typename ReduceT,
          typename OffsetT,
          typename IdxT,
          bool RET_DATA,
          bool RET_IDX>
struct MultiBlockColumnReduce {
    template <int TB_M, int TB_N, int UNROLL>
    static hiednnStatus_t LaunchKernel(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const int64_t &batch,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        int64_t gridM = UIntDivRU<int64_t>(m, TB_M * UNROLL);
        int64_t gridN = UIntDivRU<int64_t>(n, TB_N);
        int64_t gridB = batch;

        int64_t wsSize = gridB * gridM * n * sizeof(ReduceT);
        ReduceT *ws;
        DeviceWsGuard wsGuard(handle);
        if (wsGuard.GetWorkspace(&ws, wsSize) != HIEDNN_STATUS_SUCCESS) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        ColumnReduceKernel<TB_M, TB_N, UNROLL, RET_IDX, true, true, true, false,
                           PreFunc, ReduceFunc, scalar_functor::Pass<ReduceT>,
                           ReduceFuncInit<ReduceFunc, MathT>,
                           ST, DT, MathT, ReduceT, OffsetT, IdxT>
            <<<gridB * gridM * gridN, TB_M * TB_N, 0, handle.stream>>>(
            x, y, indices, ws, alpha, m, n, m * n, gridM * n,
            U32DivMod(gridM), U32DivMod(gridN));

        int64_t globalRedGridN = UIntDivRU<int64_t>(n, 32);

        ColumnReduceKernel<4, 32, 8, false, false, false, RET_DATA, RET_IDX,
                           scalar_functor::Pass<ReduceT>, ReduceFunc, PostFunc,
                           ReduceFuncInit<ReduceFunc, MathT>,
                           ReduceT, DT, ReduceT, ReduceT, OffsetT, IdxT>
            <<<batch * globalRedGridN, 4 * 32, 0, handle.stream>>>(
            ws, y, indices, nullptr, alpha, gridM, n, gridM * n, 0,
            U32DivMod(1), U32DivMod(globalRedGridN));

        return HIEDNN_STATUS_SUCCESS;
    }

    static hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const int64_t &batch,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        if (n >= 256) {
            return LaunchKernel<4, 128, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 128) {
            return LaunchKernel<4, 64, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 64) {
            return LaunchKernel<8, 32, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 32) {
            return LaunchKernel<16, 16, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 16) {
            return LaunchKernel<32, 8, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else if (n >= 8) {
            return LaunchKernel<64, 4, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        } else {
            return LaunchKernel<128, 2, 16>(
                handle, m, n, batch, alpha, x, y, indices);
        }
    }
};

template <typename PreFunc,
          typename ReduceFunc,
          typename PostFunc,
          typename ST,
          typename DT,
          typename MathT,       // precision of PreFunc operator
          typename ReduceT,     // precision of ReduceFunc operator
          typename OffsetT,     // type of offset in kernel
          typename IdxT = int,  // type of returned indices
          bool RET_DATA = true,
          bool RET_IDX = false>
hiednnStatus_t ColumnReduce(
        const HiednnCudaHandle &handle,
        const int64_t &m,
        const int64_t &n,
        const int64_t &batch,
        const DT &alpha,
        const ST *x,
        DT *y,
        IdxT *indices = nullptr) {
    hiednnStatus_t ret = HIEDNN_STATUS_TENSOR_OVERSIZE;

    if (m >= 1024 || m * n >= 8192) {
        ret = MultiBlockColumnReduce<
                  PreFunc, ReduceFunc, PostFunc, ST, DT,
                  MathT, ReduceT, OffsetT, IdxT, RET_DATA, RET_IDX>
              ::Reduce(handle, m, n, batch, alpha, x, y, indices);
    }

    if (ret == HIEDNN_STATUS_TENSOR_OVERSIZE) {
        ret = SingleBlockColumnReduce<
                  PreFunc, ReduceFunc, PostFunc, ST, DT,
                  MathT, ReduceT, OffsetT, IdxT, RET_DATA, RET_IDX>
              ::Reduce(handle, m, n, batch, alpha, x, y, indices);
    }

    return ret;
}

}  // namespace reduce

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_REDUCE_COLUMN_REDUCE_CUH_


