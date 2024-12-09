/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    row_reduce.cuh
 */

#ifndef DNN_CUDA_REDUCE_ROW_REDUCE_CUH_
#define DNN_CUDA_REDUCE_ROW_REDUCE_CUH_

#include <utils.hpp>
#include <integer_divmod.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/warp_shuffle.hpp>

#include "reduce_utils.cuh"

namespace hiednn {

namespace cuda {

namespace reduce {

template <int WIDTH, typename ReduceFunc, typename ReduceT>
__device__ __forceinline__
ReduceT RowWarpReduce(const ReduceT &x) {
    static_assert(IsPowOf2(WIDTH), "RowWarpReduce: invalid WIDTH");

    ReduceT ret = x;
    #pragma unroll
    for (int i = WIDTH; i > 1; i /= 2) {
        ReduceT y = ShflBfly(0xffffffff, ret, i / 2, i);
        ret = ReduceFunc()(ret, y);
    }
    return ret;
}

// shuffle output data for STG coherence
template <int TB_M, int TB_N, typename ReduceT>
__device__ __forceinline__
ReduceT RowReduceShuffle(const ReduceT &x,
                         const int &warpId,
                         const int &laneId) {
    static_assert(IsPowOf2(TB_N), "RowReduceShuffle: invalid TB_N");

    __shared__ ReduceT smem[TB_M];

    ReduceT ret = TB_N < WARP_SIZE ?
                  ShflIdx(0xffffffff, x, laneId * TB_N, WARP_SIZE) : x;

    if (laneId < WARP_SIZE / TB_N) {
        smem[warpId * WARP_SIZE / TB_N + laneId] = ret;
    }

    __syncthreads();

    if (threadIdx.x < TB_M) {
        ret = smem[threadIdx.x];
    }
    return ret;
}

// inter-warp reduce and shuffle output data for STG coherence
template <int TB_M, int TB_N, typename ReduceFunc, typename ReduceT>
__device__ __forceinline__
ReduceT RowBlockReduce(const ReduceT &x,
                       const int &warpId,
                       const int &laneId) {
    const int WARPS = TB_M * TB_N / WARP_SIZE;
    static_assert(IsPowOf2(WARPS) && IsPowOf2(TB_N / WARP_SIZE),
                  "RowBlockReduce: invalid thread block configuration");

    __shared__ ReduceT smem[WARPS];

    if (laneId == 0) {
        smem[warpId] = x;
    }

    __syncthreads();

    ReduceT ret;
    if (threadIdx.x < WARPS) {
        uint32_t mask = (1lu << WARPS) - 1;
        ret = smem[threadIdx.x];
        #pragma unroll
        for (int i = TB_N / WARP_SIZE; i > 1; i /= 2) {
            ReduceT y = ShflBfly(mask, ret, i / 2, i);
            ret = ReduceFunc()(ret, y);
        }

        if (TB_M > 1) {
            ret = ShflIdx(mask, ret, threadIdx.x * (TB_N / WARP_SIZE), WARPS);
        }
    }

    return ret;
}

template <int TB_M,          // m-dim of logic thread block
          int TB_N,          // n-dim of logic thread block
          int UNROLL,        // n-dim unrolling factor
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
__global__ void RowReduceKernel(
        const ST *x,
        DT *y,
        IdxT *indices,
        ReduceT *ws,
        DT alpha,
        OffsetT m,
        OffsetT n,
        OffsetT wsN,
        U32DivMod nBlockDivMod) {
    static_assert(IsPowOf2(TB_N) && (TB_M * TB_N) % WARP_SIZE == 0,
                  "RowReduceKernel: invalid thread block configuration");

    auto nBlockDm = nBlockDivMod.DivMod(blockIdx.x);
    uint32_t blockIdM = MULTI_BLOCK ? nBlockDm.div : blockIdx.x;
    uint32_t blockIdN = MULTI_BLOCK ? nBlockDm.mod : 0;
    uint32_t threadIdM = TB_M == 1 ? 0 : threadIdx.x / TB_N;
    uint32_t threadIdN = TB_M == 1 ? threadIdx.x : threadIdx.x % TB_N;

    OffsetT idxM = static_cast<OffsetT>(blockIdM) * TB_M + threadIdM;
    OffsetT idxN = static_cast<OffsetT>(blockIdN) * TB_N * UNROLL + threadIdN;

    if (TB_M > 1 && idxM >= m) return;

    // serial reduce
    ThreadReducer<MULTI_BLOCK, UNROLL, TB_N, PreFunc, ReduceFunc,
                  ReduceInitFunc, MathT, ReduceT, OffsetT, IdxT, WITH_IDX>
                 threadReducer;

    const ST *xPtr = x + idxM * n + idxN;
    bool fullTile = blockIdN * TB_N * UNROLL + TB_N * UNROLL <= n;
    ReduceT ret = threadReducer.Reduce(xPtr, idxN, TB_N, n, fullTile);

    // warp reduce
    if (TB_N > 1) {
        const int WIDTH = TB_N < WARP_SIZE ? TB_N : WARP_SIZE;
        ret = RowWarpReduce<WIDTH, ReduceFunc>(ret);
    }

    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // block reduce (inter-warp reduce)
    if (TB_N > WARP_SIZE) {
        ret = RowBlockReduce<TB_M, TB_N, ReduceFunc>(ret, warpId, laneId);
    } else if (TB_N > 1) {
        ret = RowReduceShuffle<TB_M, TB_N>(ret, warpId, laneId);
    }

    // write back
    idxM = TB_M == 1 ? blockIdM * TB_M : blockIdM * TB_M + threadIdx.x;

    if (threadIdx.x < TB_M && idxM < m) {
        OffsetT offset = USE_WS ? idxM * wsN + blockIdN : idxM;
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
struct SingleBlockRowReduce {
    template <int TB_M, int TB_N, int UNROLL>
    static hiednnStatus_t LaunchKernel(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        int64_t grid = UIntDivRU<int64_t>(m, TB_M);

        RowReduceKernel<TB_M, TB_N, UNROLL, RET_IDX, false, false,
                        RET_DATA, RET_IDX, PreFunc, ReduceFunc, PostFunc,
                        ReduceFuncInit<ReduceFunc, MathT>,
                        ST, DT, MathT, ReduceT, OffsetT, IdxT>
            <<<grid, TB_M * TB_N, 0, handle.stream>>>(
            x, y, indices, nullptr, alpha, m, n, 0, U32DivMod(1));

        return HIEDNN_STATUS_SUCCESS;
    }

    static hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        if (n >= 2048) {
            return LaunchKernel<1, 256, 8>(
                handle, m, n, alpha, x, y, indices);
        } else if (n >= 512) {
            return LaunchKernel<4, 64, 8>(
                handle, m, n, alpha, x, y, indices);
        } else if (n >= 128) {
            return LaunchKernel<16, 16, 8>(
                handle, m, n, alpha, x, y, indices);
        } else if (n >= 32) {
            return LaunchKernel<64, 4, 8>(
                handle, m, n, alpha, x, y, indices);
        } else if (n >= 8) {
            return LaunchKernel<128, 2, 4>(
                handle, m, n, alpha, x, y, indices);
        } else {
            return LaunchKernel<256, 1, 8>(
                handle, m, n, alpha, x, y, indices);
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
struct MultiBlockRowReduce {
    static hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &m,
            const int64_t &n,
            const DT &alpha,
            const ST *x,
            DT *y,
            IdxT *indices) {
        const int BLOCK = 256;
        const int UNROLL = 16;

        int64_t gridN = UIntDivRU<int64_t>(n, BLOCK * UNROLL);
        int64_t gridM = m;

        int64_t wsSize = gridN * gridM * sizeof(ReduceT);
        ReduceT *ws;
        DeviceWsGuard wsGuard(handle);
        if (wsGuard.GetWorkspace(&ws, wsSize) != HIEDNN_STATUS_SUCCESS) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        RowReduceKernel<1, BLOCK, UNROLL, RET_IDX, true, true, true, false,
                        PreFunc, ReduceFunc, scalar_functor::Pass<ReduceT>,
                        ReduceFuncInit<ReduceFunc, MathT>,
                        ST, DT, MathT, ReduceT, OffsetT, IdxT>
            <<<gridN * gridM, BLOCK, 0, handle.stream>>>(
            x, y, indices, ws, alpha, m, n, gridN, U32DivMod(gridN));

        RowReduceKernel<1, 512, 8, false, false, false, RET_DATA, RET_IDX,
                        scalar_functor::Pass<ReduceT>, ReduceFunc, PostFunc,
                        ReduceFuncInit<ReduceFunc, MathT>,
                        ReduceT, DT, ReduceT, ReduceT, OffsetT, IdxT>
            <<<gridM, 512, 0, handle.stream>>>(
            ws, y, indices, nullptr, alpha, m, gridN, 0, U32DivMod(1));

        return HIEDNN_STATUS_SUCCESS;
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
hiednnStatus_t RowReduce(
        const HiednnCudaHandle &handle,
        const int64_t &m,
        const int64_t &n,
        const DT &alpha,
        const ST *x,
        DT *y,
        IdxT *indices = nullptr) {
    hiednnStatus_t ret = HIEDNN_STATUS_TENSOR_OVERSIZE;

    if (n >= 4096) {
        ret = MultiBlockRowReduce<PreFunc, ReduceFunc, PostFunc, ST, DT, MathT,
                                  ReduceT, OffsetT, IdxT, RET_DATA, RET_IDX>
              ::Reduce(handle, m, n, alpha, x, y, indices);
    }

    if (ret == HIEDNN_STATUS_TENSOR_OVERSIZE) {
        ret = SingleBlockRowReduce<PreFunc, ReduceFunc, PostFunc, ST, DT, MathT,
                                   ReduceT, OffsetT, IdxT, RET_DATA, RET_IDX>
              ::Reduce(handle, m, n, alpha, x, y, indices);
    }

    return ret;
}

}  // namespace reduce

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_REDUCE_ROW_REDUCE_CUH_


