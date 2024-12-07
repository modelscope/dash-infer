/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    non_zero.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <tensor_desc.hpp>
#include <utils.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/prefix_sum.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, bool FAST>
struct PostProc;

template <int BLOCK, int UNROLL>
struct PostProc<BLOCK, UNROLL, true> {
    // nzMask bits of out-of-bound items should be 0
    template <typename NzMaskT,
              typename PrefixSumT,
              typename OffsetComputeT,
              typename OffsetOutputT,
              typename OutT>
    __device__ __forceinline__
    void operator()(OutT *y,
                    OffsetOutputT *offsets,
                    const NzMaskT &nzMask,
                    const PrefixSumT (&nzOffsets)[UNROLL],
                    const OffsetComputeT &offset0,
                    uint32_t xSize,
                    const uint32_t &nDims,
                    const Array<U32DivMod, TENSOR_DIM_MAX> &strideDivMods) {
        uint32_t offset[UNROLL];
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            offset[i] = offset0 + i * BLOCK;
        }

        OutT *yPtr = y;
        #pragma unroll
        for (uint32_t dim = 0; dim < TENSOR_DIM_MAX; ++dim) {
            if (dim >= nDims) {
                break;
            }

            OutT indices[UNROLL];
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                auto divmod = strideDivMods[dim].DivMod(offset[i]);
                offset[i] = divmod.mod;
                indices[i] = divmod.div;
            }

            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                if ((nzMask & (NzMaskT(1) << i)) != 0) {
                    Stg(indices[i], yPtr + nzOffsets[i]);
                }
            }
            yPtr += xSize;
        }
    }
};

template <int BLOCK, int UNROLL>
struct PostProc<BLOCK, UNROLL, false> {
    // nzMask bits of out-of-bound items should be 0
    template <typename NzMaskT,
              typename PrefixSumT,
              typename OffsetComputeT,
              typename OffsetOutputT,
              typename OutT,
              class... Args>
    __device__ __forceinline__
    void operator()(OutT *y,
                    OffsetOutputT *offsets,
                    const NzMaskT nzMask,
                    const PrefixSumT (&nzOffsets)[UNROLL],
                    const OffsetComputeT &offset0,
                    Args&&...) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            OffsetOutputT offset = static_cast<OffsetOutputT>(offset0) +
                                   i * BLOCK;
            if ((nzMask & (NzMaskT(1) << i)) != 0) {
                Stg(offset, offsets + nzOffsets[i]);
            }
        }
    }
};

/**
 * FAST:
 * true: fused kernel, compute prefix sum and write indices into nDims arrays.
 * false: compute prefix sum and write non-zero element offsets to offsets.
 *
 * 'offsets' was ignored if FAST == true
 */
template <int BLOCK, int UNROLL, bool FAST,
          typename NzMaskT, typename PrefixSumT, typename TilePrefixT,
          typename InT, typename OutT>
__global__ void NonZeroKernel(
        const InT *x,
        OutT *y,
        size_t *nonZeroCount,
        uint32_t *offsets,
        TilePrefixT *tilePrefix,
        Array<U32DivMod, TENSOR_DIM_MAX> strideDivMods,
        uint32_t nDims,
        uint32_t xSize) {
    __shared__ PrefixSumT smem[BLOCK * UNROLL];

    //---------------------------------------
    // step1: load input
    //---------------------------------------
    uint32_t offset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
    uint32_t xCount = offset0 < xSize ?
                      UIntDivRU<uint32_t>(xSize - offset0, BLOCK) : 0;
    const InT *xPtr = x + offset0;
    InT xRegs[UNROLL];

    if (xCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&xRegs[i], xPtr + i * BLOCK);
        }
    } else {
        // separated set-zero for loop can utilize CS2R instruction
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xRegs[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < xCount) {
                Ldg(&xRegs[i], xPtr + i * BLOCK);
            }
        }
    }

    //---------------------------------------
    // step2: non-zero scan
    //---------------------------------------
    static_assert(UNROLL <= sizeof(NzMaskT) * 8, "UNROLL out of bound");
    NzMaskT nonZeroMask = 0;
    PrefixSumT nonZeroRegs[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (xRegs[i] == InT(0)) {
            nonZeroRegs[i] = 0;
        } else {
            nonZeroRegs[i] = 1;
            nonZeroMask |= (NzMaskT(1) << i);
        }
    }

    // shuffle data for prefix scan
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        smem[threadIdx.x + i * BLOCK] = nonZeroRegs[i];
    }
    __syncthreads();

    //---------------------------------------
    // step3: prefix sum
    //---------------------------------------
    PrefixSumT threadSum = 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        nonZeroRegs[i] = smem[threadIdx.x * UNROLL + i];
        threadSum += nonZeroRegs[i];
    }

    uint32_t warpId = threadIdx.x / WARP_SIZE;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    PrefixSumT threadExclusivePrefix =
        prefix_sum::GlobalPrefixSum<prefix_sum::ScanMode::EXCLUSIVE, BLOCK>(
        threadSum, blockIdx.x, warpId, laneId, tilePrefix);

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        smem[threadIdx.x * UNROLL + i] = threadExclusivePrefix;
        threadExclusivePrefix += nonZeroRegs[i];
    }
    __syncthreads();

    //---------------------------------------
    // step4: post process
    //---------------------------------------
    PrefixSumT nzOffsets[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        nzOffsets[i] = smem[threadIdx.x + i * BLOCK];
    }

    PostProc<BLOCK, UNROLL, FAST>()(y, offsets, nonZeroMask, nzOffsets,
                                    offset0, xSize, nDims, strideDivMods);

    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == BLOCK - 1) {
        size_t nnz = nzOffsets[UNROLL - 1] +
                     ((nonZeroMask >> (UNROLL - 1)) & 0x1);
        *nonZeroCount = nnz;
    }
}

template <int BLOCK, int UNROLL, typename OutT>
__global__ void NonZeroIndicesKernel(
        const size_t *nonZeroCount,
        const uint32_t *offsets,
        OutT *y,
        Array<U32DivMod, TENSOR_DIM_MAX> strideDivMods,
        int nDims) {
    size_t nzCount;
    Ldg<NC>(&nzCount, nonZeroCount);

    uint32_t offset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
    if (offset0 >= nzCount) {
        return;
    }

    // step 1: load offsets
    uint32_t loopCount = offset0 < nzCount ?
                         UIntDivRU<uint32_t>(nzCount - offset0, BLOCK) : 0;

    uint32_t offsetRegs[UNROLL];
    const uint32_t *offsetsPtr = offsets + offset0;
    if (loopCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&offsetRegs[i], offsetsPtr + i * BLOCK);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < loopCount) {
                Ldg(&offsetRegs[i], offsetsPtr + i * BLOCK);
            }
        }
    }

    // step 2: compute index
    OutT indices[TENSOR_DIM_MAX][UNROLL];
    #pragma unroll
    for (int dim = 0; dim < TENSOR_DIM_MAX; ++dim) {
        if (dim >= nDims) {
            break;
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            auto divmod = strideDivMods[dim].DivMod(offsetRegs[i]);
            offsetRegs[i] = divmod.mod;
            indices[dim][i] = divmod.div;
        }
    }

    // step 3: output
    OutT *yPtr = y + offset0;
    if (loopCount >= UNROLL) {
        #pragma unroll
        for (int dim = 0; dim < TENSOR_DIM_MAX; ++dim) {
            if (dim >= nDims) {
                break;
            }
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                Stg(indices[dim][i], yPtr + i * BLOCK);
            }
            yPtr += nzCount;
        }
    } else {
        #pragma unroll
        for (int dim = 0; dim < TENSOR_DIM_MAX; ++dim) {
            if (dim >= nDims) {
                break;
            }
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                if (i < loopCount) {
                    Stg(indices[dim][i], yPtr + i * BLOCK);
                }
            }
            yPtr += nzCount;
        }
    }
}

template <typename IdxT>
hiednnStatus_t
LaunchNonZeroIndices(IdxT *y,
                     const uint32_t *offsets,
                     const size_t *nonZeroCount,
                     const Array<U32DivMod, TENSOR_DIM_MAX> &strideDivMods,
                     const int &nDims,
                     const uint32_t &xSize,
                     const cudaStream_t &stream) {
    constexpr uint32_t BLOCK = 256;

    if (nDims <= 1) {
        constexpr uint32_t UNROLL = 4;
        uint32_t nBlock = UIntDivRU<uint32_t>(xSize, BLOCK * UNROLL);
        NonZeroIndicesKernel<BLOCK, UNROLL><<<nBlock, BLOCK, 0, stream>>>(
            nonZeroCount, offsets, y, strideDivMods, nDims);
    } else if (nDims <= 3) {
        constexpr uint32_t UNROLL = 2;
        uint32_t nBlock = UIntDivRU<uint32_t>(xSize, BLOCK * UNROLL);
        NonZeroIndicesKernel<BLOCK, UNROLL><<<nBlock, BLOCK, 0, stream>>>(
            nonZeroCount, offsets, y, strideDivMods, nDims);
    } else {
        constexpr uint32_t UNROLL = 1;
        uint32_t nBlock = UIntDivRU<uint32_t>(xSize, BLOCK * UNROLL);
        NonZeroIndicesKernel<BLOCK, UNROLL><<<nBlock, BLOCK, 0, stream>>>(
            nonZeroCount, offsets, y, strideDivMods, nDims);
    }
    return HIEDNN_STATUS_SUCCESS;
}

template <bool FAST, typename T, typename IdxT>
hiednnStatus_t
LaunchNonZero(const T *x,
              IdxT *y,
              size_t *nonZeroCount,
              void *nonZeroWs,
              const int64_t (&xStrides)[TENSOR_DIM_MAX],
              const int &nDims,
              const uint32_t &xSize,
              const HiednnCudaHandle &handle) {
    using PrefixSumT = uint32_t;
    using TilePrefixT = prefix_sum::TilePrefix<PrefixSumT>;
    using NzMaskT = uint32_t;

    constexpr uint32_t BLOCK = 256;
    constexpr uint32_t UNROLL = 15;
    constexpr uint32_t TILE_SIZE = BLOCK * UNROLL;

    const auto &stream = handle.stream;
    Array<U32DivMod, 8> xStrideDivMods(xStrides, nDims);
    uint32_t nBlock = UIntDivRU<uint32_t>(xSize, TILE_SIZE);

    // separate workspace for prefix sum, this shall not be large
    size_t tilePrefixSize = prefix_sum::TilePrefixSize(nBlock);
    TilePrefixT *prefixSumWs;
    DeviceWsGuard wsGuard(handle);
    wsGuard.GetWorkspace(&prefixSumWs, tilePrefixSize * sizeof(TilePrefixT));
    if (prefixSumWs == nullptr) {
        return HIEDNN_STATUS_INTERNAL_ERROR;
    }

    // init prefix sum workspace
    prefix_sum::TilePrefixInit(prefixSumWs, tilePrefixSize, stream);

    // non-zero scan
    uint32_t *offsets = static_cast<uint32_t *>(nonZeroWs);
    NonZeroKernel<BLOCK, UNROLL, FAST, NzMaskT, PrefixSumT, TilePrefixT>
        <<<nBlock, BLOCK, 0, stream>>>(
        x, y, nonZeroCount, offsets, prefixSumWs, xStrideDivMods, nDims, xSize);

    if (!FAST) {
        return LaunchNonZeroIndices(y, offsets, nonZeroCount, xStrideDivMods,
                                    nDims, xSize, stream);
    }
    return HIEDNN_STATUS_SUCCESS;
}

template <bool FAST>
struct NonZeroMode {
    template <typename T>
    struct NonZeroImpl {
        template <typename IdxT>
        struct Impl {
            hiednnStatus_t
            operator()(const HiednnCudaHandle &handle,
                       const HiednnTensorDesc &xDesc,
                       const void *x,
                       void *y,
                       size_t *nonZeroCount,
                       void *workspace) const {
                // fast div range check
                if (xDesc.size > UINT32_MAX ||
                    U32DivMod::OutOfBound(xDesc.strides, xDesc.nDims)) {
                    return HIEDNN_STATUS_TENSOR_OVERSIZE;
                }

                uint32_t xSize = static_cast<uint32_t>(xDesc.size);
                return LaunchNonZero<FAST>(
                    static_cast<const T *>(x), static_cast<IdxT *>(y),
                    nonZeroCount, workspace, xDesc.strides, xDesc.nDims,
                    xSize, handle);
            }
        };

        hiednnStatus_t
        operator()(const HiednnCudaHandle &handle,
                   const HiednnTensorDesc &xDesc,
                   const void *x,
                   const HiednnTensorDesc &yDesc,
                   void *y,
                   size_t *nonZeroCount,
                   void *workspace) const {
            switch (yDesc.dataType) {
                case HIEDNN_DATATYPE_UINT32:
                    return Impl<uint32_t>()(handle, xDesc, x, y,
                                            nonZeroCount, workspace);
                case HIEDNN_DATATYPE_UINT64:
                    return Impl<uint64_t>()(handle, xDesc, x, y,
                                            nonZeroCount, workspace);
                default:
                    return HIEDNN_STATUS_INVALID_DATATYPE;
            }
        }
    };
};

}   // anonymous namespace

}   // namespace cuda

}   // namespace hiednn

size_t
hiednnCudaNonZeroGetWorkspaceSize(HiednnTensorDesc *xDesc) {
    return xDesc->size <= std::numeric_limits<uint32_t>::max()
                        ? xDesc->size * sizeof(uint32_t)
                        : xDesc->size * sizeof(uint64_t);
}

hiednnStatus_t
hiednnCudaNonZero(HiednnCudaHandle *cudaHandle,
                  HiednnTensorDesc *xDesc,
                  const void *x,
                  HiednnTensorDesc *yDesc,
                  void *y,
                  size_t *nonZeroCount,
                  void *workspace,
                  size_t wsSizeInBytes) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, x, yDesc, y,
                              nonZeroCount, workspace)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // only support normal tensors for now
    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // ensure enough space
    if (yDesc->size < xDesc->nDims * xDesc->size) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (wsSizeInBytes < hiednnCudaNonZeroGetWorkspaceSize(xDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->IsIntegral()) {
        return hiednn::DispatchItemSize
            <hiednn::cuda::NonZeroMode<false>::NonZeroImpl>(
                xDesc->dataType, *cudaHandle, *xDesc, x,
                *yDesc, y, nonZeroCount, workspace);
    } else {
        return hiednn::DispatchFP
            <hiednn::cuda::NonZeroMode<false>::NonZeroImpl>(
                xDesc->dataType, *cudaHandle, *xDesc, x,
                *yDesc, y, nonZeroCount, workspace);
    }
}

hiednnStatus_t
hiednnCudaFastNonZero(HiednnCudaHandle *cudaHandle,
                      HiednnTensorDesc *xDesc,
                      const void *x,
                      HiednnTensorDesc *yDesc,
                      void *y,
                      size_t *nonZeroCount) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, x, yDesc, y, nonZeroCount)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // only support normal tensors for now
    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->nDims != 2 ||
        // ensure enough space
        yDesc->dims[0] != xDesc->nDims ||
        yDesc->dims[1] != xDesc->size) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->IsIntegral()) {
        return hiednn::DispatchItemSize
            <hiednn::cuda::NonZeroMode<true>::NonZeroImpl>(
                xDesc->dataType, *cudaHandle, *xDesc, x,
                *yDesc, y, nonZeroCount, nullptr);
    } else {
        return hiednn::DispatchFP
            <hiednn::cuda::NonZeroMode<true>::NonZeroImpl>(
                xDesc->dataType, *cudaHandle, *xDesc, x,
                *yDesc, y, nonZeroCount, nullptr);
    }
}
