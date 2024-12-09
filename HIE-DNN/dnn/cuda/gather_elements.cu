/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gather_elements.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <tensor_desc.hpp>
#include <utils.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, typename IdxT, typename T,
          typename OffsetT, int N_AXES, int THE_AXIS>
__global__ void GatherElementsKernel(
        const T *x,
        const IdxT *indices,
        T *y,
        Array<U32DivMod, N_AXES> yStrideDivMods,
        Array<OffsetT, N_AXES> xStrides,
        OffsetT xAxisStride,  // xDims[THE_AXIS] * xStrides[THE_AXIS]
        uint32_t ySize) {
    // thread's first (without unrolling) y offset
    // y offset is bounded by uint32_t, because ySize <= UINT32_MAX
    uint32_t yOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
    uint32_t yCount = ySize > yOffset0 ?
                      UIntDivRU<uint32_t>(ySize - yOffset0, BLOCK) : 0;

    // load indices
    IdxT gatherIndexReg[UNROLL];
    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&gatherIndexReg[i], indices + yOffset0 + i * BLOCK);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Ldg(&gatherIndexReg[i], indices + yOffset0 + i * BLOCK);
            }
        }
    }

    OffsetT xOffsets[UNROLL]{0};
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        uint32_t yOffset = yOffset0 + i * BLOCK;
        // [0, axis)
        #pragma unroll
        for (int dim = 0; dim < THE_AXIS; ++dim) {
            auto divmod = yStrideDivMods[dim].DivMod(yOffset);
            yOffset = divmod.mod;
            xOffsets[i] += static_cast<OffsetT>(divmod.div) * xStrides[dim];
        }

        // axis
        auto divmod = yStrideDivMods[THE_AXIS].DivMod(yOffset);
        yOffset = divmod.mod;

        // [axis + 1, N_AXES)
        #pragma unroll
        for (int dim = THE_AXIS + 1; dim < N_AXES; ++dim) {
            auto divmod = yStrideDivMods[dim].DivMod(yOffset);
            yOffset = divmod.mod;
            xOffsets[i] += static_cast<OffsetT>(divmod.div) * xStrides[dim];
        }
    }

    // perform loaded index of `axis'
    // assert: unsigned addition here has the same result as signed addition
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xOffsets[i] += static_cast<OffsetT>(gatherIndexReg[i]) *
                       xStrides[THE_AXIS];
    }
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (gatherIndexReg[i] < 0) {
            xOffsets[i] += xAxisStride;
        }
    }

    T regs[UNROLL];
    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&regs[i], x + xOffsets[i]);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Stg(regs[i], y + yOffset0 + i * BLOCK);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Ldg(&regs[i], x + xOffsets[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Stg(regs[i], y + yOffset0 + i * BLOCK);
            }
        }
    }
}

template <typename IdxT, typename T, int N_AXES, int THE_AXIS>
hiednnStatus_t
LaunchGatherElements(
        const T *x,
        const IdxT *indices,
        T *y,
        const int64_t (&xDims)[TENSOR_DIM_MAX],
        const int64_t (&xStrides)[TENSOR_DIM_MAX],
        const int64_t (&yStrides)[TENSOR_DIM_MAX],
        size_t xSize,
        uint32_t ySize,
        int nDims,
        cudaStream_t stream) {
    constexpr uint32_t BLOCK = 128;
    constexpr uint32_t UNROLL = 8;
    uint32_t nBlock = UIntDivRU(ySize, BLOCK * UNROLL);

    /* If xSize <= UINT32_MAX, then x offsets, x dims, and x strides are all
     * bounded by uint32_t.
     *
     * Even if IdxT is int64_t, the value in indices tensor must
     * also be bounded by uint32_t, because its range is [-s, s) where s is
     * the dim of x. So the loaded indices can also be casted down to uint32_t.
     * Though it should be a signed integer, uint32_t will not lose information.
     *
     * If the loaded index is negative and needs adding the dim to adjust to a
     * positive value, the unsigned addition will show the same result as the
     * signed addition, because the result index is in [0, UINT32_MAX).
     */
    if (xSize <= UINT32_MAX) {
        using OffsetT = uint32_t;
        GatherElementsKernel
            <BLOCK, UNROLL, IdxT, T, OffsetT, N_AXES, THE_AXIS>
            <<<nBlock, BLOCK, 0, stream>>>(
            x, indices, y, Array<U32DivMod, N_AXES>(yStrides, nDims, 1L),
            Array<OffsetT, N_AXES>(xStrides, nDims, 1L),
            static_cast<OffsetT>(xDims[THE_AXIS]) * xStrides[THE_AXIS], ySize);
    } else {
        using OffsetT = size_t;
        GatherElementsKernel
            <BLOCK, UNROLL, IdxT, T, OffsetT, N_AXES, THE_AXIS>
            <<<nBlock, BLOCK, 0, stream>>>(
            x, indices, y, Array<U32DivMod, N_AXES>(yStrides, nDims, 1L),
            Array<OffsetT, N_AXES>(xStrides, nDims, 1L),
            static_cast<OffsetT>(xDims[THE_AXIS]) * xStrides[THE_AXIS], ySize);
    }

    CHECK_CUDA_RETURN(cudaGetLastError());
    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct GatherElementsImpl {
    template <typename IdxT>
    struct Impl {
        template <int N_AXES, int AXIS = 0>
        struct DispatchAxis {
            hiednnStatus_t
            operator()(const HiednnTensorDesc &xDesc,
                       const void *x,
                       const HiednnTensorDesc &indicesDesc,
                       const void *indices,
                       const HiednnTensorDesc &yDesc,
                       void *y,
                       int axis,
                       cudaStream_t stream) {
                if (axis == AXIS) {
                    // assert: yDesc.size <= UINT32_MAX
                    return LaunchGatherElements<IdxT, T, N_AXES, AXIS>(
                        static_cast<const T *>(x),
                        static_cast<const IdxT *>(indices), static_cast<T *>(y),
                        xDesc.dims, xDesc.strides, yDesc.strides, xDesc.size,
                        static_cast<uint32_t>(yDesc.size), xDesc.nDims, stream);
                } else {
                    return DispatchAxis<N_AXES, AXIS + 1>()(
                        xDesc, x, indicesDesc, indices, yDesc, y, axis, stream);
                }
            }
        };

        template <int N_AXES>
        struct DispatchAxis<N_AXES, N_AXES> {
            hiednnStatus_t
            operator()(const HiednnTensorDesc &xDesc,
                       const void *x,
                       const HiednnTensorDesc &indicesDesc,
                       const void *indices,
                       const HiednnTensorDesc &yDesc,
                       void *y,
                       int axis,
                       cudaStream_t stream) {
                return HIEDNN_STATUS_INTERNAL_ERROR;
            }
        };

        hiednnStatus_t
        operator()(const HiednnTensorDesc &xDesc,
                   const void *x,
                   const HiednnTensorDesc &indicesDesc,
                   const void *indices,
                   const HiednnTensorDesc &yDesc,
                   void *y,
                   int axis,
                   cudaStream_t stream) {
            const auto &nDims = xDesc.nDims;
            // fast div range check
            if (yDesc.size > UINT32_MAX
                || U32DivMod::OutOfBound(yDesc.strides, nDims)) {
                return HIEDNN_STATUS_TENSOR_OVERSIZE;
            }

            // assert: xDesc.nDims >= 1
            // assert: 0 <= axis < xDesc.nDims
            if (nDims <= 1) {
                return DispatchAxis<1>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else if (nDims <= 2) {
                return DispatchAxis<2>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else if (nDims <= 3) {
                return DispatchAxis<3>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else if (nDims <= 4) {
                return DispatchAxis<4>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else if (nDims <= 5) {
                return DispatchAxis<5>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else if (nDims <= 8) {
                return DispatchAxis<8>()(xDesc, x, indicesDesc, indices,
                                         yDesc, y, axis, stream);
            } else {
                return HIEDNN_STATUS_INTERNAL_ERROR;
            }
        }
    };

    hiednnStatus_t
    operator()(const HiednnTensorDesc &xDesc,
               const void *x,
               const HiednnTensorDesc &indicesDesc,
               const void *indices,
               const HiednnTensorDesc &yDesc,
               void *y,
               int axis,
               cudaStream_t stream) {
        switch (indicesDesc.dataType) {
            case HIEDNN_DATATYPE_INT32:
                return Impl<int32_t>()(xDesc, x, indicesDesc, indices, yDesc, y,
                                       axis, stream);
            case HIEDNN_DATATYPE_INT64:
                return Impl<int64_t>()(xDesc, x, indicesDesc, indices, yDesc, y,
                                       axis, stream);
            default:
                return HIEDNN_STATUS_INVALID_DATATYPE;
        }
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaGatherElements(HiednnCudaHandle *cudaHandle,
                         HiednnTensorDesc *xDesc,
                         const void *x,
                         HiednnTensorDesc *indicesDesc,
                         const void *indices,
                         HiednnTensorDesc *yDesc,
                         void *y,
                         int axis) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, indicesDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *indicesDesc, indices, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc, *indicesDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->nDims != indicesDesc->nDims
        || yDesc->nDims != indicesDesc->nDims
        || xDesc->dataType != yDesc->dataType) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // check & adjust axis to [0, r)
    // nDims >= 1 is guaranteed by SetTensorDescriptor
    if (axis < -xDesc->nDims || axis >= xDesc->nDims) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (axis < 0) {
        axis += xDesc->nDims;
    }

    // shape check
    for (int i = 0; i < indicesDesc->nDims; ++i) {
        // y is of the same shape as indices
        if (yDesc->dims[i] != indicesDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        // for dims other than `axis', cannot index nonexistent indices
        if (i != axis && indicesDesc->dims[i] > xDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    if (xDesc->size == 0UL || yDesc->size == 0UL) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::GatherElementsImpl>(
        xDesc->dataType, *xDesc, x, *indicesDesc, indices, *yDesc, y,
        axis, cudaHandle->stream);
}

