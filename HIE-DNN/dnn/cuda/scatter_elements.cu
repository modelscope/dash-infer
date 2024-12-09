/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scatter_elements.cu
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
          typename OffsetT, int NumAxes, int Axis>
__global__ void ScatterElementsKernel(
        const IdxT *indices,
        const T *updates,
        T *y,
        Array<U32DivMod, NumAxes> updatesStrides,
        Array<OffsetT, NumAxes> yStrides,
        OffsetT yAxisStride,  // yDims[Axis] * yStrides[Axis]
        uint32_t updatesSize) {
    // thread's first (without unrolling) updates offset
    // updates offset is bounded by uint32_t, for updatesSize <= UINT32_MAX
    uint32_t updatesOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
    uint32_t updatesCount = updatesSize > updatesOffset0 ?
        UIntDivRU<uint32_t>(updatesSize - updatesOffset0, BLOCK) : 0;

    // step 1: load indices
    IdxT indicesReg[UNROLL];
    if (updatesCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&indicesReg[i], indices + updatesOffset0 + i * BLOCK);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < updatesCount) {
                Ldg(&indicesReg[i], indices + updatesOffset0 + i * BLOCK);
            }
        }
    }

    // step 2: compute corresponding indices in y
    // step 2.1: compute original indices implied by position
    OffsetT yOffsets[UNROLL]{0};
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        uint32_t updatesOffset = updatesOffset0 + i * BLOCK;
        // [0, axis)
        #pragma unroll
        for (int dim = 0; dim < Axis; ++dim) {
            auto divmod = updatesStrides[dim].DivMod(updatesOffset);
            updatesOffset = divmod.mod;
            yOffsets[i] += static_cast<OffsetT>(divmod.div) * yStrides[dim];
        }

        // axis
        auto divmod = updatesStrides[Axis].DivMod(updatesOffset);
        updatesOffset = divmod.mod;

        // [axis + 1, NumAxes)
        #pragma unroll
        for (int dim = Axis + 1; dim < NumAxes; ++dim) {
            auto divmod = updatesStrides[dim].DivMod(updatesOffset);
            updatesOffset = divmod.mod;
            yOffsets[i] += static_cast<OffsetT>(divmod.div) * yStrides[dim];
        }
    }

    // step 2.2: apply loaded indices
    // assert: unsigned mul here has the same bits as signed mul
    // assert: unsigned add here has the same result as signed add
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        yOffsets[i] += static_cast<OffsetT>(indicesReg[i]) * yStrides[Axis];
    }
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (indicesReg[i] < 0) {
            yOffsets[i] += yAxisStride;
        }
    }

    // step 3: scatter updates into y
    // step 3.1: load updates
    T updatesReg[UNROLL];
    if (updatesCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg(&updatesReg[i], updates + updatesOffset0 + i * BLOCK);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Stg(updatesReg[i], y + yOffsets[i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < updatesCount) {
                Ldg(&updatesReg[i], updates + updatesOffset0 + i * BLOCK);
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < updatesCount) {
                Stg(updatesReg[i], y + yOffsets[i]);
            }
        }
    }
}

template <typename IdxT, typename T, int NumAxes, int Axis>
hiednnStatus_t
LaunchScatterElements(
        const T *x,
        const IdxT *indices,
        const T *updates,
        T *y,
        const int64_t *yDims,
        const int64_t *yStrides,
        const int64_t *updatesStrides,
        size_t ySize,
        size_t updatesSize,
        int nDims,
        cudaStream_t stream) {
    constexpr uint32_t BLOCK = 128;
    constexpr uint32_t UNROLLED_BYTE = 16;
    constexpr uint32_t UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                                UNROLLED_BYTE / sizeof(T) : 8;

    // fast div range check
    if (updatesSize > UINT32_MAX
        || U32DivMod::OutOfBound(updatesStrides, nDims)) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    // copy x to y first, then scatter updates into y
    if (y != x) {
        CHECK_CUDA_RETURN(cudaMemcpyAsync(
            y, x, ySize * sizeof(T), cudaMemcpyDefault, stream));
    }

    // if updatesSize == 0, no kernel launch needed, just return
    if (updatesSize == 0UL) {
        return HIEDNN_STATUS_SUCCESS;
    }

    uint32_t nBlock = UIntDivRU<uint32_t>(static_cast<uint32_t>(updatesSize),
                                          BLOCK * UNROLL);

    /* If ySize <= UINT32_MAX, then y offsets, y dims, and y strides are all
     * bounded by uint32_t.
     *
     * yDims[Axis] * yStrides[Axis] <= ySize is also bounded by uint32_t.
     *
     * Even if IdxT is int64_t, the value in indices tensor must
     * also be bounded by uint32_t, because its range is [-s, s) where s is
     * the dim of y. So the loaded indices can also be casted down to uint32_t.
     * Though it should be a signed integer, uint32_t will not lose information.
     *
     * If the loaded index is negative, i.e., its range is [-s, 0), by
     * multiplying the unsigned value from down-casting and the y stride,
     * the result is still bounded by uint32_t because the absolute value of
     * the original product is bounded by yDims[Axis] * yStrides[Axis] <= ySize.
     * Further, its bit representation is the same as the signed product.
     * By adding it to the y offset, the result is the same as the signed.
     */
    if (ySize <= UINT32_MAX) {
        using OffsetT = uint32_t;
        ScatterElementsKernel<BLOCK, UNROLL, IdxT, T, OffsetT, NumAxes, Axis>
            <<<nBlock, BLOCK, 0, stream>>>(
                indices, updates, y,
                Array<U32DivMod, NumAxes>(updatesStrides, nDims, 1L),
                Array<OffsetT, NumAxes>(yStrides, nDims, 1L),
                static_cast<OffsetT>(yDims[Axis] * yStrides[Axis]),
                static_cast<uint32_t>(updatesSize));
    } else {
        using OffsetT = size_t;
        ScatterElementsKernel<BLOCK, UNROLL, IdxT, T, OffsetT, NumAxes, Axis>
            <<<nBlock, BLOCK, 0, stream>>>(
                indices, updates, y,
                Array<U32DivMod, NumAxes>(updatesStrides, nDims, 1L),
                Array<OffsetT, NumAxes>(yStrides, nDims, 1L),
                static_cast<OffsetT>(yDims[Axis] * yStrides[Axis]),
                static_cast<uint32_t>(updatesSize));
    }

    CHECK_CUDA_RETURN(cudaGetLastError());
    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct ScatterElementsImpl {
    template <typename IdxT>
    struct Impl {
        template <int NumAxes, int Axis = 0>
        struct DispatchAxis {
            hiednnStatus_t
            operator()(const HiednnTensorDesc &xDesc,
                       const void *x,
                       const HiednnTensorDesc &indicesDesc,
                       const void *indices,
                       const HiednnTensorDesc &updatesDesc,
                       const void *updates,
                       const HiednnTensorDesc &yDesc,
                       void *y,
                       int axis,
                       cudaStream_t stream) const {
                if (axis == Axis) {
                    return LaunchScatterElements<IdxT, T, NumAxes, Axis>(
                            static_cast<const T *>(x),
                            static_cast<const IdxT *>(indices),
                            static_cast<const T *>(updates),
                            static_cast<T *>(y),
                            yDesc.dims, yDesc.strides, updatesDesc.strides,
                            yDesc.size, updatesDesc.size, yDesc.nDims,
                            stream);
                } else {
                    return DispatchAxis<NumAxes, Axis + 1>()(
                        xDesc, x, indicesDesc, indices,
                        updatesDesc, updates, yDesc, y, axis, stream);
                }
            }
        };  // ScatterElementsImpl::Impl::DispatchAxis

        template <int NumAxes>
        struct DispatchAxis<NumAxes, NumAxes> {
            hiednnStatus_t
            operator()(const HiednnTensorDesc &xDesc,
                       const void *x,
                       const HiednnTensorDesc &indicesDesc,
                       const void *indices,
                       const HiednnTensorDesc &updatesDesc,
                       const void *updates,
                       const HiednnTensorDesc &yDesc,
                       void *y,
                       int axis,
                       cudaStream_t stream) const {
                return HIEDNN_STATUS_INTERNAL_ERROR;
            }
        };  // ScatterElementsImpl::Impl::DispatchAxis

        hiednnStatus_t
        operator()(const HiednnTensorDesc &xDesc,
                   const void *x,
                   const HiednnTensorDesc &indicesDesc,
                   const void *indices,
                   const HiednnTensorDesc &updatesDesc,
                   const void *updates,
                   const HiednnTensorDesc &yDesc,
                   void *y,
                   int axis,
                   cudaStream_t stream) const {
            const auto &nDims = xDesc.nDims;
            // assert: nDims >= 1
            // assert: 0 <= axis < nDims
            if (nDims <= 1) {
                return DispatchAxis<1>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else if (nDims <= 2) {
                return DispatchAxis<2>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else if (nDims <= 3) {
                return DispatchAxis<3>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else if (nDims <= 4) {
                return DispatchAxis<4>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else if (nDims <= 5) {
                return DispatchAxis<5>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else if (nDims <= 8) {
                return DispatchAxis<8>()(xDesc, x, indicesDesc, indices,
                                         updatesDesc, updates, yDesc, y,
                                         axis, stream);
            } else {
                return HIEDNN_STATUS_INTERNAL_ERROR;
            }
        }
    };  // ScatterElementsImpl::Impl

    hiednnStatus_t
    operator()(const HiednnTensorDesc &xDesc,
               const void *x,
               const HiednnTensorDesc &indicesDesc,
               const void *indices,
               const HiednnTensorDesc &updatesDesc,
               const void *updates,
               const HiednnTensorDesc &yDesc,
               void *y,
               int axis,
               cudaStream_t stream) const {
        switch (indicesDesc.dataType) {
            case HIEDNN_DATATYPE_INT32:
                return Impl<int32_t>()(xDesc, x, indicesDesc, indices,
                                       updatesDesc, updates, yDesc, y,
                                       axis, stream);
            case HIEDNN_DATATYPE_INT64:
                return Impl<int64_t>()(xDesc, x, indicesDesc, indices,
                                       updatesDesc, updates, yDesc, y,
                                       axis, stream);
            default:
                return HIEDNN_STATUS_INVALID_DATATYPE;
        }
    }
};  // ScatterElementsImpl

}   // anonymous namespace

}   // namespace cuda

}   // namespace hiednn

hiednnStatus_t
hiednnCudaScatterElements(HiednnCudaHandle *cudaHandle,
                          HiednnTensorDesc *xDesc,
                          const void *x,
                          HiednnTensorDesc *indicesDesc,
                          const void *indices,
                          HiednnTensorDesc *updatesDesc,
                          const void *updates,
                          HiednnTensorDesc *yDesc,
                          void *y,
                          int axis,
                          hiednnScatterElemReduce_t reduction) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, x, indicesDesc, indices,
            updatesDesc, updates, yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(
            *xDesc, *indicesDesc, *updatesDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->nDims != indicesDesc->nDims ||
        !indicesDesc->SameDim(*updatesDesc) ||
        xDesc->dataType != updatesDesc->dataType ||
        *xDesc != *yDesc) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // reduction is not supported
    if (reduction != HIEDNN_SCATTERELEM_REDUCE_NONE) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // nDims is guaranteed to be >= 1
    const auto &nDims = xDesc->nDims;

    // check & adjust axis to [0, r)
    if (axis < -nDims || axis >= nDims) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (axis < 0) {
        axis += nDims;
    }

    // shape check
    for (int i = 0; i < nDims; ++i) {
        // for dims other than 'axis', cannot index nonexistent indices
        if (i != axis && indicesDesc->dims[i] > xDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    if (yDesc->size == 0UL) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchAll<hiednn::cuda::ScatterElementsImpl>(
        xDesc->dataType, *xDesc, x, *indicesDesc, indices,
        *updatesDesc, updates, *yDesc, y, axis, cudaHandle->stream);
}
