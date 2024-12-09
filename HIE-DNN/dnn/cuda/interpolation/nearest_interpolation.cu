/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    nearest_interpolation.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <tensor_desc.hpp>
#include <utils.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

#include "coordinate_functor.cuh"

namespace hiednn {

namespace cuda {

namespace interpolation {

template <int BLOCK,            // thread block
          int BATCH_UNROLL,     // unrolling factor on batch dimension
          int IMAGE_UNROLL,     // unrolling factor on DHW... dimension
          int NDIMS,            // NDIMS-interpolation
          typename CoordFunc,   // coordinate mode functor
          typename IdxT,        // type of index and offset
          typename MapT,        // type of coordinate calculation
          typename DT>
__global__ void NearestInterpolationKernel(
        const DT *x,
        DT *y,
        Array<U32DivMod, NDIMS> yDimDivMod,
        Array<CoordFunc, NDIMS> coordFunc,
        Array<IdxT, NDIMS> xStrides,
        Array<IdxT, NDIMS> xCoordMax,
        IdxT batch,
        IdxT waveBatch,
        IdxT yWaveSize,
        IdxT xWaveOffset,
        IdxT xBatchStride) {
    // limited by yDimDivMod (U32DivMod)
    static_assert(sizeof(IdxT) <= sizeof(uint32_t), "invalid IdxT");

    IdxT threadOffset = blockIdx.x * BLOCK * IMAGE_UNROLL + threadIdx.x;

    if (threadOffset >= yWaveSize) {
        return;
    }

    // y point coordinate
    IdxT batchId[IMAGE_UNROLL];
    IdxT yCoord[IMAGE_UNROLL][NDIMS];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        batchId[i] = threadOffset + i * BLOCK;
    }
    #pragma unroll
    for (int i = NDIMS - 1; i >= 0; --i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            auto dm = yDimDivMod[i].DivMod(batchId[j]);
            yCoord[j][i] = dm.mod;
            batchId[j] = dm.div;
        }
    }

    // convert integer coordinate to float
    MapT xCoordFP[IMAGE_UNROLL][NDIMS];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < NDIMS; ++j) {
            xCoordFP[i][j] = static_cast<MapT>(yCoord[i][j]);
        }
    }

    // coordinate function
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            xCoordFP[j][i] = coordFunc[i].Coordinate(xCoordFP[j][i]);
        }
    }

    // convert float coordinate to integer
    IdxT xCoord[IMAGE_UNROLL][NDIMS];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            xCoord[j][i] = static_cast<IdxT>(xCoordFP[j][i]);
            if (xCoord[j][i] > xCoordMax[i]) {
                xCoord[j][i] = xCoordMax[i];
            }
        }
    }

    // offset of input points
    IdxT xOffset[IMAGE_UNROLL];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        xOffset[i] = batchId[i] * xBatchStride;
    }
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            xOffset[j] += xCoord[j][i] * xStrides[i];
        }
    }

    // image unrolling out of bound
    bool outOfBound[IMAGE_UNROLL];
    IdxT yCount = yWaveSize > threadOffset ?
                  UIntDivRU<IdxT>(yWaveSize - threadOffset, BLOCK) : 0;
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        outOfBound[i] = i >= yCount;
    }

    // load x points
    DT xData[IMAGE_UNROLL][BATCH_UNROLL];
    #pragma unroll
    for (int imageIt = 0; imageIt < IMAGE_UNROLL; ++imageIt) {
        if (outOfBound[imageIt]) {
            break;
        }

        if (batchId[imageIt] + (BATCH_UNROLL - 1) * waveBatch < batch) {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                Ldg<NC>(&xData[imageIt][i],
                        x + xOffset[imageIt] + xWaveOffset * i);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                if (batchId[imageIt] + i * waveBatch >= batch) {
                    break;
                }
                Ldg<NC>(&xData[imageIt][i],
                        x + xOffset[imageIt] + xWaveOffset * i);
            }
        }
    }

    // store y points
    DT *yPtr = y + threadOffset;
    #pragma unroll
    for (int imageIt = 0; imageIt < IMAGE_UNROLL; ++imageIt) {
        if (outOfBound[imageIt]) {
            break;
        }

        if (batchId[imageIt] + (BATCH_UNROLL - 1) * waveBatch < batch) {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                yPtr[imageIt * BLOCK + i * yWaveSize] = xData[imageIt][i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                if (batchId[imageIt] + i * waveBatch >= batch) {
                    break;
                }
                yPtr[imageIt * BLOCK + i * yWaveSize] = xData[imageIt][i];
            }
        }
    }
}

template <int NDIMS, typename CoordFunc, typename MapT, typename T>
hiednnStatus_t LaunchNearestInterpolation(
        const HiednnCudaHandle &handle,
        const int64_t *xDims,
        const int64_t *xStrides,
        const T *x,
        const float *scale,
        const int64_t *yDims,
        const int64_t *yStrides,
        T *y) {
    // for NDIMS-interpolation, 3*NDIMS SFU instructions for each output point
    const int SFU_INSTS = 3 * NDIMS;
    const int BLOCK = 128;

    using IdxT = uint32_t;

    Array<IdxT, NDIMS> xStride;
    Array<U32DivMod, NDIMS> yDimDivMod;
    Array<CoordFunc, NDIMS> coordFunc;
    Array<IdxT, NDIMS> xCoordMax;

    for (int i = 0; i < NDIMS; ++i) {
        xStride[i] = xStrides[i + 1];
        yDimDivMod[i] = U32DivMod(yDims[i + 1]);
        coordFunc[i] = CoordFunc(xDims[i + 1], yDims[i + 1], scale[i]);
        xCoordMax[i] = static_cast<IdxT>(xDims[i + 1] - 1);
    }

    // upper bound of SFU instructions for each output point on avarage
    const int SFU_BOUND = 6;
    const int BATCH_UNROLL = ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N;
    const int IMAGE_UNROLL = ConstExpr::DivRU<16 / sizeof(T), BATCH_UNROLL>::N;
    static_assert(sizeof(T) <= 16, "LaunchNearestInterpolation: invalid T");

    IdxT batch = xDims[0];
    IdxT waveBatch = UIntDivRU<IdxT>(batch, BATCH_UNROLL);
    IdxT yWaveSize = waveBatch * yStrides[0];
    IdxT xWaveOffset = waveBatch * xStrides[0];
    IdxT xBatchStride = xStrides[0];

    IdxT grid = UIntDivRU<IdxT>(yWaveSize, BLOCK * IMAGE_UNROLL);

    NearestInterpolationKernel<
        BLOCK, BATCH_UNROLL, IMAGE_UNROLL, NDIMS, CoordFunc, IdxT, MapT>
        <<<grid, BLOCK, 0, handle.stream>>>(
        x, y, yDimDivMod, coordFunc, xStride, xCoordMax, batch,
        waveBatch, yWaveSize, xWaveOffset, xBatchStride);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename CoordFunc, typename MapT, typename T>
hiednnStatus_t NearestInterpolation(
        const HiednnCudaHandle &handle,
        const HiednnTensorDesc &xDesc,
        const T *x,
        const float *scale,
        const HiednnTensorDesc &yDesc,
        T *y) {
    // limited by integer fast division
    if (xDesc.size >= INT32_MAX || yDesc.size >= INT32_MAX) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    const int64_t *xDims = xDesc.dims;
    const int64_t *xStrides = xDesc.strides;
    const int64_t *yDims = yDesc.dims;
    const int64_t *yStrides = yDesc.strides;
    int interpDims = xDesc.nDims - 1;

    hiednnStatus_t ret;
    // support 1-, 2-, 3-D interpolation
    switch (interpDims) {
        case 1:
            ret = LaunchNearestInterpolation<1, CoordFunc, MapT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 2:
            ret = LaunchNearestInterpolation<2, CoordFunc, MapT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 3:
            ret = LaunchNearestInterpolation<3, CoordFunc, MapT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return ret;
}

template <template <typename> class RoundFunc, typename MapT, typename T>
hiednnStatus_t DispatchCoordMode(
        const HiednnCudaHandle &handle,
        hiednnInterpCoordMode_t coordMode,
        const HiednnTensorDesc &xDesc,
        const T *x,
        const float *scale,
        const HiednnTensorDesc &yDesc,
        T *y) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;

    switch (coordMode) {
        case HIEDNN_INTERP_COORD_HALF_PIXEL:
            ret = NearestInterpolation<
                HalfPixel<RoundFunc, MapT>, MapT, T>(
                handle, xDesc, x, scale, yDesc, y);
            break;
        case HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL:
            ret = NearestInterpolation<
                PytorchHalfPixel<RoundFunc, MapT>, MapT, T>(
                handle, xDesc, x, scale, yDesc, y);
            break;
        case HIEDNN_INTERP_COORD_ALIGN_CORNER:
            ret = NearestInterpolation<
                AlignCorner<RoundFunc, MapT>, MapT, T>(
                handle, xDesc, x, scale, yDesc, y);
            break;
        case HIEDNN_INTERP_COORD_ASYMMETRIC:
            ret = NearestInterpolation<
                Asymmetric<RoundFunc, MapT>, MapT, T>(
                handle, xDesc, x, scale, yDesc, y);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
            break;
    }

    return ret;
}

template <typename T>
struct NearestInterpolationImpl {
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            hiednnInterpCoordMode_t coordMode,
            hiednnInterpNearestMode_t nearestMode,
            const HiednnTensorDesc &xDesc,
            const void *x,
            const float *scale,
            const HiednnTensorDesc &yDesc,
            void *y) {
        const T *xPtr = static_cast<const T *>(x);
        const float *scalePtr = static_cast<const float *>(scale);
        T *yPtr = static_cast<T *>(y);

        using MapT = float;

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        // dispatch nearest mode
        switch (nearestMode) {
            case HIEDNN_INTERP_NEAREST_HALF_DOWN:
                ret = DispatchCoordMode<RoundHalfDown, MapT, T>(
                    handle, coordMode, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_NEAREST_HALF_UP:
                ret = DispatchCoordMode<RoundHalfUp, MapT, T>(
                    handle, coordMode, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_NEAREST_FLOOR:
                ret = DispatchCoordMode<RoundFloor, MapT, T>(
                    handle, coordMode, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_NEAREST_CEIL:
                ret = DispatchCoordMode<RoundCeil, MapT, T>(
                    handle, coordMode, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_PARAMETER;
                break;
        }

        return ret;
    }
};

}  // namespace interpolation

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaNearestInterpolation(HiednnCudaHandle *cudaHandle,
                               hiednnInterpCoordMode_t coordMode,
                               hiednnInterpNearestMode_t nearestMode,
                               HiednnTensorDesc *xDesc,
                               const void *x,
                               const float *scale,
                               int scaleSize,
                               HiednnTensorDesc *yDesc,
                               void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->dataType != yDesc->dataType ||
        xDesc->nDims != yDesc->nDims ||
        xDesc->nDims < 2 ||
        xDesc->dims[0] != yDesc->dims[0] ||
        xDesc->size == 0 || yDesc->size == 0 ||
        scaleSize != xDesc->nDims - 1) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchAll<
        hiednn::cuda::interpolation::NearestInterpolationImpl>(
        xDesc->dataType, *cudaHandle, coordMode, nearestMode,
        *xDesc, x, scale, *yDesc, y);
}


