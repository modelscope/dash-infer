/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    linear_interpolation.cu
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
#include <cuda/intrinsic/type_conversion.hpp>

#include "coordinate_functor.cuh"

namespace hiednn {

namespace cuda {

namespace interpolation {

template <int Y>
using POW2 = ConstExpr::Pow<2, Y>;

template <typename T>
struct ComputeType {
    using type = T;
};

#ifdef HIEDNN_USE_FP16
template <>
struct ComputeType<half> {
    using type = float;
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct ComputeType<bfloat16> {
    using type = float;
};
#endif

// inCoord: L_D0, L_D1, L_D2... R_D0, R_D1, R_D2...
//     L: coordinate round down
//     R: coordinate round up
template <int NDIMS, int LOOP, typename IdxT>
struct LinearOffsetHelper {
    static __device__ __forceinline__
    void GetInputOffset(const Array<IdxT, NDIMS> &stride,
                        const IdxT (&inCoord)[NDIMS * 2],
                        const IdxT &baseOffset,
                        IdxT *inOffset) {
        IdxT offsetL = baseOffset +
                        inCoord[NDIMS - LOOP] * stride[NDIMS - LOOP];
        IdxT offsetR = baseOffset +
                        inCoord[NDIMS - LOOP + NDIMS] * stride[NDIMS - LOOP];
        LinearOffsetHelper<NDIMS, LOOP - 1, IdxT>::GetInputOffset(
            stride, inCoord, offsetL, inOffset);
        LinearOffsetHelper<NDIMS, LOOP - 1, IdxT>::GetInputOffset(
            stride, inCoord, offsetR, inOffset + POW2<LOOP - 1>::N);
    }
};

template <int NDIMS, typename IdxT>
struct LinearOffsetHelper<NDIMS, 1, IdxT> {
    static __device__ __forceinline__
    void GetInputOffset(const Array<IdxT, NDIMS> &stride,
                        const IdxT (&inCoord)[NDIMS * 2],
                        const IdxT &baseOffset,
                        IdxT *inOffset) {
        IdxT offsetL = baseOffset +
                        inCoord[NDIMS - 1] * stride[NDIMS - 1];
        IdxT offsetR = baseOffset +
                        inCoord[NDIMS - 1 + NDIMS] * stride[NDIMS - 1];
        inOffset[0] = offsetL;
        inOffset[1] = offsetR;
    }
};

/*
 * NDIMS: NDIMS-linear interpolation
 * IdxT: datatype of index and offset
 *
 * inCoord(input): L_D0, L_D1, L_D2... R_D0, R_D1, R_D2...
 *     L: coordinate round down
 *     R: coordinate round up
 * inOffset(output): offset of every input point associated
 *                   with the output point
 */
template <int NDIMS, typename IdxT>
__device__ __forceinline__
void LinearGetInputOffset(const Array<IdxT, NDIMS> &stride,
                          const IdxT (&inCoord)[NDIMS * 2],
                          const IdxT &batchOffset,
                          IdxT (&inOffset)[POW2<NDIMS>::N]) {
    LinearOffsetHelper<NDIMS, NDIMS, IdxT>::GetInputOffset(
        stride, inCoord, batchOffset, inOffset);
}

/*
 * pseudocode:
 * 
 * DATA_STRIDE = 1;
 * for (DIM_ITER = NDIMS; DIM_ITER >= 1; --DIM_ITER) {
 *     // LinearOutputHelper::GetOutput():
 *     for (int i = 0; i < POW2<DIM_ITER - 1>::N; ++i) {
 *         inputData[i * DATA_STRIDE * 2] =
 *             (inputData[i * DATA_STRIDE * 2 + DATA_STRIDE] -
 *             inputData[i * DATA_STRIDE * 2]) * weight[DIM_ITER - 1] +
 *             inputData[i * DATA_STRIDE * 2];
 *     }
 *     DATA_STRIDE *= 2;
 * }
 */
template <int DIM_ITER, int DATA_STRIDE, typename T>
struct LinearOutputHelper {
    static __device__ __forceinline__
    void GetOutput(const T *weight, T *inputData) {
        #pragma unroll
        for (int i = 0; i < POW2<DIM_ITER - 1>::N; ++i) {
            inputData[i * DATA_STRIDE * 2] =
                (inputData[i * DATA_STRIDE * 2 + DATA_STRIDE] -
                 inputData[i * DATA_STRIDE * 2]) *
                weight[DIM_ITER - 1] +
                inputData[i * DATA_STRIDE * 2];
        }

        LinearOutputHelper<DIM_ITER - 1, DATA_STRIDE * 2, T>
            ::GetOutput(weight, inputData);
    }
};

template <int DATA_STRIDE, typename T>
struct LinearOutputHelper<1, DATA_STRIDE, T> {
    static __device__ __forceinline__
    void GetOutput(const T *weight, T *inputData) {
        inputData[0] = (inputData[DATA_STRIDE] - inputData[0]) * weight[0] +
                       inputData[0];
    }
};

template <int NDIMS, typename T>
__device__ __forceinline__
T LinearGetOutput(const T (&weight)[NDIMS],
                  T (&inputData)[POW2<NDIMS>::N]) {
    LinearOutputHelper<NDIMS, 1, T>::GetOutput(weight, inputData);
    return inputData[0];
}

template <int BLOCK,            // thread block
          int UNROLL,           // number of output points processed by 1 thread
          int NDIMS,            // NDIMS-interpolation
          typename CoordFunc,   // coordinate mode functor
          typename IdxT,        // type of index and offset
          typename CompT,       // computation precision
          typename DT>
__global__ void LinearInterpolationKernel(
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

    IdxT globalTid = blockIdx.x * BLOCK + threadIdx.x;

    if (globalTid >= yWaveSize) {
        return;
    }

    // y point coordinate
    IdxT batchId = globalTid;
    IdxT yCoord[NDIMS];
    #pragma unroll
    for (int i = NDIMS - 1; i >= 0; --i) {
        auto dm = yDimDivMod[i].DivMod(batchId);
        yCoord[i] = dm.mod;
        batchId = dm.div;
    }

    // convert integer coordinate to float
    CompT xCoordFP[NDIMS];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        xCoordFP[i] = static_cast<CompT>(yCoord[i]);
    }

    // coordinate function
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        xCoordFP[i] = coordFunc[i].Coordinate(xCoordFP[i]);
    }

    // floor and ceil
    // xCoord: FLOOR_D0, FLOOR_D1, FLOOR_D2, ... CEIL_D0, CEIL_D1, CEIL_D2, ...
    IdxT xCoord[NDIMS * 2];
    CompT xCoordFloorFP[NDIMS];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        xCoordFloorFP[i] = cmath_floor(xCoordFP[i]);
        xCoord[i] = F2I_RD<IdxT>(xCoordFP[i]);
        xCoord[i + NDIMS] = F2I_RU<IdxT>(xCoordFP[i]);
        if (xCoord[i + NDIMS] > xCoordMax[i]) {
            xCoord[i + NDIMS] = xCoordMax[i];
        }
    }

    // interpolation weight
    CompT xWeight[NDIMS];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        xWeight[i] = xCoordFP[i] - xCoordFloorFP[i];
    }

    constexpr int N_INPUT = POW2<NDIMS>::N;

    // get offset of all input points
    IdxT xOffset[N_INPUT];
    LinearGetInputOffset<NDIMS, IdxT>(
        xStrides, xCoord, batchId * xBatchStride, xOffset);

    // load x point
    DT xData[UNROLL][N_INPUT];
    if (batchId + (UNROLL - 1) * waveBatch < batch) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            #pragma unroll
            for (int j = 0; j < N_INPUT; ++j) {
                Ldg<NC>(&xData[i][j], x + xOffset[j] + xWaveOffset * i);
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (batchId + i * waveBatch >= batch) {
                break;
            }
            #pragma unroll
            for (int j = 0; j < N_INPUT; ++j) {
                Ldg<NC>(&xData[i][j], x + xOffset[j] + xWaveOffset * i);
            }
        }
    }

    CompT xComp[UNROLL][N_INPUT];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < N_INPUT; ++j) {
            xComp[i][j] = static_cast<CompT>(xData[i][j]);
        }
    }

    // get y point
    CompT yData[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        yData[i] = LinearGetOutput<NDIMS, CompT>(xWeight, xComp[i]);
    }
    DT yStgReg[UNROLL];
    for (int i = 0; i < UNROLL; ++i) {
        yStgReg[i] = static_cast<DT>(yData[i]);
    }

    // write back
    DT *yPtr = y + globalTid;
    if (batchId + (UNROLL - 1) * waveBatch < batch) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yPtr[i * yWaveSize] = yStgReg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (batchId + i * waveBatch >= batch) {
                break;
            }
            yPtr[i * yWaveSize] = yStgReg[i];
        }
    }
}

template <int NDIMS,
          typename CoordFunc,
          typename CompT,
          typename ScaleT,
          typename T>
hiednnStatus_t LaunchLinearInterpolation(
        const HiednnCudaHandle &handle,
        const int64_t *xDims,
        const int64_t *xStrides,
        const T *x,
        const ScaleT *scale,
        const int64_t *yDims,
        const int64_t *yStrides,
        T *y) {
    const int X_POINTS = POW2<NDIMS>::N;
    // number of registers for X_POINTS
    const int X_REGS = sizeof(T) < 4 ?
                       X_POINTS : X_POINTS * sizeof(T) / sizeof(uint32_t);
    // for NDIMS-interpolation, 4*NDIMS SFU instructions for each output point
    const int SFU_INSTS = 4 * NDIMS;
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

    if (handle.deviceProp.major >= 6 && handle.deviceProp.minor == 0) {
        // configuration for GPUs with HBM memory
        // upper bound of SFU instructions for each output point on avarage
        const int SFU_BOUND = 2;

        const int UNROLL = X_REGS > 64 ? 1 :
            ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N * sizeof(T) > 8 ?
            ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N : 8 / sizeof(T);

        IdxT batch = xDims[0];
        IdxT waveBatch = UIntDivRU<IdxT>(batch, UNROLL);
        IdxT yWaveSize = waveBatch * yStrides[0];
        IdxT xWaveOffset = waveBatch * xStrides[0];
        IdxT xBatchStride = xStrides[0];

        IdxT grid = UIntDivRU<IdxT>(yWaveSize, BLOCK);

        LinearInterpolationKernel<BLOCK, UNROLL, NDIMS, CoordFunc, IdxT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            x, y, yDimDivMod, coordFunc, xStride, xCoordMax, batch,
            waveBatch, yWaveSize, xWaveOffset, xBatchStride);
    } else {
        // configuration for GPUs with GDDR memory
        // upper bound of SFU instructions for each output point on avarage
        const int SFU_BOUND = 6;

        const int UNROLL = X_REGS >= 64 ? 1 :
            ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N * sizeof(T) > 8 ?
            ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N : 8 / sizeof(T);

        IdxT batch = xDims[0];
        IdxT waveBatch = UIntDivRU<IdxT>(batch, UNROLL);
        IdxT yWaveSize = waveBatch * yStrides[0];
        IdxT xWaveOffset = waveBatch * xStrides[0];
        IdxT xBatchStride = xStrides[0];

        IdxT grid = UIntDivRU<IdxT>(yWaveSize, BLOCK);

        LinearInterpolationKernel<BLOCK, UNROLL, NDIMS, CoordFunc, IdxT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            x, y, yDimDivMod, coordFunc, xStride, xCoordMax, batch,
            waveBatch, yWaveSize, xWaveOffset, xBatchStride);
    }

    return HIEDNN_STATUS_SUCCESS;
}

template <typename CoordFunc, typename CompT, typename ScaleT, typename T>
hiednnStatus_t LinearInterpolation(
        const HiednnCudaHandle &handle,
        const HiednnTensorDesc &xDesc,
        const T *x,
        const ScaleT *scale,
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
            ret = LaunchLinearInterpolation<1, CoordFunc, CompT, ScaleT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 2:
            ret = LaunchLinearInterpolation<2, CoordFunc, CompT, ScaleT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 3:
            ret = LaunchLinearInterpolation<3, CoordFunc, CompT, ScaleT, T>(
                handle, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return ret;
}

template <typename T>
struct LinearInterpolationImpl {
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            hiednnInterpCoordMode_t coordMode,
            const HiednnTensorDesc &xDesc,
            const void *x,
            const void *scale,
            const HiednnTensorDesc &yDesc,
            void *y) {
        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);
        using CompT = typename ComputeType<T>::type;

        using ScaleT = typename ScaleType<T>::type;
        const ScaleT *scalePtr = static_cast<const ScaleT *>(scale);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (coordMode) {
            case HIEDNN_INTERP_COORD_HALF_PIXEL:
                ret = LinearInterpolation
                    <HalfPixel<RoundNon, CompT>, CompT, ScaleT, T>(
                    handle, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL:
                ret = LinearInterpolation
                    <PytorchHalfPixel<RoundNon, CompT>, CompT, ScaleT, T>(
                    handle, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_COORD_ALIGN_CORNER:
                ret = LinearInterpolation
                    <AlignCorner<RoundNon, CompT>, CompT, ScaleT, T>(
                    handle, xDesc, xPtr, scalePtr, yDesc, yPtr);
                break;
            case HIEDNN_INTERP_COORD_ASYMMETRIC:
                ret = LinearInterpolation
                    <Asymmetric<RoundNon, CompT>, CompT, ScaleT, T>(
                    handle, xDesc, xPtr, scalePtr, yDesc, yPtr);
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
hiednnCudaLinearInterpolation(HiednnCudaHandle *cudaHandle,
                              hiednnInterpCoordMode_t coordMode,
                              HiednnTensorDesc *xDesc,
                              const void *x,
                              const void *scale,
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

    return hiednn::DispatchFP<
        hiednn::cuda::interpolation::LinearInterpolationImpl>(
        xDesc->dataType, *cudaHandle, coordMode, *xDesc, x, scale, *yDesc, y);
}


