/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cubic_interpolation.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <tensor_desc.hpp>
#include <utils.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <cmath_wrapper.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>
#include <cuda/intrinsic/type_conversion.hpp>
#include <cuda/intrinsic/fast_math.hpp>

#include "coordinate_functor.cuh"

namespace hiednn {

namespace cuda {

namespace interpolation {

template <int Y>
using POW4 = ConstExpr::Pow<4, Y>;

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

template <typename T>
__device__ __forceinline__
void CubicWeightInit(T (&weight)[4], const T &coord, const T &coeff) {
    T offset = coord - cmath_floor(coord);

    /*
     * r: offset
     * c: coeff
     * weight[0] = ((c * (r + 1) - 5 * c) * (r + 1) + 8 * c) * (r + 1) - 4 * c;
     * weight[1] = ((c + 2) * r - (c + 3)) * r * r + 1;
     * r = 1 - r;
     * weight[2] = ((c + 2) * r - (c + 3)) * r * r + 1;
     * weight[3] = ((c * (r + 1) - 5 * c) * (r + 1) + 8 * c) * (r + 1) - 4 * c;
     */
    T rr = offset * offset;
    T rrr = rr * offset;
    T cr = coeff * offset;
    T crr = coeff * rr;
    T crrr = coeff * rrr;

    weight[0] = crrr - 2 * crr + cr;
    weight[1] = crrr + 2 * rrr - crr - 3 * rr + 1;
    weight[2] = -cr + 2 * crr + 3 * rr - crrr - 2 * rrr;
    weight[3] = crr - crrr;
}

template <typename CoordT, typename CompT>
__device__ __forceinline__
void CubicWeightNormalize(CompT (&weight)[4],
                          CoordT coordStart,
                          CoordT coordMax) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (coordStart + i < 0 || coordStart + i > coordMax) {
            weight[i] = CompT(0);
        }
    }

    CompT weightAcc = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        weightAcc += weight[i];
    }

    CompT weightAccRcp = FastMath::FRcp(weightAcc);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        weight[i] *= weightAccRcp;
    }
}

// LOOP init: NDIMS
template <int NDIMS, int LOOP, typename IdxT, typename CompT, typename DT>
struct CubicLoader {
    static __device__ __forceinline__
    void Load(const DT *x,
              const IdxT (&xOffset)[NDIMS][4],
              const CompT (&weight)[4],
              CompT *yData) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            CubicLoader<NDIMS, LOOP - 1, IdxT, CompT, DT>::Load(
                x + xOffset[NDIMS - LOOP][i], xOffset, weight,
                yData + i * POW4<LOOP - 2>::N);
        }
    }
};

template <int NDIMS, typename IdxT, typename CompT, typename DT>
struct CubicLoader<NDIMS, 1, IdxT, CompT, DT> {
    static __device__ __forceinline__
    void Load(const DT *x,
              const IdxT (&xOffset)[NDIMS][4],
              const CompT (&weight)[4],
              CompT *yData) {
        DT xData[4];
        Ldg<NC>(&xData[0], x + xOffset[NDIMS - 1][0]);
        Ldg<NC>(&xData[1], x + xOffset[NDIMS - 1][1]);
        Ldg<NC>(&xData[2], x + xOffset[NDIMS - 1][2]);
        Ldg<NC>(&xData[3], x + xOffset[NDIMS - 1][3]);

        *yData = static_cast<CompT>(xData[0]) * weight[0] +
                 static_cast<CompT>(xData[1]) * weight[1] +
                 static_cast<CompT>(xData[2]) * weight[2] +
                 static_cast<CompT>(xData[3]) * weight[3];
    }
};

template <int NDIMS, typename IdxT, typename CompT, typename DT>
__device__ __forceinline__ void CubicLoadData(
        const DT *x,
        const IdxT (&xOffset)[NDIMS][4],
        const CompT (&weight)[4],
        CompT (&yData)[POW4<NDIMS - 1>::N]) {
    CubicLoader<NDIMS, NDIMS, IdxT, CompT, DT>::Load(x, xOffset, weight, yData);
}

// NDIMS >= 2
// LOOP init: NDIMS - 1
template <int NDIMS, int LOOP, typename T>
struct CubicOutputHelper {
    static_assert(NDIMS >= 2, "CubicOutputHelper: invalid NDIMS");

    static __device__ __forceinline__
    void GetOutput(const T (&weight)[NDIMS][4], T *yData) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            CubicOutputHelper<NDIMS, LOOP - 1, T>::GetOutput(
                weight, yData + i * POW4<LOOP - 1>::N);
        }

        yData[0] = yData[0 * POW4<LOOP - 1>::N] * weight[NDIMS - LOOP - 1][0] +
                   yData[1 * POW4<LOOP - 1>::N] * weight[NDIMS - LOOP - 1][1] +
                   yData[2 * POW4<LOOP - 1>::N] * weight[NDIMS - LOOP - 1][2] +
                   yData[3 * POW4<LOOP - 1>::N] * weight[NDIMS - LOOP - 1][3];
    }
};

template <int NDIMS, typename T>
struct CubicOutputHelper<NDIMS, 1, T> {
    static_assert(NDIMS >= 2, "CubicOutputHelper: invalid NDIMS");

    static __device__ __forceinline__
    void GetOutput(const T (&weight)[NDIMS][4], T *yData) {
        yData[0] = yData[0] * weight[NDIMS - 2][0] +
                   yData[1] * weight[NDIMS - 2][1] +
                   yData[2] * weight[NDIMS - 2][2] +
                   yData[3] * weight[NDIMS - 2][3];
    }
};

template <typename T>
struct CubicOutputHelper<1, 0, T> {
    static __device__ __forceinline__
    void GetOutput(const T (&weight)[1][4], T *yData) {}
};

template <int NDIMS, typename T>
__device__ __forceinline__ void CubicGetOutput(
        const T (&weight)[NDIMS][4],
        T (&yData)[POW4<NDIMS - 1>::N]) {
    CubicOutputHelper<NDIMS, NDIMS - 1, T>::GetOutput(weight, yData);
}

template <int BLOCK,            // thread block
          int BATCH_UNROLL,     // unrolling factor on batch dimension
          int IMAGE_UNROLL,     // unrolling factor on DHW... dimension
          int NDIMS,            // NDIMS-interpolation
          bool EXCLUDE_OUTSIDE,
          typename CoordFunc,   // coordinate mode functor
          typename IdxT,        // type of index and offset
          typename CoordT,      // type of signed coordinate
          typename CompT,       // computation precision
          typename DT>
__global__ void CubicInterpolationKernel(
        const DT *x,
        DT *y,
        CompT coeff,
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

    // CoordT should be signed integer for out-of-bound check
    static_assert(SameType<CoordT, int8_t>::Same ||
                  SameType<CoordT, int16_t>::Same ||
                  SameType<CoordT, int32_t>::Same ||
                  SameType<CoordT, int64_t>::Same,
                  "CoordT should be signed integer");

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
    CompT xCoordFP[IMAGE_UNROLL][NDIMS];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < NDIMS; ++j) {
            xCoordFP[i][j] = static_cast<CompT>(yCoord[i][j]);
        }
    }

    // coordinate function
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            xCoordFP[j][i] = coordFunc[i].Coordinate<false>(xCoordFP[j][i]);
        }
    }

    // coordinate of the 1'st x point for each dimension
    CoordT xCoordStart[IMAGE_UNROLL][NDIMS];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < NDIMS; ++j) {
            xCoordStart[i][j] = F2I_RD<CoordT>(xCoordFP[i][j]) - 1;
        }
    }

    // cubic interpolation weights
    CompT weight[IMAGE_UNROLL][NDIMS][4];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            CubicWeightInit(weight[j][i], xCoordFP[j][i], coeff);
            if (EXCLUDE_OUTSIDE) {
                CubicWeightNormalize<CoordT, CompT>(
                    weight[j][i], xCoordStart[j][i], xCoordMax[i]);
            }
        }
    }

    // ldg coordinate
    CoordT xCoord[IMAGE_UNROLL][NDIMS][4];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                xCoord[j][i][k] = xCoordStart[j][i] + k;
                if (xCoord[j][i][k] < 0) {
                    xCoord[j][i][k] = 0;
                }
                if (xCoord[j][i][k] > xCoordMax[i]) {
                    xCoord[j][i][k] = xCoordMax[i];
                }
            }
        }
    }

    // ldg offset
    IdxT xOffset[IMAGE_UNROLL][NDIMS][4];
    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                xOffset[j][i][k] = xCoord[j][i][k] * xStrides[i];
            }
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

    // register buffer for calculate y points
    CompT yData[IMAGE_UNROLL][BATCH_UNROLL][POW4<NDIMS - 1>::N];

    // load x point and calculate cubic interpolation of the NDIM-1'th dimension
    #pragma unroll
    for (int imageIt = 0; imageIt < IMAGE_UNROLL; ++imageIt) {
        if (outOfBound[imageIt]) {
            break;
        }

        const DT *xPtr = x + batchId[imageIt] * xBatchStride;

        if (batchId[imageIt] + (BATCH_UNROLL - 1) * waveBatch < batch) {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                CubicLoadData<NDIMS, IdxT, CompT, DT>(
                    xPtr + xWaveOffset * i, xOffset[imageIt],
                    weight[imageIt][NDIMS - 1], yData[imageIt][i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                if (batchId[imageIt] + i * waveBatch >= batch) {
                    break;
                }
                CubicLoadData<NDIMS, IdxT, CompT, DT>(
                    xPtr + xWaveOffset * i, xOffset[imageIt],
                    weight[imageIt][NDIMS - 1], yData[imageIt][i]);
            }
        }
    }

    // calculate cubic interpolation of 0 ~ NDIM-2 dimension
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < BATCH_UNROLL; ++j) {
            CubicGetOutput<NDIMS, CompT>(weight[i], yData[i][j]);
        }
    }

    // convert to DT and store to GMEM
    DT yStgReg[IMAGE_UNROLL][BATCH_UNROLL];
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        #pragma unroll
        for (int j = 0; j < BATCH_UNROLL; ++j) {
            yStgReg[i][j] = static_cast<DT>(yData[i][j][0]);
        }
    }

    DT *yPtr = y + threadOffset;
    #pragma unroll
    for (int imageIt = 0; imageIt < IMAGE_UNROLL; ++imageIt) {
        if (outOfBound[imageIt]) {
            break;
        }

        if (batchId[imageIt] + (BATCH_UNROLL - 1) * waveBatch < batch) {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                yPtr[imageIt * BLOCK + i * yWaveSize] = yStgReg[imageIt][i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BATCH_UNROLL; ++i) {
                if (batchId[imageIt] + i * waveBatch >= batch) {
                    break;
                }
                yPtr[imageIt * BLOCK + i * yWaveSize] = yStgReg[imageIt][i];
            }
        }
    }
}

template <int NDIMS,
          bool EXCLUDE_OUTSIDE,
          typename CoordFunc,
          typename CompT,
          typename ScaleT,
          typename T>
hiednnStatus_t LaunchCubicInterpolation(
        const HiednnCudaHandle &handle,
        ScaleT coeff,
        const int64_t *xDims,
        const int64_t *xStrides,
        const T *x,
        const ScaleT *scale,
        const int64_t *yDims,
        const int64_t *yStrides,
        T *y) {
    // 3*NDIMS SFU instructions for each output point for normal mode,
    // 4*NDIMS SFU instructions for each output point for EXCLUDE_OUTSIDE mode.
    const int SFU_INSTS = EXCLUDE_OUTSIDE ? 4 * NDIMS : 3 * NDIMS;
    const int BLOCK = 128;

    using IdxT = uint32_t;
    using CoordT = int32_t;

    // approximate registers usage:
    // 4: ldg buffer
    // 4^(NDIMS-1): cubic interpolation buffer
    // 4*NDIMS: cubic weights
    const int DATA_BUFFER = 4 + POW4<NDIMS - 1>::N + 4 * NDIMS;
    const int DATA_REGS = sizeof(T) <= sizeof(uint32_t) ?
                          DATA_BUFFER :
                          DATA_BUFFER * sizeof(T) / sizeof(uint32_t);

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
    const int SFU_BOUND = 5;
    const int BATCH_UNROLL = DATA_REGS > 128 ? 1 :
                             ConstExpr::DivRU<SFU_INSTS, SFU_BOUND>::N;
    const int IMAGE_UNROLL = DATA_REGS > 128 ? 1 :
                             BATCH_UNROLL * sizeof(T) > 8 ? 1 :
                             8 / sizeof(T) / BATCH_UNROLL;

    IdxT batch = xDims[0];
    IdxT waveBatch = UIntDivRU<IdxT>(batch, BATCH_UNROLL);
    IdxT yWaveSize = waveBatch * yStrides[0];
    IdxT xWaveOffset = waveBatch * xStrides[0];
    IdxT xBatchStride = xStrides[0];

    IdxT grid = UIntDivRU<IdxT>(yWaveSize, BLOCK * IMAGE_UNROLL);

    CubicInterpolationKernel
        <BLOCK, BATCH_UNROLL, IMAGE_UNROLL, NDIMS,
         EXCLUDE_OUTSIDE, CoordFunc, IdxT, CoordT, T>
        <<<grid, BLOCK, 0, handle.stream>>>(
        x, y, static_cast<CompT>(coeff), yDimDivMod, coordFunc, xStride,
        xCoordMax, batch, waveBatch, yWaveSize, xWaveOffset, xBatchStride);

    return HIEDNN_STATUS_SUCCESS;
}

template <bool EXCLUDE_OUTSIDE,
          typename CoordFunc,
          typename CompT,
          typename ScaleT,
          typename T>
hiednnStatus_t CubicInterpolation(
        const HiednnCudaHandle &handle,
        ScaleT coeff,
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
            ret = LaunchCubicInterpolation
                <1, EXCLUDE_OUTSIDE, CoordFunc, CompT, ScaleT, T>(
                handle, coeff, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 2:
            ret = LaunchCubicInterpolation
                <2, EXCLUDE_OUTSIDE, CoordFunc, CompT, ScaleT, T>(
                handle, coeff, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        case 3:
            ret = LaunchCubicInterpolation
                <3, EXCLUDE_OUTSIDE, CoordFunc, CompT, ScaleT, T>(
                handle, coeff, xDims, xStrides, x, scale, yDims, yStrides, y);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return ret;
}

template <typename T>
struct CubicInterpolationImpl {
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            hiednnInterpCoordMode_t coordMode,
            const void *cubicCoefficient,
            int excludeOutside,
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
        ScaleT coeff = *static_cast<const ScaleT *>(cubicCoefficient);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;

        if (excludeOutside == 0) {
            switch (coordMode) {
                case HIEDNN_INTERP_COORD_HALF_PIXEL:
                    ret = CubicInterpolation
                        <false, HalfPixel<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL:
                    ret = CubicInterpolation
                        <false, PytorchHalfPixel<RoundNon, CompT>,
                         CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_ALIGN_CORNER:
                    ret = CubicInterpolation
                        <false, AlignCorner<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_ASYMMETRIC:
                    ret = CubicInterpolation
                        <false, Asymmetric<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                default:
                    ret = HIEDNN_STATUS_INVALID_PARAMETER;
                    break;
            }
        } else {
            switch (coordMode) {
                case HIEDNN_INTERP_COORD_HALF_PIXEL:
                    ret = CubicInterpolation
                        <true, HalfPixel<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL:
                    ret = CubicInterpolation
                        <true, PytorchHalfPixel<RoundNon, CompT>,
                         CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_ALIGN_CORNER:
                    ret = CubicInterpolation
                        <true, AlignCorner<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                case HIEDNN_INTERP_COORD_ASYMMETRIC:
                    ret = CubicInterpolation
                        <true, Asymmetric<RoundNon, CompT>, CompT, ScaleT, T>(
                        handle, coeff, xDesc, xPtr, scalePtr, yDesc, yPtr);
                    break;
                default:
                    ret = HIEDNN_STATUS_INVALID_PARAMETER;
                    break;
            }
        }

        return ret;
    }
};

}  // namespace interpolation

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaCubicInterpolation(HiednnCudaHandle *cudaHandle,
                             hiednnInterpCoordMode_t coordMode,
                             const void *cubicCoefficient,
                             int excludeOutside,
                             HiednnTensorDesc *xDesc,
                             const void *x,
                             const void *scale,
                             int scaleSize,
                             HiednnTensorDesc *yDesc,
                             void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, cubicCoefficient, xDesc, yDesc) ||
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

    if (excludeOutside != 0 && excludeOutside != 1) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchFP<
        hiednn::cuda::interpolation::CubicInterpolationImpl>(
        xDesc->dataType, *cudaHandle, coordMode, cubicCoefficient,
        excludeOutside, *xDesc, x, scale, *yDesc, y);
}


