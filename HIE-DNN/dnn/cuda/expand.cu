/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    expand.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>

#include <cuda/cuda_handle.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <uint32_t BLOCK, uint32_t UNROLL, uint32_t MAX_XNDIMS, typename T>
__global__ void ExpandKernel(
        const T *__restrict__ x,
        T *y,
        uint32_t ySize,
        Array<U32DivMod, MAX_XNDIMS> yDivMod,
        Array<uint32_t, MAX_XNDIMS> xStrides) {
    uint32_t yOffset = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    uint32_t xOffset[UNROLL];
    uint32_t divTmp[UNROLL];

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xOffset[i] = 0;
        divTmp[i] = yOffset + i * BLOCK;
    }

    #pragma unroll
    for (int i = 0; i < MAX_XNDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            auto dm = yDivMod[i].DivMod(divTmp[j]);
            uint32_t idx = dm.mod;
            divTmp[j] = dm.div;
            xOffset[j] += idx * xStrides[i];
        }
    }

    T xReg[UNROLL];

    if (yOffset + (UNROLL - 1) * BLOCK < ySize) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xReg[i] = x[xOffset[i]];
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            y[yOffset + i * BLOCK] = xReg[i];
        }
    } else {
        uint32_t yCount = ySize > yOffset ?
                          UIntDivRU(ySize - yOffset, BLOCK) : 0;
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                xReg[i] = x[xOffset[i]];
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                y[yOffset + i * BLOCK] = xReg[i];
            }
        }
    }
}

template <uint32_t MAX_XNDIMS, typename T>
hiednnStatus_t
LaunchExpandKernel(
        const T *x, T *y, size_t ySize, int xNDim, int yNDim,
        const int64_t *xDims, const int64_t *xStrides, const int64_t *yDims,
        cudaStream_t stream) {
    // U32DivMod only work for integers from 0 to INT32_MAX (2^32 - 1)
    if (ySize > UINT32_MAX || U32DivMod::OutOfBound(yDims, yNDim)) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    Array<uint32_t, MAX_XNDIMS> xStridesCmem;
    Array<U32DivMod, MAX_XNDIMS> yDivMod;

    for (int i = 0; i < xNDim; ++i) {
        xStridesCmem[i] = xDims[xNDim - 1 - i] > 1 ?
                          xStrides[xNDim - 1 - i] : 0;
        yDivMod[i] = U32DivMod(yDims[yNDim - 1 - i]);
    }
    for (int i = xNDim; i < MAX_XNDIMS; ++i) {
        xStridesCmem[i] = 0;
        yDivMod[i] = U32DivMod(1);
    }

    // launch kernel
    const uint32_t BLOCK_SIZE = 128;
    const uint32_t UNROLLED_BYTE = 16;
    const uint32_t UNROLL = UNROLLED_BYTE / sizeof(T);

    uint32_t nBlock = UIntDivRU<uint32_t>(ySize, BLOCK_SIZE * UNROLL);

    ExpandKernel<BLOCK_SIZE, UNROLL, MAX_XNDIMS>
        <<<nBlock, BLOCK_SIZE, 0, stream>>>(x, y, ySize, yDivMod, xStridesCmem);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct ExpandImpl {
    hiednnStatus_t operator()(const HiednnTensorDesc &xDesc,
                              const void *x,
                              const HiednnTensorDesc &yDesc,
                              void *y,
                              cudaStream_t stream) {
        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);

        size_t ySize = yDesc.size;
        int64_t xNDim = xDesc.nDims;
        int64_t yNDim = yDesc.nDims;
        const int64_t *pXDims = xDesc.dims;
        const int64_t *pXStrides = xDesc.strides;
        const int64_t *pYDims = yDesc.dims;

        static_assert(TENSOR_DIM_MAX > 6,
                      "cuda::ExpandImpl: invalid TENSOR_DIM_MAX");

        if (xNDim <= 2) {
            return LaunchExpandKernel<2, T>(
                xPtr, yPtr, ySize, xNDim, yNDim,
                pXDims, pXStrides, pYDims, stream);
        } else if (xNDim <= 4) {
            return LaunchExpandKernel<4, T>(
                xPtr, yPtr, ySize, xNDim, yNDim,
                pXDims, pXStrides, pYDims, stream);
        } else if (xNDim <= 6) {
            return LaunchExpandKernel<6, T>(
                xPtr, yPtr, ySize, xNDim, yNDim,
                pXDims, pXStrides, pYDims, stream);
        } else {
            return LaunchExpandKernel<TENSOR_DIM_MAX, T>(
                xPtr, yPtr, ySize, xNDim, yNDim,
                pXDims, pXStrides, pYDims, stream);
        }
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaExpand(HiednnCudaHandle *cudaHandle,
                 HiednnTensorDesc *xDesc,
                 const void *x,
                 HiednnTensorDesc *yDesc,
                 void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!xDesc->UniBroadcastableTo(*yDesc) ||
        xDesc->dataType != yDesc->dataType) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::ExpandImpl>(
               xDesc->dataType, *xDesc, x, *yDesc, y, cudaHandle->stream);
}
