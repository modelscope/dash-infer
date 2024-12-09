/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    expand.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cstddef>
#include <cstdint>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>

#include <cpp/cpp_handle.hpp>

namespace hiednn {

namespace cpp {

namespace {

template <typename T>
void ExpandKernel(
        const T *x, T *y, size_t ySize, int xNDim,
        Array<U32DivMod, TENSOR_DIM_MAX> yDivMod,
        Array<uint32_t, TENSOR_DIM_MAX> xStrides) {
    for (uint32_t yOffset = 0; yOffset < ySize; ++yOffset) {
        uint32_t xOffset = 0;
        uint32_t offset = yOffset;
        for (int i = 0; i < xNDim; ++i) {
            auto dm = yDivMod[i].DivMod(offset);
            uint32_t idx = dm.mod;
            offset = dm.div;
            xOffset += idx * xStrides[i];
        }
        y[yOffset] = x[xOffset];
    }
}

template <typename T>
struct ExpandImpl {
    hiednnStatus_t operator()(const HiednnTensorDesc &xDesc,
                              const void *x,
                              const HiednnTensorDesc &yDesc,
                              void *y) {
        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);

        int xNDim = xDesc.nDims;
        int yNDim = yDesc.nDims;
        const int64_t *pXDims = xDesc.dims;
        const int64_t *pYDims = yDesc.dims;
        const int64_t *pXStrides = xDesc.strides;

        // fast divmod only support unsigned integers <= INT32_MAX
        if (yDesc.size > UINT32_MAX ||
            U32DivMod::OutOfBound(pYDims, yNDim)) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        Array<uint32_t, TENSOR_DIM_MAX> xStrides;
        Array<U32DivMod, TENSOR_DIM_MAX> yDivMod;

        for (int i = 0; i < xNDim; ++i) {
            xStrides[i] = pXDims[xNDim - 1 - i] > 1 ?
                          pXStrides[xNDim - 1 - i] : 0;
            yDivMod[i] = U32DivMod(pYDims[yNDim - 1 - i]);
        }

        ExpandKernel<T>(xPtr, yPtr, yDesc.size, xNDim, yDivMod, xStrides);

        return HIEDNN_STATUS_SUCCESS;
    }
};

}  // anonymous namespace

}  // namespace cpp

}  // namespace hiednn

hiednnStatus_t
hiednnCppExpand(HiednnCppHandle *cppHandle,
                HiednnTensorDesc *xDesc,
                const void *x,
                HiednnTensorDesc *yDesc,
                void *y) {
    if (!hiednn::CheckNullptr(cppHandle, xDesc, yDesc) ||
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

    return hiednn::DispatchItemSize<hiednn::cpp::ExpandImpl>(
               xDesc->dataType, *xDesc, x, *yDesc, y);
}


