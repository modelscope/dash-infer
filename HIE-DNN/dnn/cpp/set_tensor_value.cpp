/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    set_tensor_value.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cstddef>
#include <cstring>

#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>

#include <cpp/cpp_handle.hpp>

namespace hiednn {

namespace cpp {

namespace {

template <typename T>
void SetTensorValueConstKernel(T *y, size_t n, T value) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = value;
    }
}

template <typename T>
void SetTensorValueRangeKernel(T *y, size_t n, T start, T delta) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = start + i * delta;
    }
}

template <typename T>
void SetTensorValueDiagonalKernel(T *y,
                                  T value,
                                  int64_t firstRow,
                                  int64_t lastRow,
                                  int64_t stride0,
                                  int rShift) {
    for (int64_t rowIdx = firstRow; rowIdx < lastRow; ++rowIdx) {
        y[rowIdx * stride0 + (rowIdx + rShift)] = value;
    }
}

template <typename T>
hiednnStatus_t
SetTensorValueDiagonal(int rShift,
                       T value,
                       const HiednnTensorDesc &yDesc,
                       T *y) {
    std::memset(y, 0, yDesc.size * sizeof(T));

    int64_t m = yDesc.dims[0];
    int64_t n = yDesc.dims[1];
    if (rShift < 0) {
        m += rShift;  // m -= abs(rShift)
    } else {
        n -= rShift;
    }

    int64_t nnz = std::min(m, n);
    if (nnz <= 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    int64_t firstRow = rShift < 0 ? -rShift : 0;
    int64_t lastRow = firstRow + nnz;
    SetTensorValueDiagonalKernel(
        y, value, firstRow, lastRow, yDesc.dims[1], rShift);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct SetTensorValueConstImpl {
    hiednnStatus_t operator()(const void *valuePtr,
                              const HiednnTensorDesc &yDesc,
                              void *y) {
        T *y_ptr = static_cast<T *>(y);
        T value = *static_cast<const T *>(valuePtr);

        SetTensorValueConstKernel<T>(y_ptr, yDesc.size, value);

        return HIEDNN_STATUS_SUCCESS;
    }
};

template <typename T>
struct SetTensorValueRangeImpl {
    hiednnStatus_t operator()(const void *pStart,
                              const void *pDelta,
                              const HiednnTensorDesc &yDesc,
                              void *y) {
        if (yDesc.nDims != 1) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        T start = *static_cast<const T *>(pStart);
        T delta = *static_cast<const T *>(pDelta);
        T *y_ptr = static_cast<T *>(y);

        if (yDesc.nDims != 1) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        SetTensorValueRangeKernel<T>(y_ptr, yDesc.size, start, delta);

        return HIEDNN_STATUS_SUCCESS;
    }
};

template <typename T>
struct SetTensorValueDiagonalImpl {
    hiednnStatus_t operator()(int rShift,
                              const void *valuePtr,
                              const HiednnTensorDesc &yDesc,
                              void *y) {
        if (yDesc.nDims != 2 ||
            std::abs(rShift) >= yDesc.dims[0]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        T *y_ptr = static_cast<T *>(y);
        T value = *static_cast<const T *>(valuePtr);

        return SetTensorValueDiagonal(rShift, value, yDesc, y_ptr);
    }
};

}  // anonymous namespace

}  // namespace cpp

}  // namespace hiednn

hiednnStatus_t
hiednnCppSetTensorValue(HiednnCppHandle *cppHandle,
                        hiednnSetTensorValueMode_t mode,
                        const void *p0,
                        const void *p1,
                        hiednnTensorDesc_t yDesc,
                        void *y) {
    if (!hiednn::CheckNullptr(cppHandle, yDesc) ||
        !hiednn::CheckTensorPtr(*yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (mode) {
        case HIEDNN_SETTENSOR_CONST:
            ret = hiednn::DispatchItemSize<
                  hiednn::cpp::SetTensorValueConstImpl>(
                      yDesc->dataType, p0, *yDesc, y);
            break;
        case HIEDNN_SETTENSOR_RANGE:
            ret = hiednn::DispatchAll<
                  hiednn::cpp::SetTensorValueRangeImpl>(
                      yDesc->dataType, p0, p1, *yDesc, y);
            break;
        case HIEDNN_SETTENSOR_DIAGONAL:
            ret = hiednn::DispatchItemSize<
                  hiednn::cpp::SetTensorValueDiagonalImpl>(
                      yDesc->dataType, *static_cast<const int *>(p0),
                      p1, *yDesc, y);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_OPTYPE;
            break;
    }

    return ret;
}


