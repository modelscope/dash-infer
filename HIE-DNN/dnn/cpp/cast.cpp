/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cstddef>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <datatype_extension/datatype_extension.hpp>

#include <cpp/cpp_handle.hpp>

namespace hiednn {

namespace cpp {

namespace {

template <typename ST, typename DT>
void CastKernel(const ST *x, DT *y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = static_cast<DT>(x[i]);
    }
}

template <typename ST>
struct CastImpl {
    template <typename DT>
    struct CastDispatchDst {
        hiednnStatus_t operator()(const void *x,
                                  void *y,
                                  size_t n) {
            const ST *xPtr = static_cast<const ST *>(x);
            DT *yPtr = static_cast<DT *>(y);
            CastKernel<ST, DT>(xPtr, yPtr, n);
            return HIEDNN_STATUS_SUCCESS;
        }
    };

    hiednnStatus_t operator()(hiednnDataType_t dstType,
                              const void *x,
                              void *y,
                              size_t n) {
        return DispatchAll<CastDispatchDst>(dstType, x, y, n);
    }
};

}  // anonymous namespace

}  // namespace cpp

}  // namespace hiednn

hiednnStatus_t
hiednnCppCast(HiednnCppHandle *cppHandle,
              HiednnTensorDesc *xDesc,
              const void *x,
              HiednnTensorDesc *yDesc,
              void *y) {
    if (!hiednn::CheckNullptr(cppHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!xDesc->SameDimStride(*yDesc) || !xDesc->contiguous) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchAll<hiednn::cpp::CastImpl>(xDesc->dataType,
                                                      yDesc->dataType,
                                                      x, y, xDesc->size);
}


