/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tensor_desc.cpp
 */

#include <hiednn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <tensor_desc.hpp>
#include <utils.hpp>

hiednnStatus_t
hiednnCreateTensorDesc(HiednnTensorDesc **desc) {
    *desc = new HiednnTensorDesc();
    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnDestroyTensorDesc(HiednnTensorDesc *desc) {
    if (desc != nullptr) {
        delete desc;
        return HIEDNN_STATUS_SUCCESS;
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }
}

hiednnStatus_t
hiednnSetTensorDesc(HiednnTensorDesc *desc,
                    hiednnDataType_t dataType,
                    int nDims,
                    const int64_t *dim,
                    const int64_t *stride) {
    if (nDims > TENSOR_DIM_MAX || nDims <= 0) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    memset(desc, 0, sizeof(HiednnTensorDesc));

    desc->nDims = nDims;
    desc->dataType = dataType;

    std::memcpy(desc->dims, dim, nDims * sizeof(int64_t));
    std::memcpy(desc->strides, stride, nDims * sizeof(int64_t));

    // set tensor format
    size_t size = 1;
    int64_t max_stride = -1;
    int max_stride_idx = -1;
    bool is_normal = true;

    for (int i = nDims - 1; i >= 0; --i) {
        if (stride[i] > max_stride) {
            max_stride = stride[i];
            max_stride_idx = i;
        }
        is_normal = is_normal && (static_cast<size_t>(stride[i]) == size);
        size *= static_cast<size_t>(dim[i]);
    }

    desc->size = size;
    desc->tensorFormat = is_normal? HIEDNN_TENSORFORMAT_NORMAL :
                                    HIEDNN_TENSORFORMAT_CUSTOMIZED;

    // is contiguous?
    desc->contiguous =
        static_cast<size_t>(max_stride * dim[max_stride_idx]) == size;

    // is NHWC?
    if (nDims == 4 && !is_normal) {
        if (stride[1] == 1 &&
            stride[3] == dim[1] &&
            stride[2] == dim[1] * dim[3] &&
            stride[0] == dim[1] * dim[3] * dim[2]) {
            desc->tensorFormat = HIEDNN_TENSORFORMAT_NHWC;
        }
    }

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnSetNormalTensorDesc(HiednnTensorDesc *desc,
                          hiednnDataType_t dataType,
                          int nDims,
                          const int64_t *dim) {
    if (nDims > TENSOR_DIM_MAX || nDims <= 0) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    std::memset(desc, 0, sizeof(HiednnTensorDesc));

    desc->nDims = nDims;
    desc->dataType = dataType;
    desc->tensorFormat = HIEDNN_TENSORFORMAT_NORMAL;
    desc->contiguous = true;
    std::memcpy(desc->dims, dim, nDims * sizeof(int64_t));

    int64_t size = 1;
    for (int i = nDims - 1; i >= 0; --i) {
        desc->strides[i] = size;
        size *= dim[i];
    }

    desc->size = static_cast<size_t>(size);

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnSet4dTensorDesc(HiednnTensorDesc *desc,
                      hiednnDataType_t dataType,
                      hiednnTensorFormat_t tensorFormat,
                      int64_t n,
                      int64_t c,
                      int64_t h,
                      int64_t w) {
    if (tensorFormat == HIEDNN_TENSORFORMAT_CUSTOMIZED) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    std::memset(desc, 0, sizeof(HiednnTensorDesc));

    desc->nDims = 4;

    desc->dims[0] = n;
    desc->dims[1] = c;
    desc->dims[2] = h;
    desc->dims[3] = w;

    desc->size = static_cast<size_t>(n * c * h * w);
    desc->dataType = dataType;
    desc->tensorFormat = tensorFormat;
    desc->contiguous = true;

    switch (tensorFormat) {
        case HIEDNN_TENSORFORMAT_NORMAL:
            desc->strides[0] = c * h * w;
            desc->strides[1] = h * w;
            desc->strides[2] = w;
            desc->strides[3] = 1;
            break;
        case HIEDNN_TENSORFORMAT_NHWC:
            desc->strides[0] = h * w * c;
            desc->strides[1] = 1;
            desc->strides[2] = w * c;
            desc->strides[3] = c;
            break;
        case HIEDNN_TENSORFORMAT_NcHWc4:
            // consider packed C as 5'st demension
            desc->nDims = 5;
            if (c % 4 != 0) {
                return HIEDNN_STATUS_INVALID_PARAMETER;
            }

            desc->dims[1] /= 4;
            desc->dims[4] = 4;
            desc->strides[0] = c * h * w;
            desc->strides[1] = h * w * 4;
            desc->strides[2] = w * 4;
            desc->strides[3] = 4;
            desc->strides[4] = 1;
            break;
        case HIEDNN_TENSORFORMAT_NcHWc32:
            desc->nDims = 5;
            if (c % 32 != 0) {
                return HIEDNN_STATUS_INVALID_PARAMETER;
            }

            desc->dims[1] /= 32;
            desc->dims[4] = 32;
            desc->strides[0] = c * h * w;
            desc->strides[1] = h * w * 32;
            desc->strides[2] = w * 32;
            desc->strides[3] = 32;
            desc->strides[4] = 1;
            break;
        default:
            // tensorFormat should be specified.
            return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnGet4dTensorDesc(
        HiednnTensorDesc * const tensorDesc,
        hiednnDataType_t *dataType,
        hiednnTensorFormat_t *tensorFormat,
        int64_t *n,
        int64_t *c,
        int64_t *h,
        int64_t *w) {
    if (!hiednn::CheckNullptr(n, c, h, w, dataType, tensorFormat, tensorDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (tensorDesc->nDims == 4) {
        // normal 4d-tensor.
        *dataType = tensorDesc->dataType;
        *tensorFormat = tensorDesc->tensorFormat;
        *n = tensorDesc->dims[0];
        *c = tensorDesc->dims[1];
        *h = tensorDesc->dims[2];
        *w = tensorDesc->dims[3];
    } else if (tensorDesc->IsVectC()) {
        *dataType = tensorDesc->dataType;
        *tensorFormat = tensorDesc->tensorFormat;
        *n = tensorDesc->dims[0];
        *c = tensorDesc->dims[1] * tensorDesc->dims[4];
        *h = tensorDesc->dims[2];
        *w = tensorDesc->dims[3];
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnGetTensorDesc(
        HiednnTensorDesc * const tensorDesc,
        hiednnDataType_t *dataType,
        hiednnTensorFormat_t *tensorFormat,
        int requestDims,
        int *nDims,
        int64_t *dim,
        int64_t *stride) {
    if (!hiednn::CheckNullptr(
            tensorDesc, dataType, tensorFormat, nDims, dim, stride)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (requestDims <= 0) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    *dataType = tensorDesc->dataType;
    *tensorFormat = tensorDesc->tensorFormat;

    if (tensorDesc->IsVectC()) {
        // recover dims and strides for vect_c format
        int64_t dimsVectC[4];
        dimsVectC[0] = tensorDesc->dims[0];
        dimsVectC[1] = tensorDesc->dims[1] * tensorDesc->dims[4];
        dimsVectC[2] = tensorDesc->dims[2];
        dimsVectC[3] = tensorDesc->dims[3];

        int offset = requestDims < 4 ? requestDims : 4;
        *nDims = requestDims < 4 ? requestDims : 4;
        std::memcpy(dim, dimsVectC + offset, (*nDims) * sizeof(int64_t));
        std::memset(stride, 0, requestDims * sizeof(int64_t));
    } else {
        // return lower @requestDims dimensions
        int offset = requestDims < tensorDesc->nDims ?
                     tensorDesc->nDims - requestDims : 0;
        *nDims = requestDims < tensorDesc->nDims ?
                 requestDims : tensorDesc->nDims;
        std::memcpy(dim, tensorDesc->dims + offset,
                    (*nDims) * sizeof(int64_t));
        std::memcpy(stride, tensorDesc->strides + offset,
                    (*nDims) * sizeof(int64_t));
    }

    return HIEDNN_STATUS_SUCCESS;
}

