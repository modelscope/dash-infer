/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tensor_desc.hpp
 */

#ifndef DNN_INCLUDE_TENSOR_DESC_HPP_
#define DNN_INCLUDE_TENSOR_DESC_HPP_

#include <hiednn.h>

#include <cstddef>
#include <cstdint>

constexpr int TENSOR_DIM_MAX = HIEDNN_DIM_MAX;

struct HiednnTensorDesc {
    int64_t dims[TENSOR_DIM_MAX];
    int64_t strides[TENSOR_DIM_MAX];
    size_t size;
    int nDims;
    hiednnDataType_t dataType;
    hiednnTensorFormat_t tensorFormat;
    bool contiguous;

    bool operator==(const HiednnTensorDesc &desc) const {
        if (desc.nDims != nDims ||
            desc.size != size ||
            desc.dataType != dataType ||
            desc.tensorFormat != tensorFormat) {
            return false;
        }
        bool ret = true;
        for (int i = 0; i < nDims; ++i) {
            ret = ret && (desc.dims[i] == dims[i]);
            ret = ret && (desc.strides[i] == strides[i]);
        }
        return ret;
    }

    bool operator!=(const HiednnTensorDesc &desc) const {
        return !(*this == desc);
    }

    bool SameDim(const HiednnTensorDesc &desc) const {
        if (nDims != desc.nDims) {
            return false;
        }
        bool ret = true;
        for (int i = 0; i < nDims; ++i) {
            ret = ret && (desc.dims[i] == dims[i]);
        }
        return ret;
    }

    bool SameStride(const HiednnTensorDesc &desc) const {
        if (nDims != desc.nDims) {
            return false;
        }
        bool ret = true;
        for (int i = 0; i < nDims; ++i) {
            ret = ret && (desc.strides[i] == strides[i]);
        }
        return ret;
    }

    bool SameDimStride(const HiednnTensorDesc &desc) const {
        return SameDim(desc) && SameStride(desc);
    }

    bool IsVectC() const {
        if (nDims == 5 && (
            tensorFormat == HIEDNN_TENSORFORMAT_NcHWc4 ||
            tensorFormat == HIEDNN_TENSORFORMAT_NcHWc32)) {
            return true;
        } else {
            return false;
        }
    }

    bool Broadcastable(const HiednnTensorDesc &desc) const {
        int minDim = nDims > desc.nDims ? desc.nDims : nDims;
        for (int i = nDims - minDim, j = desc.nDims - minDim;
             i < nDims; ++i, ++j) {
            if (dims[i] != desc.dims[j] && dims[i] != 1 && desc.dims[j] != 1) {
                return false;
            }
        }
        return true;
    }

    // return true only if @*this can broadcast to @desc
    bool UniBroadcastableTo(const HiednnTensorDesc &desc) const {
        if (nDims > desc.nDims) {
            return false;
        }
        for (int i = 0, j = desc.nDims - nDims; i < nDims; ++i, ++j) {
            if (dims[i] != desc.dims[j] && dims[i] != 1) {
                return false;
            }
        }
        return true;
    }

    bool IsIntegral() const {
        switch (dataType) {
            case HIEDNN_DATATYPE_INT8:
            case HIEDNN_DATATYPE_UINT8:
            case HIEDNN_DATATYPE_INT16:
            case HIEDNN_DATATYPE_UINT16:
            case HIEDNN_DATATYPE_INT32:
            case HIEDNN_DATATYPE_UINT32:
            case HIEDNN_DATATYPE_INT64:
            case HIEDNN_DATATYPE_UINT64:
                return true;
            default:
                return false;
        }
    }
};

#endif  // DNN_INCLUDE_TENSOR_DESC_HPP_

