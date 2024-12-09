/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    datatype_dispatch.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_DISPATCH_HPP_
#define DNN_INCLUDE_DATATYPE_DISPATCH_HPP_

#include <utility>
#include <cstdint>

#include <utils.hpp>
#include <datatype_extension/datatype_extension.hpp>

namespace hiednn {

// --------------------------------------------------------------
// dispatch for all types:
// FP64, FP32, FP16, BF16, INT64, UINT64, INT32, UINT32,
// INT16, UINT16, INT8, UINT8
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchAll(hiednnDataType_t dtype, Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (dtype) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = Functor<half>()(std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = Functor<bfloat16>()(std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = Functor<float>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = Functor<double>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT8:
            ret = Functor<std::int8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT8:
            ret = Functor<std::uint8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = Functor<std::int16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT16:
            ret = Functor<std::uint16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = Functor<std::int32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT32:
            ret = Functor<std::uint32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = Functor<std::int64_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT64:
            ret = Functor<std::uint64_t>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// --------------------------------------------------------------
// dispatch for floating point types:
// FP64, FP32, FP16, BF16
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchFP(hiednnDataType_t dtype, Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (dtype) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = Functor<half>()(std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = Functor<bfloat16>()(std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = Functor<float>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = Functor<double>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// --------------------------------------------------------------
// dispatch for integer types:
// INT64, UINT64, INT32, UINT32, INT16, UINT16, INT8, UINT8
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchInt(hiednnDataType_t dtype, Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (dtype) {
        case HIEDNN_DATATYPE_INT8:
            ret = Functor<std::int8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT8:
            ret = Functor<std::uint8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = Functor<std::int16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT16:
            ret = Functor<std::uint16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = Functor<std::int32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT32:
            ret = Functor<std::uint32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = Functor<std::int64_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT64:
            ret = Functor<std::uint64_t>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// --------------------------------------------------------------
// dispatch for signed types:
// FP64, FP32, FP16, BF16, INT64, INT32, INT16, INT8
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchSigned(hiednnDataType_t dtype, Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (dtype) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = Functor<half>()(std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = Functor<bfloat16>()(std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = Functor<float>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = Functor<double>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT8:
            ret = Functor<std::int8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = Functor<std::int16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = Functor<std::int32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = Functor<std::int64_t>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// --------------------------------------------------------------
// dispatch for integer types:
// UINT64, UINT32, UINT16, UINT8
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchUnsigned(hiednnDataType_t dtype, Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (dtype) {
        case HIEDNN_DATATYPE_UINT8:
            ret = Functor<std::uint8_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT16:
            ret = Functor<std::uint16_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT32:
            ret = Functor<std::uint32_t>()(std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT64:
            ret = Functor<std::uint64_t>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// --------------------------------------------------------------
// dispatch for OPs only care about the item size,
// such as data movement OP
// --------------------------------------------------------------
template <template <typename> class Functor, class... Args>
hiednnStatus_t DispatchItemSize(hiednnDataType_t dtype, Args&&... args) {
    size_t itemSize = ItemSizeInByte(dtype);

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (itemSize) {
        case 8:
            ret = Functor<std::uint64_t>()(std::forward<Args>(args) ...);
            break;
        case 4:
            ret = Functor<std::uint32_t>()(std::forward<Args>(args) ...);
            break;
        case 2:
            ret = Functor<std::uint16_t>()(std::forward<Args>(args) ...);
            break;
        case 1:
            ret = Functor<std::uint8_t>()(std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
            break;
    }

    return ret;
}

}  // namespace hiednn

#endif  // DNN_INCLUDE_DATATYPE_DISPATCH_HPP_


