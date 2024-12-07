/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce_dispatch.hpp
 */

#ifndef DNN_CUDA_REDUCE_REDUCE_DISPATCH_HPP_
#define DNN_CUDA_REDUCE_REDUCE_DISPATCH_HPP_

#include <hiednn.h>
#include <utility>
#include <cstdint>
#include <utils.hpp>
#include <datatype_extension/datatype_extension.hpp>

namespace hiednn {

namespace cuda {

namespace reduce {

// -----------------------------------------------------
// Dispatch output datatype (DT)
// -----------------------------------------------------

/*
 * ST(CompT): DT, ...
 *
 * int8_t(int32_t): int8_t, int16_t, int32_t, half, bfloat16, float
 * uint8_t(uint32_t): uint8_t, uint16_t, uint32_t, half, bfloat16, float
 * int16_t(int32_t): int16_t, int32_t
 * uint16_t(uint32_t): uint16_t, uint32_t
 *
 * half(float): half, float
 * bfloat16(float): bfloat16, float
 *
 * ST=DT=CompT for other types
 */

// for all types except int8_t/uint8_t/int16_t/uint16_t/half/bfloat16
template <typename ST, typename Impl>
struct ReduceDispatchDT {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        if (dt != HiednnDataType<ST>::type) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        } else {
            return Impl().template Run<ST, ST, ST>(std::forward<Args>(args)...);
        }
    }
};

// int8_t
template <typename Impl>
struct ReduceDispatchDT<int8_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_INT8:
                ret = Impl().template Run<int8_t, int8_t, int32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_INT16:
                ret = Impl().template Run<int8_t, int16_t, int32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_INT32:
                ret = Impl().template Run<int8_t, int32_t, int32_t>(
                      std::forward<Args>(args)...);
                break;
#ifdef HIEDNN_USE_FP16
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<int8_t, half, int32_t>(
                      std::forward<Args>(args)...);
                break;
#endif
#ifdef HIEDNN_USE_BF16
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<int8_t, bfloat16, int32_t>(
                      std::forward<Args>(args)...);
                break;
#endif
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<int8_t, float, int32_t>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// uint8_t
template <typename Impl>
struct ReduceDispatchDT<uint8_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_UINT8:
                ret = Impl().template Run<uint8_t, uint8_t, uint32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_UINT16:
                ret = Impl().template Run<uint8_t, uint16_t, uint32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_UINT32:
                ret = Impl().template Run<uint8_t, uint32_t, uint32_t>(
                      std::forward<Args>(args)...);
                break;
#ifdef HIEDNN_USE_FP16
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<uint8_t, half, uint32_t>(
                      std::forward<Args>(args)...);
                break;
#endif
#ifdef HIEDNN_USE_BF16
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<uint8_t, bfloat16, uint32_t>(
                      std::forward<Args>(args)...);
                break;
#endif
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<uint8_t, float, uint32_t>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// int16_t
template <typename Impl>
struct ReduceDispatchDT<int16_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_INT16:
                ret = Impl().template Run<int16_t, int16_t, int32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_INT32:
                ret = Impl().template Run<int16_t, int32_t, int32_t>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// uint16_t
template <typename Impl>
struct ReduceDispatchDT<uint16_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_UINT16:
                ret = Impl().template Run<uint16_t, uint16_t, uint32_t>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_UINT32:
                ret = Impl().template Run<uint16_t, uint32_t, uint32_t>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// half
#ifdef HIEDNN_USE_FP16
template <typename Impl>
struct ReduceDispatchDT<half, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<half, half, float>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<half, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};
#endif

// bfloat16
#ifdef HIEDNN_USE_BF16
template <typename Impl>
struct ReduceDispatchDT<bfloat16, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<bfloat16, bfloat16, float>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<bfloat16, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};
#endif

/*
 * Dispatch for floating point output type
 *
 * ST(CompT): DT, ...
 *
 * int8_t/uint8_t(float): half, bfloat16, float
 * other integer types(float): float
 *
 * half(float): half, float
 * bfloat16(float): bfloat16, float
 * ST=DT=CompT for other floating point types
 */

// for all integer ST except int8_t/uint8_t
template <typename ST, typename Impl>
struct ReduceDispatchDTFP {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        if (dt != HIEDNN_DATATYPE_FP32) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        } else {
            return Impl().template Run<ST, float, float>(
                   std::forward<Args>(args)...);
        }
    }
};

// int8_t
template <typename Impl>
struct ReduceDispatchDTFP<int8_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
#ifdef HIEDNN_USE_FP16
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<int8_t, half, float>(
                      std::forward<Args>(args)...);
                break;
#endif
#ifdef HIEDNN_USE_BF16
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<int8_t, bfloat16, float>(
                      std::forward<Args>(args)...);
                break;
#endif
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<int8_t, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// uint8_t
template <typename Impl>
struct ReduceDispatchDTFP<uint8_t, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
#ifdef HIEDNN_USE_FP16
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<uint8_t, half, float>(
                      std::forward<Args>(args)...);
                break;
#endif
#ifdef HIEDNN_USE_BF16
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<uint8_t, bfloat16, float>(
                      std::forward<Args>(args)...);
                break;
#endif
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<uint8_t, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};

// half
#ifdef HIEDNN_USE_FP16
template <typename Impl>
struct ReduceDispatchDTFP<half, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_FP16:
                ret = Impl().template Run<half, half, float>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<half, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};
#endif

// bfloat16
#ifdef HIEDNN_USE_BF16
template <typename Impl>
struct ReduceDispatchDTFP<bfloat16, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (dt) {
            case HIEDNN_DATATYPE_BF16:
                ret = Impl().template Run<bfloat16, bfloat16, float>(
                      std::forward<Args>(args)...);
                break;
            case HIEDNN_DATATYPE_FP32:
                ret = Impl().template Run<bfloat16, float, float>(
                      std::forward<Args>(args)...);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }
        return ret;
    }
};
#endif

// float
template <typename Impl>
struct ReduceDispatchDTFP<float, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        if (dt != HIEDNN_DATATYPE_FP32) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        } else {
            return Impl().template Run<float, float, float>(
                   std::forward<Args>(args)...);
        }
    }
};

// double
template <typename Impl>
struct ReduceDispatchDTFP<double, Impl> {
    template <class... Args>
    hiednnStatus_t operator()(hiednnDataType_t dt, Args&&... args) {
        if (dt != HIEDNN_DATATYPE_FP64) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        } else {
            return Impl().template Run<double, double, double>(
                   std::forward<Args>(args)...);
        }
    }
};

// -----------------------------------------------------
// Dispatch input datatype (ST)
// -----------------------------------------------------

template <typename Impl, class... Args>
hiednnStatus_t ReduceDispatch(hiednnDataType_t st,
                              hiednnDataType_t dt,
                              Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (st) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = ReduceDispatchDT<half, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = ReduceDispatchDT<bfloat16, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = ReduceDispatchDT<float, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = ReduceDispatchDT<double, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT8:
            ret = ReduceDispatchDT<int8_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT8:
            ret = ReduceDispatchDT<uint8_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = ReduceDispatchDT<int16_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT16:
            ret = ReduceDispatchDT<uint16_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = ReduceDispatchDT<int32_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT32:
            ret = ReduceDispatchDT<uint32_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = ReduceDispatchDT<int64_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT64:
            ret = ReduceDispatchDT<uint64_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// dispatch for @st is signed type
template <typename Impl, class... Args>
hiednnStatus_t ReduceDispatchSigned(hiednnDataType_t st,
                                    hiednnDataType_t dt,
                                    Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (st) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = ReduceDispatchDT<half, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = ReduceDispatchDT<bfloat16, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = ReduceDispatchDT<float, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = ReduceDispatchDT<double, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT8:
            ret = ReduceDispatchDT<int8_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = ReduceDispatchDT<int16_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = ReduceDispatchDT<int32_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = ReduceDispatchDT<int64_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

// dispatch for @dt is floating point type
template <typename Impl, class... Args>
hiednnStatus_t ReduceDispatchFP(hiednnDataType_t st,
                                hiednnDataType_t dt,
                                Args&&... args) {
    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (st) {
#ifdef HIEDNN_USE_FP16
        case HIEDNN_DATATYPE_FP16:
            ret = ReduceDispatchDTFP<half, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
#ifdef HIEDNN_USE_BF16
        case HIEDNN_DATATYPE_BF16:
            ret = ReduceDispatchDTFP<bfloat16, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
#endif
        case HIEDNN_DATATYPE_FP32:
            ret = ReduceDispatchDTFP<float, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_FP64:
            ret = ReduceDispatchDTFP<double, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT8:
            ret = ReduceDispatchDTFP<int8_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT8:
            ret = ReduceDispatchDTFP<uint8_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT16:
            ret = ReduceDispatchDTFP<int16_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT16:
            ret = ReduceDispatchDTFP<uint16_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT32:
            ret = ReduceDispatchDTFP<int32_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT32:
            ret = ReduceDispatchDTFP<uint32_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_INT64:
            ret = ReduceDispatchDTFP<int64_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        case HIEDNN_DATATYPE_UINT64:
            ret = ReduceDispatchDTFP<uint64_t, Impl>()(
                  dt, std::forward<Args>(args) ...);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_DATATYPE;
            break;
    }

    return ret;
}

}  // namespace reduce

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_REDUCE_REDUCE_DISPATCH_HPP_


