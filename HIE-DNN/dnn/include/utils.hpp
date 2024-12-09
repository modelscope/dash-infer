/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utils.hpp
 */

#ifndef DNN_INCLUDE_UTILS_HPP_
#define DNN_INCLUDE_UTILS_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <tensor_desc.hpp>
#include <device_function_modifier.hpp>
#include <datatype_extension/datatype_extension.hpp>

#define CHECK_HIEDNN_RETURN(EXPR) { \
    auto err = (EXPR); \
    if (err != HIEDNN_STATUS_SUCCESS) { \
        return err; \
    } \
}

namespace hiednn {

// ----------------------------------------------------
// check pointers
// return false if any of the pointers is nullptr
// ----------------------------------------------------
inline bool CheckNullptr(const void *ptr) {
    return ptr != nullptr;
}

template <class... Args>
inline bool CheckNullptr(const void *ptr, Args&&... args) {
    return (ptr != nullptr) && CheckNullptr(std::forward<Args>(args) ...);
}

// ----------------------------------------------------
// check tensor pointers
// return false if any of pointers is nullptr but the tensor size is not zero
// ----------------------------------------------------
inline bool CheckTensorPtr(const HiednnTensorDesc &desc, const void *ptr) {
    return (desc.size == 0 || ptr != nullptr);
}

template <class... Args>
inline bool CheckTensorPtr(const HiednnTensorDesc &desc,
                           const void *ptr,
                           Args&&... args) {
    return (desc.size == 0 || ptr != nullptr) &&
           CheckTensorPtr(std::forward<Args>(args) ...);
}

// ----------------------------------------------------
// check NORMAL format
// return false if tensorFormat of any of the descriptor is not NORMAL
// ----------------------------------------------------
inline bool CheckNormalFormat(const HiednnTensorDesc &desc) {
    return desc.tensorFormat == HIEDNN_TENSORFORMAT_NORMAL;
}

template <class... Args>
inline bool CheckNormalFormat(const HiednnTensorDesc &desc, Args&&... args) {
    return (desc.tensorFormat == HIEDNN_TENSORFORMAT_NORMAL) &&
           CheckNormalFormat(std::forward<Args>(args) ...);
}

// ----------------------------------------------------
// integer round up division
// ----------------------------------------------------
template <typename T>
inline T SIntDivRU(T x, T y) {
    // round away from 0
    T ret;

    if (x == 0) {
        ret = 0;
    } else if (((x ^ y) & (T(1) << (8 * sizeof(T) - 1))) != 0) {
        ret = (x - y - (x > 0 ? 1 : -1)) / y;
    } else {
        ret = (x + y - (x > 0 ? 1 : -1)) / y;
    }

    return ret;
}

template <typename T>
HOST_DEVICE_FUNCTION
T UIntDivRU(T x, T y) {
    // only work for non-negative integer
    return (x + y - 1) / y;
}

// ----------------------------------------------------
// integer assertion
// ----------------------------------------------------
template <typename T>
HOST_DEVICE_FUNCTION
constexpr bool IsPowOf2(const T &x) {
    return (x & (x - 1)) == 0;
}

// ----------------------------------------------------
// cross-device array
// ----------------------------------------------------
template <typename T, int N>
struct Array {
    T data[N];

    Array() {}

    // constructors with arguments only work for host
    template <typename S>
    Array(const S *src, int n) {
        for (int i = 0; i < n; ++i) {
            data[i] = static_cast<T>(src[i]);
        }
    }

    template <typename S>
    Array(const S *src, int n, const S &fill) {
        for (int i = 0; i < n; ++i) {
            data[i] = static_cast<T>(src[i]);
        }

        for (int i = n; i < N; ++i) {
            data[i] = static_cast<T>(fill);
        }
    }

    HOST_DEVICE_FUNCTION
    T& operator[](unsigned i) { return data[i]; }

    HOST_DEVICE_FUNCTION
    const T& operator[](unsigned i) const { return data[i]; }

    HOST_DEVICE_FUNCTION
    constexpr int size() const { return N; }
};

// ----------------------------------------------------
// return item size in byte of @datatype
// ----------------------------------------------------
inline size_t ItemSizeInByte(hiednnDataType_t datatype) {
    size_t size = 0;
    switch (datatype) {
        case HIEDNN_DATATYPE_FP64:
        case HIEDNN_DATATYPE_INT64:
        case HIEDNN_DATATYPE_UINT64:
            size = 8;
            break;
        case HIEDNN_DATATYPE_FP32:
        case HIEDNN_DATATYPE_INT32:
        case HIEDNN_DATATYPE_UINT32:
            size = 4;
            break;
        case HIEDNN_DATATYPE_FP16:
        case HIEDNN_DATATYPE_BF16:
        case HIEDNN_DATATYPE_INT16:
        case HIEDNN_DATATYPE_UINT16:
            size = 2;
            break;
        case HIEDNN_DATATYPE_INT8:
        case HIEDNN_DATATYPE_UINT8:
        case HIEDNN_DATATYPE_BOOL:
            size = 1;
            break;
        default:
            break;
    }

    return size;
}

// ----------------------------------------------------
// map build-in datatype to hiednn datatype
// ----------------------------------------------------
template <typename T>
struct HiednnDataType;

template <>
struct HiednnDataType<int8_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_INT8;
};

template <>
struct HiednnDataType<uint8_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_UINT8;
};

template <>
struct HiednnDataType<int16_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_INT16;
};

template <>
struct HiednnDataType<uint16_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_UINT16;
};

template <>
struct HiednnDataType<int32_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_INT32;
};

template <>
struct HiednnDataType<uint32_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_UINT32;
};

template <>
struct HiednnDataType<int64_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_INT64;
};

template <>
struct HiednnDataType<uint64_t> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_UINT64;
};

template <>
struct HiednnDataType<float> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_FP32;
};

template <>
struct HiednnDataType<double> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_FP64;
};

#ifdef HIEDNN_USE_FP16
template <>
struct HiednnDataType<half> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_FP16;
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct HiednnDataType<bfloat16> {
    static const hiednnDataType_t type = HIEDNN_DATATYPE_BF16;
};
#endif

// ----------------------------------------------------
// ConstExpr (Compile Time Constant) Helper
// ----------------------------------------------------
struct ConstExpr {
    // constexpr Pow<X, Y> = X^Y
    template <int X, int Y>
    struct Pow {
        static constexpr int N = X * Pow<X, Y - 1>::N;
    };

    template <int X>
    struct Pow<X, 0> {
        static constexpr int N = 1;
    };

    // constexpr DivRU<X, Y> = (X + Y - 1) / Y
    template <int X, int Y>
    struct DivRU {
        static constexpr int N = (X + Y - 1) / Y;
    };

    template <int X>
    struct DivRU<X, 0> {
        static constexpr int N = 0;
    };
};

// ----------------------------------------------------
// Same type judgement
// ----------------------------------------------------
template <typename A, typename B>
struct SameType {
    static constexpr bool Same = false;
};

template <typename A>
struct SameType<A, A> {
    static constexpr bool Same = true;
};

}  // namespace hiednn

#endif  // DNN_INCLUDE_UTILS_HPP_


