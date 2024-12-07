/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    packed_memory_access.hpp
 */

#ifndef DNN_INCLUDE_PACKED_MEMORY_ACCESS_HPP_
#define DNN_INCLUDE_PACKED_MEMORY_ACCESS_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace hiednn {

/*
 * packed memory access to maximum LSU utilization
 * and memory bandwidth.
 */

namespace PackConfig {
    // max size in byte of the vector (packed elements)
    constexpr size_t MAX_PACKED_BYTE = 16;
    // max number of packed elements
    constexpr size_t MAX_PACKED_SIZE = 8;
}

/*
 * return a dummy pack size for oversized packed type.
 *
 * for example:
 *
 * template <..., int PACK_SIZE, typename T> void kernel(...) {}
 *
 * size_t pack_size = GetPackSize(...);
 * switch (pack_size) {
 *     case 8:
 *         kernel<..., ValidPack<T, 8>(), T>(...);
 *         break;
 *     case 4:
 *         kernel<..., ValidPack<T, 4>(), T>(...);
 *         break;
 *     ...
 * }
 * 
 * for T = float, 8-float pack is oversized, ValidPack<...>()
 * will redirect kernel<..., 8, float> to kernel<..., 1, float>
 * and avoid generating kernel<..., 8, float>
 */
template <typename T1, int N>
constexpr size_t ValidPack() {
    return N <= PackConfig::MAX_PACKED_SIZE &&
           sizeof(T1) * N <= PackConfig::MAX_PACKED_BYTE ?
           N : 1;
}

template <typename T1, typename T2, int N>
constexpr size_t ValidPack() {
    return N <= PackConfig::MAX_PACKED_SIZE &&
           sizeof(T1) * N <= PackConfig::MAX_PACKED_BYTE &&
           sizeof(T2) * N <= PackConfig::MAX_PACKED_BYTE ?
           N : 1;
}

// Get pack size based on alignment
template <typename T>
inline size_t GetPackSize(const T *ptr) {
    if (sizeof(T) > PackConfig::MAX_PACKED_BYTE) {
        return 1;
    }

    size_t packSize = PackConfig::MAX_PACKED_BYTE / sizeof(T) <
                      PackConfig::MAX_PACKED_SIZE ?
                      PackConfig::MAX_PACKED_BYTE / sizeof(T) :
                      PackConfig::MAX_PACKED_SIZE;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    while (addr % (packSize * sizeof(T)) != 0) {
        packSize /= 2;
    }

    return packSize;
}

// helper datatype for packed memory access
template <typename T, int N>
struct alignas(N * sizeof(T)) VT {
    T data[N];
};

}  // namespace hiednn

#endif  // DNN_INCLUDE_PACKED_MEMORY_ACCESS_HPP_


