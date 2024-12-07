/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scan_utils.cuh
 */

#ifndef DNN_CUDA_PREFIX_SCAN_SCAN_UTILS_CUH_
#define DNN_CUDA_PREFIX_SCAN_SCAN_UTILS_CUH_

#include <cstdint>

namespace hiednn {

namespace cuda {

const int MEM_ALIGN_BYTE = 128;
const int SMEM_BANKS_ROW = 128;  // 4Byte * 32 Bank

// ------------------------------------------------------------
// prefix sum compute type configuration
//
//  int8_t, int16_t: int32_t
//  uint8_t, uint16_t: uint32_t
//  half, bfloat16: float
//  others: same with input data type
// ------------------------------------------------------------
template <typename T>
struct PrefixComputeT {
    using type = T;
};

template <>
struct PrefixComputeT<int8_t> {
    using type = int32_t;
};

template <>
struct PrefixComputeT<uint8_t> {
    using type = uint32_t;
};

template <>
struct PrefixComputeT<int16_t> {
    using type = int32_t;
};

template <>
struct PrefixComputeT<uint16_t> {
    using type = uint32_t;
};

#ifdef HIEDNN_USE_FP16
template <>
struct PrefixComputeT<half> {
    using type = float;
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct PrefixComputeT<bfloat16> {
    using type = float;
};
#endif

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_PREFIX_SCAN_SCAN_UTILS_CUH_


