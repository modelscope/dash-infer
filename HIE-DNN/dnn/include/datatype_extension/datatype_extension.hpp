/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    datatype_extension.hpp
 */

#ifndef DNN_INCLUDE_DATATYPE_EXTENSION_DATATYPE_EXTENSION_HPP_
#define DNN_INCLUDE_DATATYPE_EXTENSION_DATATYPE_EXTENSION_HPP_

#ifdef HIEDNN_USE_FP16
#include "half/hiednn_half.hpp"
#endif

#ifdef HIEDNN_USE_BF16
#include "bfloat16/hiednn_bfloat16.hpp"
#endif

namespace __hiednn_buildin {

#ifdef __CUDA_ARCH__
#define __EXT_TYPE_DEVICE_FUNC __device__ __forceinline__
#else
#define __EXT_TYPE_DEVICE_FUNC inline
#endif

// ---------- convert to half ----------
#if defined(HIEDNN_USE_FP16) && defined(HIEDNN_USE_BF16)
__EXT_TYPE_DEVICE_FUNC
half::half(bfloat16 v) : half(static_cast<float>(v)) {}
#endif
// -------------------------------------

// ---------- convert to bfloat16 ----------
#if defined(HIEDNN_USE_FP16) && defined(HIEDNN_USE_BF16)
__EXT_TYPE_DEVICE_FUNC
bfloat16::bfloat16(half v) : bfloat16(static_cast<float>(v)) {}
#endif
// -----------------------------------------

#undef __EXT_TYPE_DEVICE_FUNC

}  // namespace __hiednn_buildin

#endif  // DNN_INCLUDE_DATATYPE_EXTENSION_DATATYPE_EXTENSION_HPP_


