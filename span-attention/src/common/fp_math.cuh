/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp_math.cuh
 */

#pragma once

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include "common/data_type.h"
#include "common/func_modifier.h"

namespace span {

#ifdef ENABLE_FP16
DEVICE_FUNC half_t max(const half_t a, const half_t b) {
  auto ret = __hmax(reinterpret_cast<const __half&>(a),
                    reinterpret_cast<const __half&>(b));
  return reinterpret_cast<half_t&>(ret);
}

DEVICE_FUNC half_t min(const half_t a, const half_t b) {
  auto ret = __hmin(reinterpret_cast<const __half&>(a),
                    reinterpret_cast<const __half&>(b));
  return reinterpret_cast<half_t&>(ret);
}
#endif

#ifdef ENABLE_BF16
DEVICE_FUNC bfloat16_t max(const bfloat16_t a, const bfloat16_t b) {
#if __CUDA_ARCH__ >= 800
  auto ret = __hmax(reinterpret_cast<const __nv_bfloat16&>(a),
                    reinterpret_cast<const __nv_bfloat16&>(b));
  return reinterpret_cast<bfloat16_t&>(ret);
#else
  return a > b ? a : b;
#endif  // __CUDA_ARCH__
}

DEVICE_FUNC bfloat16_t min(const bfloat16_t a, const bfloat16_t b) {
#if __CUDA_ARCH__ >= 800
  auto ret = __hmin(reinterpret_cast<const __nv_bfloat16&>(a),
                    reinterpret_cast<const __nv_bfloat16&>(b));
  return reinterpret_cast<bfloat16_t&>(ret);
#else
  return a < b ? a : b;
#endif  // __CUDA_ARCH__
}
#endif

}  // namespace span
