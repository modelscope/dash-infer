/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cutlass_type.h
 */

#pragma once

#ifdef ENABLE_FP16
#include <cutlass/half.h>
#endif

#ifdef ENABLE_BF16
#include <cutlass/bfloat16.h>
#endif

#include "common/data_type.h"

namespace span {

template <typename T>
struct CutlassType {
  using Type = T;
};

template <>
struct CutlassType<float> {
  using Type = float;
};

#ifdef ENABLE_FP16
template <>
struct CutlassType<half_t> {
  using Type = cutlass::half_t;
};
#endif

#ifdef ENABLE_BF16
template <>
struct CutlassType<bfloat16_t> {
  using Type = cutlass::bfloat16_t;
};
#endif

}  // namespace span
