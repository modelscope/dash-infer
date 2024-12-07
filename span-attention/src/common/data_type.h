/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    data_type.h
 */

#pragma once

#include <span_attn.h>

#include <string>

#ifdef ENABLE_FP16
#include <cutlass/half.h>
#endif

#ifdef ENABLE_BF16
#include <cutlass/bfloat16.h>
#endif

namespace span {

#ifdef ENABLE_FP16
typedef cutlass::half_t half_t;
#endif

#ifdef ENABLE_BF16
typedef cutlass::bfloat16_t bfloat16_t;
#endif

template <DataType T>
struct TypeAdapter {
  using Type = float;
};

template <>
struct TypeAdapter<DataType::FP32> {
  using Type = float;
};

#ifdef ENABLE_FP16
template <>
struct TypeAdapter<DataType::FP16> {
  using Type = half_t;
};
#endif

#ifdef ENABLE_BF16
template <>
struct TypeAdapter<DataType::BF16> {
  using Type = bfloat16_t;
};
#endif

size_t sizeof_type(DataType type);

}  // namespace span
