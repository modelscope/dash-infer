/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    enum.h
 */

#pragma once

#include <cstdint>

namespace hie {

// keep consistent with onnx datatype
enum class DataType : int32_t {
  DATATYPE_UNDEFINED = 0,
  // Basic types.
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16 = 10,

  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,   // complex with float32 real and imaginary components
  COMPLEX128 = 15,  // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16 = 16,
  INT4 = 17,

  // 8-bit floating-point number to be accurate at a smaller dynamic range than
  // half precision. The E4 and M3 represent a 4-bit exponent and a 3-bit
  // mantissa respectively.
  FLOAT8E4M3 = 18,
  // 8-bit floating-point number to be accurate at a similar dynamic range than
  // half precision. The E5 and M2 represent a 5-bit exponent and a 2-bit
  // mantissa respectively.
  FLOAT8E5M2 = 19,
  // Future extensions go here.
  // ...
  POINTER = 20,
};

enum class Packing : int32_t {
  PACKING_UNDEFINED = 0,
  NCHW = 1,
  NHWC = 2,
  NC4HW = 3,
  NC4HW4 = 4,
  NC16HW = 5,
  NC8HW = 6,
  NC16HW16 = 7,
  NC32HW32 = 8,
  NC32HW = 9,
  CHWN = 10
};

enum DeviceType : int32_t {
  DEVICETYPE_UNDEFINED = 0,
  CPU = 1,
  GPU = 2,
  // ...
  MAX_DEVICE_TYPES,
  CPU_PINNED = MAX_DEVICE_TYPES + 1,
};

enum class ModelType : int32_t { FRAMEWORK_ONNX = 0, FRAMEWORK_HIE };

}  // namespace hie
