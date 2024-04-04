/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    datatype_dispatcher.h
 */
#pragma once
#include <common/common.h>
#ifdef ENABLE_FP16
#include <common/float16.h>
#endif
#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif
namespace allspark {
template <typename Functor, typename... Args>
void DispatchCPU(DataType dtype, Functor&& F, Args&&... args) {
  switch (dtype) {
    case DataType::FLOAT32: {
      std::forward<Functor>(F).template operator()<float>(
          std::forward<Args>(args)...);
      break;
    }
    default: {
      LOG(ERROR) << "unsupported datatype " << DataType_Name(dtype)
                 << " for CPU dispatch";
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }
}
template <typename Functor, typename... Args>
void DispatchCast(DataType dtype0, DataType dtype1, Functor&& F,
                  Args&&... args) {
  switch (dtype0) {
    case DataType::INT64: {
      switch (dtype1) {
        case DataType::FLOAT32: {
          std::forward<Functor>(F).template operator()<int64_t, float>(
              std::forward<Args>(args)...);
          break;
        }
        default: {
          LOG(ERROR) << "unsupported datatype " << DataType_Name(dtype1)
                     << " for Cast dispatch";
          throw AsException("ALLSPARK_RUNTIME_ERROR");
        }
      }
      break;
    }
    default: {
      LOG(ERROR) << "unsupported datatype " << DataType_Name(dtype0)
                 << " for Cast dispatch";
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }
}

/**
 * @brief DataType Trait
 *
 */

template <typename T>
struct DataTypeTrait;

#ifdef ENABLE_FP16
template <>
struct DataTypeTrait<half> {
  static constexpr DataType data_type = DataType::FLOAT16;
};
#endif
#ifdef ENABLE_BF16
template <>
struct DataTypeTrait<hie::bfloat16> {
  static constexpr DataType data_type = DataType::BFLOAT16;
};
#endif
template <>
struct DataTypeTrait<float> {
  static constexpr DataType data_type = DataType::FLOAT32;
};
template <>
struct DataTypeTrait<uint8_t> {
  static constexpr DataType data_type = DataType::UINT8;
};
template <>
struct DataTypeTrait<int8_t> {
  static constexpr DataType data_type = DataType::INT8;
};
template <>
struct DataTypeTrait<int16_t> {
  static constexpr DataType data_type = DataType::INT16;
};
template <>
struct DataTypeTrait<int32_t> {
  static constexpr DataType data_type = DataType::INT32;
};
template <>
struct DataTypeTrait<int64_t> {
  static constexpr DataType data_type = DataType::INT64;
};
template <>
struct DataTypeTrait<bool> {
  static constexpr DataType data_type = DataType::BOOL;
};

}  // namespace allspark
