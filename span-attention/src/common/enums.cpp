/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    enums.cpp
 */

#include "common/enums.h"

namespace span {

std::string to_string(SaStatus code) {
  switch (code) {
    case SaStatus::SUCCESS:
      return "SUCCESS";
    case SaStatus::CUDA_ERROR:
      return "CUDA_ERROR";
    case SaStatus::RUNTIME_ERROR:
      return "RUNTIME_ERROR";
    case SaStatus::PARAM_ERROR:
      return "PARAM_ERROR";
    case SaStatus::EXCEED_LIMIT_ERROR:
      return "EXCEED_LIMIT_ERROR";
    case SaStatus::INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case SaStatus::UNKNOWN_ERROR:
      /* fall through */
    default:
      return "UNKNOWN_ERROR";
  }
}

std::string to_string(DataType type) {
  switch (type) {
    case DataType::FP32:
      return "DataType::FP32";
    case DataType::FP16:
      return "DataType::FP16";
    case DataType::BF16:
      return "DataType::BF16";
    default:
      return std::string("<invalid DataType: ") +
             std::to_string(static_cast<int>(type)) + ">";
  }
}

std::string to_string(QuantMode mode) {
  switch (mode) {
    case QuantMode::NONE:
      return "QuantMode::NONE";
    case QuantMode::I8:
      return "QuantMode::I8";
    case QuantMode::U4:
      return "QuantMode::U4";
    default:
      return std::string("<invalid QuantMode: ") +
             std::to_string(static_cast<int>(mode)) + ">";
  }
}

}  // namespace span
