/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    data_type.cpp
 */

#include "common/data_type.h"

#include <stdexcept>

#include "common/enums.h"
#include "common/error.h"
#include "common/logger.h"

namespace span {

size_t sizeof_type(DataType type) {
  switch (type) {
    case DataType::FP32:
      return 4UL;
#ifdef ENABLE_FP16
    case DataType::FP16:
      return 2UL;
#endif
#ifdef ENABLE_BF16
    case DataType::BF16:
      return 2UL;
#endif
    default:
      LOG(ERROR) << "sizeof_type: unsupported data type: " << to_string(type)
                 << std::endl;
      throw SpanAttnError(SaStatus::PARAM_ERROR,
                          "unsupported data type: " + to_string(type));
  }
}

}  // namespace span
