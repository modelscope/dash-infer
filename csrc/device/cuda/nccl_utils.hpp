/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    nccl_utils.hpp
 */

#include <cuda.h>
#include <nccl.h>
namespace allspark {
static ncclDataType_t GetNcclType(DataType dtype) {
  if (dtype == DataType::FLOAT32) {
    return ncclFloat32;
  } else if (dtype == DataType::INT64) {
    return ncclInt64;
  } else if (dtype == DataType::INT32) {
    return ncclInt32;
  } else if (dtype == DataType::FLOAT16) {
    return ncclFloat16;
#if CUDA_VERSION >= 11000 && NCCL_VERSION_CODE >= 21104
  } else if (dtype == DataType::BFLOAT16) {
    return ncclBfloat16;
#endif
  } else {
    LOG(ERROR) << " not supported in DataType:" << DataType_Name(dtype)
               << std::endl;
    throw AsException("NCCL_NOT_SUPPORT_TYPE_ERROR");
  }
}
}  // namespace allspark