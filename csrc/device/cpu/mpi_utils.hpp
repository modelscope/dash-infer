/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mpi_utils.hpp
 */

#ifdef ENABLE_MULTINUMA
#include <mpi.h>

#include "cpu/cpu_common.h"

namespace allspark {
static MPI_Datatype GetMpiType(DataType dtype) {
  if (dtype == DataType::FLOAT32) {
    return MPI_FLOAT;
  } else if (dtype == DataType::INT64) {
    return MPI_INT64_T;
  } else if (dtype == DataType::INT32) {
    return MPI_INT32_T;
  } else if (dtype == DataType::INT16) {
    return MPI_INT16_T;
  } else if (dtype == DataType::INT8) {
    return MPI_INT8_T;
  } else if (dtype == DataType::UINT8) {
    return MPI_UINT8_T;
  } else {
    LOG(ERROR) << " not supported in DataType:" << DataType_Name(dtype)
               << std::endl;
    throw AsException("MPI_NOT_SUPPORT_TYPE_ERROR");
  }
}
#endif
}  // namespace allspark