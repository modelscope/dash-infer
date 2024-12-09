/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    memory_func.cpp
 */

#include "memory_func.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace allspark {

void MemsetZero(void* dst_data, DeviceType dst_device, int64_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  if (dst_device == DeviceType::CPU) {
    memset(dst_data, 0, nbytes);
#ifdef ENABLE_CUDA
  } else if (dst_device == DeviceType::CUDA) {
    cudaMemset(dst_data, 0, nbytes);
#endif
  } else {
    LOG(ERROR) << "Not supported device " << DeviceType_Name(dst_device)
               << std::endl;
  }
}

}  // namespace allspark
