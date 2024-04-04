/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_allocator.h
 */

#pragma once
#include <common/allocator.h>

namespace allspark {
// to support neon to load consecutive 8xfp32 values into a 256-bit NEON
// register
#define TENSOR_ALIGN_IN_BYTES (256)
class CPUAllocator : public Allocator {
 public:
  AsStatus Alloc(void** ptr, int64_t nbytes, const std::string& name) {
    if (nbytes == 0) {
      *ptr = nullptr;
      return AsStatus::ALLSPARK_SUCCESS;
    }
    int ret = posix_memalign(ptr, TENSOR_ALIGN_IN_BYTES, nbytes);
    if (ret != 0) {
      LOG(ERROR) << "Alloc cpu memory failed, size : " << nbytes << std::endl;
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    // memset(*ptr, 0, nbytes);
    return AsStatus::ALLSPARK_SUCCESS;
  }

  AsStatus Free(void* ptr) {
    free(ptr);
    ptr = nullptr;
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
