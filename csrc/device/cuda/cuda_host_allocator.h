/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_host_allocator.h
 */

#pragma once
#include <common/allocator.h>
#include <cuda_runtime.h>
#include <utility/mem_registry.h>

namespace allspark {

/**
 * cuda host memory allocator, which will allocat pinned memory for faster host
 * and device transfer.
 */

class CUDAHostAllocator : public Allocator {
 public:
  AsStatus Alloc(void** ptr, int64_t nbytes, const std::string& name) {
    if (nbytes == 0) {
      *ptr = nullptr;
      return AsStatus::ALLSPARK_SUCCESS;
    }
#pragma openmp critical
    cudaError_t ret = cudaHostAlloc(ptr, nbytes, cudaHostAllocDefault);
    if (cudaSuccess != ret) {
      LOG(ERROR) << "Alloc CUDA Pinned memory failed, size : " << nbytes << ","
                 << cudaGetErrorString(ret) << std::endl;
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    // FIXME: there is no more clear way to mark this memory was on cpu, but
    // pinned memory.
    util::RegisterMem(uint64_t(*ptr), name, nbytes, DeviceType::CPU);
#ifdef CONFIG_MEM_DEBUG
    DLOG(INFO) << "CUDA Host alloc, size:" << nbytes << " *ptr= " << std::hex
               << *ptr << std::endl;
#endif
    return AsStatus::ALLSPARK_SUCCESS;
  }

  AsStatus Free(void* ptr) {
    // DLOG(INFO) << "GPU free" << std::endl;
#pragma openmp critical
    util::UnRegisterMem(uint64_t(ptr));
    cudaError_t ret = cudaFreeHost(ptr);
    if (cudaSuccess != ret) {
      LOG(ERROR) << "free CUDA Pinned memory failed, "
                 << cudaGetErrorString(ret) << std::endl;
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    ptr = nullptr;
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
