/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_allocator.h
 */

#pragma once
#include <common/allocator.h>
#include <cuda_runtime.h>
#include <utility/mem_registry.h>

// #define CONFIG_MEM_DEBUG
namespace allspark {
class CUDAAllocator : public Allocator {
 public:
  CUDAAllocator(bool do_register_mem = true)
      : do_register_mem_(do_register_mem) {}
  AsStatus Alloc(void** ptr, int64_t nbytes, const std::string& name) {
    if (nbytes == 0) {
      *ptr = nullptr;
      return AsStatus::ALLSPARK_SUCCESS;
    }
#pragma openmp critical
    cudaError_t ret = cudaMalloc(ptr, nbytes);
    if (cudaSuccess != ret) {
      LOG(ERROR) << "Alloc CUDA memory failed, size : " << nbytes << ","
                 << cudaGetErrorString(ret) << std::endl;
      print_backtrace();  // NOTE(liyifei): in overbooking mode, malloc failure
                          // does not mean
      // the failure of the whole engine, so we must reset CUDA error
      // status after catching the malloc error. (20231104)
      cudaGetLastError();
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    if (do_register_mem_)
      util::RegisterMem(uint64_t(*ptr), name, nbytes, DeviceType::CUDA);

#ifdef CONFIG_MEM_DEBUG
    DLOG(INFO) << "GPU alloc " << name << ", size:" << nbytes
               << " *ptr= " << std::hex << *ptr << std::endl;
#endif
    return AsStatus::ALLSPARK_SUCCESS;
  }

  AsStatus Free(void* ptr) {
#ifdef CONFIG_MEM_DEBUG
    DLOG(INFO) << "GPU free: " << std::hex << ptr << std::endl;
#endif
#pragma openmp critical
    if (do_register_mem_) util::UnRegisterMem(uint64_t(ptr));
    cudaError_t ret = cudaFree(ptr);
    if (cudaSuccess != ret) {
      LOG(ERROR) << "free CUDA memory failed, " << std::hex << ptr << " "
                 << cudaGetErrorString(ret) << std::endl;
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    ptr = nullptr;
    return AsStatus::ALLSPARK_SUCCESS;
  }

 private:
  bool do_register_mem_ = true;
};

}  // namespace allspark
