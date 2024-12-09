/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_cache_allocator.h
 */

#pragma once

#include <nvml.h>

#include <shared_mutex>

#include "device/cache_allocator.h"
#include "device/cuda/cuda_context.h"

namespace allspark {

class CudaCacheAllocator : public CacheAllocator {
  // R/W lock
  mutable std::shared_mutex rw_mutex_;

  const CUDAContext* ctx_;
  const int device_id_;

  nvmlDevice_t nvml_handle_;

  // assert: physical memory should not change
  int64_t device_physical_memory_;
  // assume: reserved memory should not change
  int64_t device_reserved_memory_;
  // by assumption above, allocatable memory should not change
  int64_t device_allocatable_memory_;

 public:
  explicit CudaCacheAllocator(const DeviceContext* ctx);
  virtual ~CudaCacheAllocator() = default;

  virtual int64_t GetDeviceTotalMemory() const override;
  virtual int64_t GetDeviceFreeMemory() const override;
  virtual int64_t GetDeviceUsedMemory() const override;
};

}  // namespace allspark