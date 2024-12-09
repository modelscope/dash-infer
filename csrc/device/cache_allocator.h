/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cache_allocator.h
 */

#pragma once

#include <memory>

#include "common/common.h"
#include "common/device_context.h"

namespace allspark {

class CacheAllocator {
  DeviceType device_type_;

  // TODO: allocator

 public:
  using Ptr = std::shared_ptr<CacheAllocator>;

  explicit CacheAllocator(DeviceType device_type) : device_type_(device_type){};
  virtual ~CacheAllocator() = default;

  DeviceType GetDeviceType() const { return device_type_; }

  virtual int64_t GetDeviceTotalMemory() const = 0;
  virtual int64_t GetDeviceFreeMemory() const = 0;
  virtual int64_t GetDeviceUsedMemory() const = 0;

  // TODO: Alloc, Free
};

}  // namespace allspark