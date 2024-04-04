/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allocator.h
 */

#pragma once
#include <common/common.h>

#include <string>

namespace allspark {

// base allocator interface
class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual AsStatus Alloc(void** ptr, int64_t nbytes,
                         const std::string& name) = 0;
  virtual AsStatus Free(void* ptr) = 0;
};

}  // namespace allspark
