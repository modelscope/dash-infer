/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    block.h
 */
#pragma once

#include <common.h>

#include <cstring>
#include <memory>
namespace allspark {

class AsTensor;

class Block {
 public:
  using Ptr = std::shared_ptr<Block>;

  Block() {}

  virtual DeviceType GetDeviceType() const = 0;
  virtual int DeviceId() const = 0;
  virtual int64_t Size() const = 0;
  virtual int64_t Resize(int64_t new_size) = 0;
  virtual void Free() = 0;
  virtual void* RawData() = 0;
  virtual void BindTensor(AsTensor* tensor) = 0;
  virtual void UnBindTensor(AsTensor* tensor) = 0;
  // virtual bool Freeze() = 0;
  // virtual bool UnFreeze() = 0;

  virtual void Destory() = 0;
  virtual void Restore() = 0;

  // void SetFreezer(const IHIEModelFreezer::Ptr& freezer) { freezer_ = freezer;
  // }

 protected:
  virtual ~Block() {}
  // IHIEModelFreezer::Ptr freezer_;
};

}  // namespace allspark
