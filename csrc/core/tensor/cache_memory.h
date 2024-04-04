/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cache_memory.h
 */
#pragma once
#include "common/common.h"
#include "common/device_context.h"
#include "data.h"  // NOLINT
namespace allspark {
class CacheMemory {
 public:
  explicit CacheMemory(DeviceType device_type, int64_t per_size)
      : device_type_(device_type), per_size_(per_size) {
    data_ = std::make_shared<DenseData>("cache", per_size, device_type);
  }
  explicit CacheMemory(DeviceType device_type, int64_t per_size,
                       int64_t per_size_param)
      : device_type_(device_type), per_size_(per_size) {
    data_ = std::make_shared<DenseData>("cache", per_size, device_type);
    zero_ = std::make_shared<DenseData>("zero", per_size_param, device_type);
    scale_ = std::make_shared<DenseData>("scale", per_size_param, device_type);
  }
  void Alloc(int64_t real_size) {
    if (data_ == nullptr) {
      LOG(ERROR) << "KVcache data_ == nullptr";
      throw AsException(("ALLSPARK_MEMORY_ERROR"));
    }
    if (real_size <= data_->GetSize()) {
      return;
    }
    // 需要重新分配,默认多分一份
    int64_t size = (real_size / per_size_ + 1) * per_size_;
    std::shared_ptr<DenseData> tmp_data =
        std::make_shared<DenseData>("cache", size, device_type_);
    switch (device_type_) {
      case DeviceType::CPU: {
        memset(tmp_data->GetRawData(), 0, size);
        memcpy(tmp_data->GetRawData(), data_->GetRawData(), data_->GetSize());
        break;
      }
      default:
        LOG(ERROR) << " CacheMemory->alloc does not support "
                   << DeviceType_Name(device_type_) << " device type"
                   << std::endl;
        return;
    }
    data_ = tmp_data;
  }
  void* GetData() { return data_->GetRawData(); }
  void* GetZero() { return zero_->GetRawData(); }
  void* GetScale() { return scale_->GetRawData(); }
  void Free() {
    data_.reset();
    zero_.reset();
    scale_.reset();
  }

 private:
  DeviceType device_type_;
  std::shared_ptr<DenseData> data_;
  std::shared_ptr<DenseData> zero_, scale_;
  int64_t per_size_ = 10 * 1024 * 1024;  // 默认10M
};
}  // namespace allspark
