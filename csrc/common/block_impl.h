/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    block_impl.h
 */

#pragma once

#include <cpu/cpu_allocator.h>

#include <unordered_set>

#include "device_context.h"
#ifdef ENABLE_CUDA
#include <cuda/cuda_allocator.h>
#endif
#include <bfc_allocator.h>
#include <common/allocator.h>
namespace allspark {

template <DeviceType Device, size_t Alignment>
class BlockImpl : public Block {
 public:
  BlockImpl() = delete;
  BlockImpl(int64_t size, int device_id = 0, bool iskey = false)
      : device_id_(device_id) {
    switch (Device) {
      case DeviceType::CPU: {
        allocator_ = std::make_shared<CPUAllocator>();
        break;
      }
#ifdef ENABLE_CUDA
      case DeviceType::CUDA: {
        allocator_ = GetBFCAllocator(Device);
        if (allocator_ == nullptr)
          allocator_ = std::make_shared<CUDAAllocator>();
        break;
      }
#endif
      default: {
        LOG(ERROR) << "DeviceType::" << Device
                   << " is not supported. Please check build option."
                   << std::endl;
        throw AsException("ALLSPARK_PARAM_ERROR");
      }
    }
    if (!iskey) {
      Resize(size);
    } else {
      size_ = size;
    }
  }

  BlockImpl(const BlockImpl&) = delete;
  BlockImpl& operator=(const BlockImpl&) = delete;

  BlockImpl(BlockImpl&& other) {
    device_id_ = other.device_id_;
    size_ = other.size_;
    allocator_ = other.allocator_;
    ptr_ = other.ptr_;

    other.device_id_ = 0;
    other.size_ = 0;
    other.allocator_ = nullptr;
    other.ptr_ = nullptr;
  }

  BlockImpl& operator=(BlockImpl&& other) {
    device_id_ = other.device_id_;
    size_ = other.size_;
    allocator_ = other.allocator_;
    ptr_ = other.ptr_;

    other.device_id_ = 0;
    other.size_ = 0;
    other.allocator_ = nullptr;
    other.ptr_ = nullptr;
    return *this;
  }

  ~BlockImpl() { Free(); }

  DeviceType GetDeviceType() const override { return Device; }

  int DeviceId() const override { return device_id_; }

  int64_t Size() const override { return size_; }

  int64_t Resize(int64_t new_size) override {
    if (size_ < new_size) {
      Free();
      allocator_->Alloc(&ptr_, (int64_t)new_size, "BLOCK");
      size_ = new_size;
    }
    return size_;
  }

  void Free() override {
    if (ptr_ != nullptr) {
      allocator_->Free(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  void Destory() override {
    if (ptr_ != nullptr) {
      // DeviceGuard<Device> guard(device_id_);
      allocator_->Free(ptr_);
      ptr_ = nullptr;
    }
  }

  void Restore() override {
    if (ptr_ == nullptr && size_ > 0) {
      // DeviceGuard<Device> guard(device_id_);
      allocator_->Alloc((void**)ptr_, (int64_t)size_, "BLOCK");
      // HIE_ENFORCE(ptr_, enum_to_string(Device), " memory allocate
      // failed");
    }
  }

  void* RawData() override {
    // Restore();
    // UnFreeze();
    return ptr_;
  }

  bool operator<(const BlockImpl& other) const {
    if (size_ != other.size_) return size_ < other.size_;
    return this < &other;
  }

  void BindTensor(AsTensor* tensor) override { tensors_.insert(tensor); }

  void UnBindTensor(AsTensor* tensor) override { tensors_.erase(tensor); }

  // bool Freeze() override {
  //     if (!freezed_ && ptr_ && size_ > 0 && freezer_ && tensors_.size() ==
  //     1) {
  //         freezed_tensor_ = *(tensors_.begin());
  //         HIETensor hie_tensor(freezed_tensor_);
  //         bool is_freezable = freezer_->CheckFreezable(hie_tensor);
  //         if (is_freezable && freezer_->OnFreeze(hie_tensor)) {
  //             this->Destory();
  //             freezed_ = true;
  //             return true;
  //         }
  //     }
  //     return false;
  // }

  // bool UnFreeze() override {
  //     if (freezed_) {
  //         HIE_ENFORCE(freezer_, "invalid freezer");
  //         HIE_ENFORCE(freezed_tensor_ != nullptr, "invalid freezed
  //         tensor"); HIE_ENFORCE(tensors_.count(freezed_tensor_) != 0,
  //         "invalid freezed tensor"); this->Restore(); freezed_ = false;
  //         HIETensor hie_tensor(freezed_tensor_);
  //         HIE_ENFORCE(freezer_->OnUnfreeze(hie_tensor) == true, "unfreeze
  //         failed"); return true;
  //     }
  //     return false;
  // }

 private:
  int device_id_ = 0;                               // gpu
  int64_t size_ = 0;                                // block size in bytes
  std::shared_ptr<Allocator> allocator_ = nullptr;  // allocator
  void* ptr_ = nullptr;                             // memory address
  std::unordered_set<AsTensor*> tensors_;  // most recently binded tensor

  // AsTensor* freezed_tensor_ = nullptr;
  // bool freezed_ = false;
};

}  // namespace allspark
