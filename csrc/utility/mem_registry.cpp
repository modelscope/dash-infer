/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mem_registry.cpp
 */

#include "mem_registry.h"

#include <stdlib.h>

#include <algorithm>

#include "check.h"
#include "check_cuda.h"
#include "common/common.h"
#include "core/model/model.h"
#include "core/tensor/data.h"
#include "core/tensor/tensor.h"
#include "device/bfc_allocator.h"
#include "mutex_wrapper.h"

namespace allspark {
namespace util {

#define DISABLE_MEM_REG 1

static MemRegistry g_mem_registry;
void SetMemPersistent(uint64_t addr, bool is_persistent) {
#if DISABLE_MEM_REG
  return;
#endif

  unique_lock_wrapper<std::shared_timed_mutex> lock(g_mem_registry.mutex_,
                                                    "SetMemPersistent");
  if (not g_mem_registry.registry_.count(addr)) {
    LOG(WARNING) << "SetMemPersistent: key not found in registry: " << std::hex
                 << addr << std::endl;
    return;
  }
  g_mem_registry.registry_[addr].persistent = is_persistent;

#ifdef CONFIG_MEM_DEBUG
  DLOG(INFO) << "SetPersistent to " << is_persistent << ", "
             << g_mem_registry.registry_[addr].name << " " << std::hex << addr
             << std::endl;
#endif
}
void SetSwapThreshold(int64_t threshold_bytes) {
#if DISABLE_MEM_REG
  return;
#endif

  unique_lock_wrapper<std::shared_timed_mutex> lock(g_mem_registry.mutex_);
  g_mem_registry.swap_threshold_ = threshold_bytes;
}
int64_t GetSwapThreshold() {
  std::shared_lock<std::shared_timed_mutex> lock(g_mem_registry.mutex_);
  return g_mem_registry.swap_threshold_;
}
void RegisterMem(uint64_t addr, const std::string& name, int64_t size,
                 DeviceType device_type) {
#if DISABLE_MEM_REG
  return;
#endif
  unique_lock_wrapper<std::shared_timed_mutex> lock(g_mem_registry.mutex_,
                                                    "RegisterMem");

#ifdef CONFIG_MEM_DEBUG
  DLOG(INFO) << "Register " << name << std::endl;
#endif
  if (g_mem_registry.registry_.count(addr)) {
    LOG(ERROR) << "key already exists in registry: " << std::hex << addr
               << std::endl;
    return;
  }
  MemRegistry::RegEntry entry{device_type, size, name, false, addr};
  g_mem_registry.registry_[addr] = entry;
  g_mem_registry.realtime_total_bytes_ += entry.size;
  g_mem_registry.peak_total_bytes_ = std::max(
      g_mem_registry.peak_total_bytes_, g_mem_registry.realtime_total_bytes_);
}
void UnRegisterMem(uint64_t addr) {
#if DISABLE_MEM_REG
  return;
#endif
  unique_lock_wrapper<std::shared_timed_mutex> lock(g_mem_registry.mutex_,
                                                    "UnRegMem");
  if (not g_mem_registry.registry_.count(addr)) {
    LOG(ERROR) << "UnRegisterMem: key not found in registry: " << std::hex
               << addr << std::endl;  // should NEVER enter here
    return;
  }

#ifdef CONFIG_MEM_DEBUG
  DLOG(INFO) << "UnRegister " << g_mem_registry.registry_[addr].name << " "
             << std::hex << addr << std::endl;
#endif
  g_mem_registry.realtime_total_bytes_ -= g_mem_registry.registry_[addr].size;
  g_mem_registry.registry_.erase(addr);
}
void RegFreeMem() {
#if DISABLE_MEM_REG
  return;
#endif
  unique_lock_wrapper<std::shared_timed_mutex> lock(g_mem_registry.mutex_,
                                                    "RegFreeMem");

  std::vector<uint64_t> addr_to_erase;
  for (auto& pair : g_mem_registry.registry_) {
    auto& addr = pair.first;
    auto& entry = pair.second;
    if (entry.device_type != DeviceType::CPU) {
      if (not entry.persistent and
          entry.size > g_mem_registry.swap_threshold_) {
        DLOG(INFO) << "RegFreeMem freeing " << entry.name << " " << std::hex
                   << addr << std::endl;
        addr_to_erase.push_back(addr);
      }
    }
  }
  for (auto addr : addr_to_erase) {
#ifdef ENABLE_CUDA
    auto allocator = GetBFCAllocator(DeviceType::CUDA);
    if (allocator == nullptr)
      cudaFree((void*)addr);
    else
      allocator->Free((void*)addr);
#endif
    g_mem_registry.registry_.erase(addr);
  }
  g_mem_registry.realtime_total_bytes_ = g_mem_registry.peak_total_bytes_ = 0;
}

void RegStatMem() {
#if DISABLE_MEM_REG
  return;
#endif
  std::shared_lock<std::shared_timed_mutex> lock(g_mem_registry.mutex_);
  LOG(INFO) << "===================== model CUDA mem stat ====================="
            << std::endl;
  LOG(INFO) << "Peak=" << g_mem_registry.peak_total_bytes_
            << ", realtime total=" << g_mem_registry.realtime_total_bytes_
            << std::endl;
  std::vector<MemRegistry::RegEntry> v;
  for (auto& pair : g_mem_registry.registry_) {
    auto& addr = pair.first;
    auto& entry = pair.second;
    v.emplace_back(entry);
  }
  std::sort(v.begin(), v.end(),
            [](const MemRegistry::RegEntry& a, const MemRegistry::RegEntry& b) {
              return a.size < b.size;
            });
  for (auto& entry : v) {
    LOG(INFO) << entry.name << "\t" << entry.size << std::hex << "\t"
              << entry.addr << std::endl;
  }
  LOG(INFO)
      << "===================== model CUDA mem stat END ====================="
      << std::endl;
}

bool SyncWeightsBuffer(TensorMap& weights_buffer,
                       const std::string& weight_name,
                       std::shared_ptr<AsTensor> src_cpu_tensor_ptr) {
#if DISABLE_MEM_REG
  return true;
#endif
  std::unique_lock<std::shared_timed_mutex> lock(g_mem_registry.mutex_);
  if (weights_buffer.size() ==
      0)  // disable cuda-mem swap, set by user when building model
    return true;
  if (weights_buffer.count(weight_name) == 0) {
    throw AsException(
        "SyncWeightsBuffer(): Inconsistent state: weights_buffer not empty, "
        "but key " +
        weight_name + " not found!");
  }
  weights_buffer[weight_name] = src_cpu_tensor_ptr;
  return true;
}

}  // namespace util
}  // namespace allspark
