/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mem_registry.h
 */

#pragma once

#include <common/common.h>

#include <shared_mutex>
#include <string>

#include "core/tensor/tensor.h"

namespace allspark {
namespace util {

void RegisterMem(uint64_t addr, const std::string& name, int64_t size,
                 DeviceType device_type);
void UnRegisterMem(uint64_t addr);
void RegStatMem();
void RegFreeMem();
void SetMemPersistent(uint64_t addr, bool is_persistent);
void SetSwapThreshold(int64_t threshold_bytes);
int64_t GetSwapThreshold();
bool SyncWeightsBuffer(TensorMap& weights_buffer,
                       const std::string& weight_name,
                       std::shared_ptr<AsTensor> src_cpu_tensor_ptr);

class MemRegistry {
 public:
  friend void RegisterMem(uint64_t addr, const std::string& name, int64_t size,
                          DeviceType device_type);
  friend void UnRegisterMem(uint64_t addr);
  friend void RegStatMem();
  friend void RegFreeMem();
  friend void SetMemPersistent(uint64_t addr, bool is_persistent);
  friend void SetSwapThreshold(int64_t threshold_bytes);
  friend int64_t GetSwapThreshold();
  friend bool SyncWeightsBuffer(TensorMap& weights_buffer,
                                const std::string& weight_name,
                                std::shared_ptr<AsTensor> src_cpu_tensor_ptr);

 private:
  struct RegEntry {
    DeviceType device_type = DeviceType::CPU;
    int64_t size = 0;
    std::string name = "NONAME";
    bool persistent = false;
    uint64_t addr = 0;  // for debug only
  };
  int64_t peak_total_bytes_ = 0;
  int64_t realtime_total_bytes_ = 0;
  int64_t swap_threshold_ = -1;
  std::unordered_map<uint64_t, RegEntry> registry_;
  std::shared_timed_mutex mutex_;
};

}  // namespace util
}  // namespace allspark
