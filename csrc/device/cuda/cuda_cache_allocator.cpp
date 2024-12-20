/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_cache_allocator.cpp
 */

#include "cuda_cache_allocator.h"

#include <cstdlib>
#include <string>
#include <utility>

#include "device/bfc_allocator.h"
#include "utility/check.h"

// #define NVML_USE_V2_API

namespace allspark {

namespace {

std::shared_ptr<BFCAllocator> get_bfc(const CUDAContext* ctx) {
  std::shared_ptr<Allocator> allocator = GetBFCAllocator(ctx->GetDeviceType());
  return std::dynamic_pointer_cast<BFCAllocator>(allocator);
}

}  // namespace

#define CHECK_NVML_RET(expr)                                               \
  do {                                                                     \
    nvmlReturn_t __ret = (expr);                                           \
    if (__ret != NVML_SUCCESS) {                                           \
      LOG(ERROR) << "[NVML Error] " << __FILE__ << ":" << __LINE__ << ": " \
                 << nvmlErrorString(ret) << " in " << #expr;               \
      return AsStatus::ALLSPARK_RUNTIME_ERROR;                             \
    }                                                                      \
  } while (0)

#define CHECK_NVML(expr)                                           \
  do {                                                             \
    nvmlReturn_t __ret = (expr);                                   \
    if (__ret != NVML_SUCCESS) {                                   \
      LOG(ERROR) << "[NVML Error] " << __FILE__ << ":" << __LINE__ \
                 << ": error code " << int(__ret) << " "           \
                 << nvmlErrorString(__ret) << " in " << #expr;     \
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);                  \
    }                                                              \
  } while (0)

CudaCacheAllocator::CudaCacheAllocator(const DeviceContext* ctx)
    : CacheAllocator(DeviceType::CUDA),
      ctx_(dynamic_cast<const CUDAContext*>(ctx)),
      device_id_(ctx_->GetDeviceId()),
      nvml_handle_{nullptr},
      device_physical_memory_(0),
      device_reserved_memory_(0),
      device_allocatable_memory_(0) {
  // check BFC first
  auto bfc_allocator = get_bfc(ctx_);

  if (bfc_allocator != nullptr) {
    LOG(INFO) << "CudaCacheAllocator: found BFC allocator";

    //! NOTE: no BFC growth allowed!
    // default is ON
    bool allow_growth = true;
    const char* bfc_allow_growth = std::getenv("BFC_ALLOW_GROWTH");
    if (bfc_allow_growth && std::string(bfc_allow_growth) == "OFF") {
      allow_growth = false;
    }
    if (allow_growth) {
      LOG(WARNING) << "CudaCacheAllocator: do not allow BFC growth for now, "
                      "BFC_ALLOW_GROWTH should be set to OFF";
    }

    auto stats = bfc_allocator->GetStats();
    device_allocatable_memory_ = stats.bytes_limit;
  } else {
    LOG(INFO) << "CudaCacheAllocator: no BFC allocator, backed by NVML";
    // ensure NVML initialized
    CHECK_NVML(nvmlInit_v2());

    // we cannot really trust device context, so just get the handle every time
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_id_, &nvml_handle_));
    if (nvml_handle_ == nullptr) {
      LOG(ERROR) << "CudaCacheAllocator: failed to get NVML handle";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

// total = reserved + used + free
#ifdef NVML_USE_V2_API
    nvmlMemory_v2_t mem_info{NVML_STRUCT_VERSION(Memory, 2)};
    CHECK_NVML(nvmlDeviceGetMemoryInfo_v2(nvml_handle_, &mem_info));
    device_physical_memory_ = static_cast<int64_t>(mem_info.total);
    device_reserved_memory_ = static_cast<int64_t>(mem_info.reserved);
    // available == used + free
    size_t available_mem = mem_info.total - mem_info.reserved;
#else
    nvmlMemory_t mem_info;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(nvml_handle_, &mem_info));
    device_physical_memory_ = static_cast<int64_t>(mem_info.total);
    // available == used + free
    size_t available_mem = mem_info.total;
#endif

    /*
     * NOTE: the logic below mostly follows from InitBFCAllocator in
     * csrc/device/bfc_allocator.cpp
     */
    LOG(INFO) << "CudaCacheAllocator device " << device_id_
              << ", reserved memory (MB): " << (device_reserved_memory_ >> 20)
              << ", memory in use (MB): " << (mem_info.used >> 20)
              << ", free memory (MB): " << (mem_info.free >> 20)
              << ", total memory (MB): " << (mem_info.total >> 20);

    // for pytorch by default
    uint64_t leftover_bytes = 600ULL << 20;
    const char* bfc_leftover_mb = std::getenv("BFC_LEFTOVER_MB");
    if (bfc_leftover_mb) {
      try {
        leftover_bytes = std::stoull(std::string(bfc_leftover_mb)) << 20;
      } catch (const std::exception& e) {
        LOG(WARNING) << "Invalid numeric format for BFC_LEFTOVER_MB: "
                     << bfc_leftover_mb
                     << ", will use default val: " << (leftover_bytes >> 20);
      }
    }

    // if available mem is less than leftover, free mem is also less than
    // leftover, so we only check free mem
    if (mem_info.free < leftover_bytes) {
      LOG(ERROR) << "CudaCacheAllocator device " << device_id_
                 << ", not enough memory: free memory " << (mem_info.free >> 20)
                 << " MB, while " << (leftover_bytes >> 20)
                 << " MB leftover is required";
      AS_THROW(AsStatus::ALLSPARK_MEMORY_ERROR);
    }

    float ratio = 0.975;
    const char* mem_ratio = std::getenv("BFC_MEM_RATIO");
    if (mem_ratio) {
      try {
        float ratio_tmp = std::stof(std::string(mem_ratio));
        if (ratio_tmp <= 0 || ratio_tmp > 1) {
          LOG(WARNING) << "Invalid float range for env var BFC_MEM_RATIO: "
                       << mem_ratio << ", will use default val: " << ratio;
        } else {
          ratio = ratio_tmp;
        }
      } catch (std::exception& e) {
        LOG(WARNING) << "Invalid float format for env var BFC_MEM_RATIO: "
                     << mem_ratio << ", will use default val: " << ratio;
      }
    }

    // NVML-backed allocator will never grow available memory
    // assert: available_mem >= leftover_bytes, because free >= leftover
    device_allocatable_memory_ =
        static_cast<int64_t>((available_mem - leftover_bytes) * ratio);

    if (static_cast<int64_t>(mem_info.free) < device_allocatable_memory_) {
      LOG(WARNING) << "CudaCacheAllocator device " << device_id_
                   << ", at THIS MOMENT, not enough memory: free memory "
                   << (mem_info.free >> 20) << " MB, while "
                   << (device_allocatable_memory_ >> 20)
                   << " MB allocatable memory is required";
    }
  }

  LOG(INFO) << "CudaCacheAllocator device " << device_id_
            << ", total allocatable memory (MB): "
            << (device_allocatable_memory_ >> 20);
}

int64_t CudaCacheAllocator::GetDeviceTotalMemory() const {
  std::shared_lock read_lock(rw_mutex_);
  return device_allocatable_memory_;
}

int64_t CudaCacheAllocator::GetDeviceFreeMemory() const {
  std::shared_lock read_lock(rw_mutex_);

  // check BFC first
  auto bfc_allocator = get_bfc(ctx_);

  if (bfc_allocator != nullptr) {
    auto stats = bfc_allocator->GetStats();
    return stats.bytes_limit - stats.bytes_in_use;
  } else {
    // san check
    if (device_id_ != ctx_->GetDeviceId()) {
      LOG(ERROR) << "CudaCacheAllocator: device ID mismatch";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

#ifdef NVML_USE_V2_API
    nvmlMemory_v2_t mem_info{NVML_STRUCT_VERSION(Memory, 2)};
    CHECK_NVML(nvmlDeviceGetMemoryInfo_v2(nvml_handle_, &mem_info));
#else
    nvmlMemory_t mem_info;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(nvml_handle_, &mem_info));
#endif

    // TODO: after using cache mem pool
#if 0
    return std::min(static_cast<int64_t>(mem_info.free),
                    device_allocatable_memory_);
#else
    return static_cast<int64_t>(mem_info.free);
#endif
  }
}

int64_t CudaCacheAllocator::GetDeviceUsedMemory() const {
  std::shared_lock read_lock(rw_mutex_);

  // check BFC first
  auto bfc_allocator = get_bfc(ctx_);

  if (bfc_allocator != nullptr) {
    auto stats = bfc_allocator->GetStats();
    return stats.bytes_in_use;
  } else {
    // san check
    if (device_id_ != ctx_->GetDeviceId()) {
      LOG(ERROR) << "CudaCacheAllocator: device ID mismatch";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

#ifdef NVML_USE_V2_API
    nvmlMemory_v2_t mem_info{NVML_STRUCT_VERSION(Memory, 2)};
    CHECK_NVML(nvmlDeviceGetMemoryInfo_v2(nvml_handle_, &mem_info));
#else
    nvmlMemory_t mem_info;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(nvml_handle_, &mem_info));
#endif

    // TODO: after using cache mem pool
#if 0
    int64_t free_mem = std::min(static_cast<int64_t>(mem_info.free),
                                device_allocatable_memory_);
    // assert: free_mem <= device_allocatable_memory_, so ret >= 0
    return device_allocatable_memory_ - free_mem;
#else
    return static_cast<int64_t>(mem_info.used);
#endif
  }
}

#undef CHECK_NVML_RET
#undef CHECK_NVML

}  // namespace allspark