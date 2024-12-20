/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_context.h
 */

#pragma once
#include <common/block_allocator.h>
#include <common/block_impl.h>
#include <common/device_context.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef ENABLE_CUSPARSELT
#include <cusparseLt.h>
#endif
#include <hiednn_cuda.h>
#include <nccl.h>
#include <nvml.h>

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

namespace allspark {

enum CUDASMDef : int {
  SM_Undefine = 0,
  SM_Volta = 0x0700,
  SM_70 = 0x0700,
  SM_Ampere = 0x0800,
  SM_80 = 0x0800,
  SM_86 = 0x0860,
  SM_Hopper = 0x0900,
  SM_90 = 0x0900,
  SM_MAX = -1,
};

class CUDAContext : public DeviceContext {
 public:
  CUDAContext()
      : stream_(0),
        cublas_handle_(0),
        cublaslt_handle_(0),
        hiednn_handle_(0),
        device_id_(0) {
    comm_ = 0;
    nranks_ = 1;
    rank_ = 0;
  }

  virtual ~CUDAContext() override;

  virtual void Init() override{};

  DeviceType GetDeviceType() const override { return DeviceType::CUDA; }

  void SetNumThreads(int num_threads) override {
    LOG(WARNING) << "CUDAContext::SetNumThreads() is not implemented."
                 << std::endl;
  }

  void SetDeviceId(int device_id) override;

  int GetDeviceId() override { return device_id_; }

  void FreeBlock(const Block::Ptr& block) override { allocator_.Free(block); }

  void ResetBlockPools() override { allocator_.ResetPools(); }

  Block::Ptr AllocBlock(int64_t nbytes) override {
    int gpu_id = rank_;
    return allocator_.Alloc(nbytes, gpu_id);
  }

  int GetRank() const override;

  int GetNranks() const override;

  void SetSparsityMatmulMode(bool enable_sparsity_matmul) override;
  bool GetSparsityMatmulMode() const override {
    return enable_sparsity_matmul_;
  };

  virtual void SetDtype(DataType new_dtype);
  // ---------------------------------------------------------------------------

  static int GetStreamProcessorVersion(int device_id);

  int GetStreamProcessorVersion();

  void InitNCCL(int rank, const ncclUniqueId& id, int nRanks);

  ncclComm_t GetNCCLComm() const;

  int GetDeviceId() const { return device_id_; }

  cudaStream_t GetStream() const { return stream_; }

  void Synchronize() const;

  cublasHandle_t GetCublasHandle() const { return cublas_handle_; }

  cublasLtHandle_t GetCublasLtHandle() const {
    return reinterpret_cast<cublasLtHandle_t>(cublas_handle_);
  }

  hiednnCudaHandle_t GetHiednnHandle() const { return hiednn_handle_; }

#ifdef ENABLE_CUSPARSELT
  cusparseLtHandle_t GetCuSparseHandle() const { return cslt_handle_; }
#endif

  void NsysProfilerStart() const;
  void NsysProfilerStop() const;

 private:
  int device_id_ = 0;
  thread_local static int last_device_id_of_this_thread_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
  hiednnCudaHandle_t hiednn_handle_;
  ncclComm_t comm_;
  int nranks_;
  int rank_;
  using GPUBlock = BlockImpl<DeviceType::CUDA, 0>;
  BlockAllocator<GPUBlock> allocator_;

#ifdef ENABLE_CUSPARSELT
  cusparseLtHandle_t cslt_handle_;
#endif
  bool cslt_handle_initialized_ = false;
  bool enable_sparsity_matmul_ = false;
};

class CUDAHostMemoryGuard {
  void* _ptr;

 public:
  CUDAHostMemoryGuard(void* ptr) { _ptr = ptr; }
  ~CUDAHostMemoryGuard() {
    if (_ptr) cudaFreeHost(_ptr);
  }
};

class NVMLManager {
 public:
  static bool CheckConsistentZeroUtilization() {
    return NVMLManager::getInstance().NvmlCheckConsistentZeroUtilization();
  }

  NVMLManager(const NVMLManager&) = delete;
  NVMLManager& operator=(const NVMLManager&) = delete;

 private:
  static NVMLManager& getInstance() {
    static NVMLManager instance;
    return instance;
  }

  bool NvmlCheckConsistentZeroUtilization() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto wait_duration =
        std::chrono::high_resolution_clock::now() - last_time_point_;
    if (wait_duration < std::chrono::milliseconds(time_interval_ms_))
      return false;
    last_time_point_ = std::chrono::high_resolution_clock::now();
    auto utilization_rates = NvmlGetAllDeviceUtilizationRates();
    last_results_.push_back(utilization_rates);
    if (last_results_.size() > total_results_) {
      last_results_.pop_front();
    }
    if (last_results_.size() == total_results_) {
      bool all_same = true;
      // 检查所有记录是否都相同
      for (const auto& result : last_results_) {
        if (result != last_results_.front()) {
          all_same = false;
          break;
        }
      }
      // 检查第一个记录中是否包含0
      if (all_same) {
        for (int value : last_results_.front()) {
          if (value == 0) {
            return true;  // 所有10次记录都相同且包含0
          }
        }
      }
    }
    return false;
  }
  std::vector<int> NvmlGetAllDeviceUtilizationRates() {
    std::vector<int> utilization_rates;
    nvmlReturn_t result;
    nvmlDevice_t device;
    for (int i = 0; i < device_count_; i++) {
      nvmlUtilization_t utilization;
      result = nvmlDeviceGetHandleByIndex(i, &device);
      if (NVML_SUCCESS != result) {
        continue;
      }
      nvmlDeviceGetUtilizationRates(device, &utilization);
      utilization_rates.push_back(utilization.gpu);
    }
    return utilization_rates;
  }
  NVMLManager() {
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
      LOG(ERROR) << "Failed to initialize NVML: " << nvmlErrorString(result)
                 << std::endl;
    } else {
      nvmlDeviceGetCount(&device_count_);
      initialized_ = true;
      LOG(INFO) << "NVML initialized successfully. device_count_: "
                << device_count_ << std::endl;
    }
  }

  ~NVMLManager() {
    if (initialized_) {
      nvmlReturn_t result = nvmlShutdown();
      if (result != NVML_SUCCESS) {
        LOG(ERROR) << "Failed to shutdown NVML: " << nvmlErrorString(result)
                   << std::endl;
      } else {
        LOG(INFO) << "NVML shutdown successfully." << std::endl;
      }
    }
  }
  unsigned int device_count_ = 0;
  bool initialized_ = false;
  int time_interval_ms_ = 10 * 1000;
  int total_results_ = 10;
  std::chrono::high_resolution_clock::time_point last_time_point_ =
      std::chrono::high_resolution_clock::now();
  std::mutex mtx_;
  std::deque<std::vector<int>> last_results_;
};

}  // namespace allspark
