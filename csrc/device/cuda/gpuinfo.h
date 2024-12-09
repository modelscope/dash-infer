/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gpuinfo.h
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

namespace hie {

class GPUInfo {
 public:
  GPUInfo() {
    // TODO
    // map device ID for multiple GPUs
    int gpu_id = 0;
    // HIE_ENFORCE_EQ(cudaGetDevice(&gpu_id), 0);
    // HIE_ENFORCE_EQ(cudaGetDeviceProperties(&gpuProps_, gpu_id), 0);

    smVersion_ = (gpuProps_.major << 8 | gpuProps_.minor);
  }
  virtual ~GPUInfo() {}
  bool support_imma() const {
    return smVersion_ == 0x0702 || smVersion_ == 0x0705;
  }
  bool support_idp4a() const {
    return smVersion_ == 0x0601 || smVersion_ == 0x0700 || support_imma();
  }
  const char* getDeviceName() const { return gpuProps_.name; }
  int getTextAlignment() const { return gpuProps_.textureAlignment; }
  int getSMVersion() const { return smVersion_; }
  int getMultiProcessorCount() const { return gpuProps_.multiProcessorCount; }
  int getMaxSMemSizePerSM() const {
    return gpuProps_.sharedMemPerMultiprocessor;
  }

  int getMaxSMemSizePerBlock() const;

 private:
  int smVersion_;
  cudaDeviceProp gpuProps_;
};

// TODO
// this only return gpuinfo instance on current gpu
// so it always return the one on which we first run
// or we must not use static variable (more overhead)
std::shared_ptr<GPUInfo> getGPUInfoInstance();

}  // namespace hie
