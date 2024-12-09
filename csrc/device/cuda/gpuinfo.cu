/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gpuinfo.cu
 */

#include "gpuinfo.h"

namespace hie {

std::shared_ptr<GPUInfo> getGPUInfoInstance() {
  static std::shared_ptr<GPUInfo> g_instance;
  if (!g_instance) {
    g_instance = std::make_shared<GPUInfo>();
  }
  return g_instance;
}

int GPUInfo::getMaxSMemSizePerBlock() const {
#if CUDA_VERSION >= 9000
  if (getSMVersion() == 0x700) {
    // Volta has 98KB of memory, despite only reporting 48KB.
    // assert(gpuProps.sharedMemPerBlock() == 48 * 1024);
    return 96 * 1024;
  }
#endif
  return gpuProps_.sharedMemPerBlock;
}

}  // namespace hie
