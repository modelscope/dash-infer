/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    device_context_factory.cpp
 */

#include <common/device_context.h>
#include <cpu/cpu_context.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

using namespace allspark;

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateDeviceContext(
    const DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return DeviceContextFactory::CreateCPUContext();
    case DeviceType::CUDA:
      return DeviceContextFactory::CreateCUDAContext();
    default:
      LOG(ERROR) << "DeviceType Error.";
      break;
  }
  return nullptr;
}

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateCPUContext() {
  return std::make_shared<CPUContext>();
}

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateCUDAContext() {
#ifdef ENABLE_CUDA
  return std::make_shared<CUDAContext>();
#else
  LOG(ERROR) << "try to get cuda device context without compile support cuda.";
  return nullptr;
#endif
}
