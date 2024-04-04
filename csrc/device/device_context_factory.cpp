/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    device_context_factory.cpp
 */

#include <common/device_context.h>
#include <cpu/cpu_context.h>
using namespace allspark;

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateDeviceContext(
    const DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return DeviceContextFactory::CreateCPUContext();
    default:
      LOG(ERROR) << "DeviceType Error.";
      break;
  }
  return nullptr;
}

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateCPUContext() {
  return std::make_shared<CPUContext>();
}
