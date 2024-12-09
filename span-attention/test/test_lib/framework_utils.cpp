/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    framework_utils.cpp
 */

#include "framework_utils.hpp"

namespace common {
bool AsyncH2D(const void* host, asTensor* device, size_t size,
              cudaStream_t stream) {
  auto err = cudaMemcpyAsync(device->GetDataPtr(), host, size,
                             cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    std::cerr << "ERROR: AsyncH2D: " << err << ", " << cudaGetErrorString(err)
              << std::endl;
  }
  return err == cudaSuccess;
}

bool AsyncD2H(const asTensor* device, void* host, size_t size,
              cudaStream_t stream) {
  auto err = cudaMemcpyAsync(host, device->GetDataPtr(), size,
                             cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    std::cerr << "ERROR: AsyncD2H: " << err << ", " << cudaGetErrorString(err)
              << std::endl;
  }
  return err == cudaSuccess;
}

void AddTensor(allspark::TensorMap& t_map, const std::string& name,
               allspark::DataType data_type) {
  t_map.insert(std::make_pair<std::string, std::unique_ptr<asTensor>>(
      name.c_str(), std::make_unique<asTensor>(name, asCUDA, data_type)));
}

}  // namespace common
