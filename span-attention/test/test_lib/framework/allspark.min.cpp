/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark.min.cpp
 */
#include "allspark.min.hpp"

namespace allspark {

CUDAContext::~CUDAContext() {
  try {
    if (stream_) {
      AS_CHECK_CUDA(cudaStreamDestroy(stream_));
    }
  } catch (std::exception& e) {
    // avoid the gcc warnning.
    std::cerr << "Exception in destroy cuda context: " << e.what();
  }
}

void CUDAContext::SetDeviceId(int device_id) {
  // if have old handler, destory the device id and handler.
  if (stream_) {
    AS_CHECK_CUDA(cudaSetDevice(device_id_));
    AS_CHECK_CUDA(cudaStreamDestroy(stream_));
  }

  // setup new device id related resource.
  device_id_ = device_id;
  AS_CHECK_CUDA(cudaSetDevice(device_id_));
  AS_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

int CUDAContext::GetStreamProcessorVersion(int device_id) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int sm_version = (device_prop.major << 8 | device_prop.minor);
  std::cout << "Get SM version return " << std::hex << "0x" << sm_version
            << " Device Id: " << device_id << std::endl;
  return sm_version;
}

int CUDAContext::GetStreamProcessorVersion() {
  return GetStreamProcessorVersion(device_id_);
}

void CUDAContext::Synchronize() const {
  AS_CHECK_CUDA(cudaStreamSynchronize(stream_));
}

void CUDAContext::SetDtype(DataType new_dtype) {
  // Check dtype is supported status

  // for cuda support:
  // ref:
  // https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
  // check bf16 support status.
  if (new_dtype == DataType::BFLOAT16) {
    int sm_version = this->GetStreamProcessorVersion();
    // sm_version will get value like: 806 (in hex)
    if (sm_version < 0x800) {
      std::cout
          << "Current device: " << device_id_ << " sm: " << std::hex
          << sm_version
          << " not support bfloat16 datatype, please set float16 as data type."
          << std::endl;

      throw std::invalid_argument(
          "bfloat16 is not supported for current device.");
    }
  }

  // TODO: check fb8 support status
  dtype = new_dtype;
}

int CUDAContext::GetRank() const { return rank_; }
int CUDAContext::GetNranks() const { return nranks_; }

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateDeviceContext(
    const DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CUDA:
      return DeviceContextFactory::CreateCUDAContext();
    default:
      std::cerr << "DeviceType Error" << std::endl;
      break;
  }
  return nullptr;
}

std::shared_ptr<DeviceContext> DeviceContextFactory::CreateCUDAContext() {
  return std::make_shared<CUDAContext>();
}

}  // namespace allspark
