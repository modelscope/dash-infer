/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark.min.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "check_cuda.h"
#include "shape.h"

namespace allspark {

enum class AsCacheMode {
  AsCacheDefault = 0,
  AsCacheQuantI8 = 1,
  AsCacheQuantU4 = 2,
};

enum DataType : int {
  DATATYPE_UNDEFINED = 0,
  FLOAT32 = 1,
  FLOAT16 = 2,
  INT8 = 3,
  INT16 = 4,
  INT32 = 5,
  INT64 = 6,
  STRING = 7,
  BOOL = 8,
  BFLOAT16 = 9,
  UINT8 = 10,
  POINTER = 20,
};

enum DeviceType : int {
  DEVICETYPE_UNDEFINED = 0,
  CUDA = 2,
};

// AllSpark status code
enum class AsStatus : int {
  ALLSPARK_SUCCESS = 0,  // 正常状态

  ALLSPARK_UNKNOWN_ERROR = 1,        // 兜底错误类型
  ALLSPARK_PARAM_ERROR = 2,          // 传入的参数值错误
  ALLSPARK_IO_ERROR = 3,             // IO访问错误
  ALLSPARK_MEMORY_ERROR = 4,         // 内存越界等错误
  ALLSPARK_RUNTIME_ERROR = 5,        // 运行时触发的OP错误返回
  ALLSPARK_EXCEED_LIMIT_ERROR = 7,   // 参数或输入超限
  ALLSPARK_INVALID_CALL_ERROR = 8,   // 无效调用
  ALLSPARK_EMPTY_REQUEST = 9,        // 无有效请求
  ALLSPARK_ILLEGAL_REQUEST_ID = 10,  // 没有找到有效的request_id
  ALLSPARK_CACHE_MEMORY_OUT = 11,
  ALLSPARK_REQUEST_DENIED = 12,   // 停服务状态，拒绝服务
  ALLSPARK_LORA_NOT_LOADED = 13,  // 用户指定的lora没加载
  ALLSPARK_DEPRECATED = 20,       // 触发过时接口
  ALLSPARK_STREAMING = 200,       // 流式返回
};

inline size_t SizeofType(DataType data_type) {
  switch (data_type) {
    case DataType::POINTER:
      return sizeof(void*);
    case DataType::INT64:
      return 8;
    case DataType::FLOAT32:
    case DataType::INT32:
      return 4;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
    case DataType::INT16:
      return 2;
    case DataType::INT8:
    case DataType::UINT8:
      return 1;
    case DataType::DATATYPE_UNDEFINED:
      return 0;
    default:
      return 1;
  }
}

class AsTensor {
 public:
  AsTensor(std::string name, DeviceType device, DataType data_type)
      : name_(std::move(name)), dtype_(data_type), block_(nullptr) {
    if (device != DeviceType::CUDA) {
      throw std::runtime_error("AsTensor: only support device CUDA");
    }
  }
  ~AsTensor() {}

  void* GetDataPtr() const { return block_; }
  size_t GetSizeInByte() const { return shape_.Count() * SizeofType(dtype_); }

  AsStatus Free() {
    if (block_) {
      AS_CHECK_CUDA(cudaFree(block_));
      block_ = nullptr;
      shape_ = std::move(Shape());
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  AsStatus SetShape(Shape shape) {
    int64_t nbytes = shape.Count() * SizeofType(dtype_);

    if (block_ != nullptr) {
      std::cerr << "AsTensor: does not support resizing a tensor for now";
      return AsStatus::ALLSPARK_PARAM_ERROR;
    } else {
      void* dense_data{nullptr};
      AS_CHECK_CUDA(cudaMalloc(&dense_data, nbytes));

      if (dense_data) {
        block_ = dense_data;
      } else {
        std::cerr << "AsTensor: failed to allocate " << nbytes
                  << " bytes with cudaMalloc";
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
    }

    shape_ = std::move(shape);
    return AsStatus::ALLSPARK_SUCCESS;
  }

 private:
  const std::string name_;
  const DataType dtype_;

  void* block_;
  Shape shape_;
};

using TensorMap = std::map<std::string, std::shared_ptr<AsTensor>>;

class DeviceContext {
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext& operator=(DeviceContext&&) = delete;

 public:
  DeviceContext() = default;
  virtual ~DeviceContext() {}
  virtual DeviceType GetDeviceType() const = 0;
  virtual int GetRank() const = 0;
  virtual int GetNranks() const = 0;
  virtual void SetDeviceId(int device_id) = 0;
  virtual int GetDeviceId() = 0;
  virtual void Synchronize() const = 0;

 protected:
  DataType dtype = DataType::DATATYPE_UNDEFINED;
};

class CUDAContext : public DeviceContext {
 public:
  CUDAContext() : device_id_(0), stream_(0) {
    nranks_ = 1;
    rank_ = 0;
  }

  virtual ~CUDAContext() override;

  DeviceType GetDeviceType() const override { return DeviceType::CUDA; }

  void SetDeviceId(int device_id) override;

  int GetDeviceId() override { return device_id_; }

  int GetRank() const override;

  int GetNranks() const override;

  virtual void SetDtype(DataType new_dtype);
  // ---------------------------------------------------------------------------

  static int GetStreamProcessorVersion(int device_id);

  int GetStreamProcessorVersion();

  int GetDeviceId() const { return device_id_; }

  cudaStream_t GetStream() const { return stream_; }

  void Synchronize() const;

 private:
  int device_id_ = 0;
  cudaStream_t stream_;
  int nranks_;
  int rank_;
};

// use factory class to avoid device related dependency pollute user program.
class DeviceContextFactory {
 public:
  static std::shared_ptr<DeviceContext> CreateDeviceContext(
      const DeviceType device_type);
  static std::shared_ptr<DeviceContext> CreateCUDAContext();
};

}  // namespace allspark
