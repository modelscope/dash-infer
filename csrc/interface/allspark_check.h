/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_check.h
 */

#pragma once

#include <exception>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

class AsException : public std::exception {
 public:
  explicit AsException(const char* msg) noexcept : msg_(msg) {}
  explicit AsException(const std::string& err_str) noexcept
      : msg_(err_str.c_str()) {}
  const char* what() const noexcept override { return msg_; }

 private:
  const char* msg_;
};

class AsModelException : public AsException {
 public:
  explicit AsModelException(const char* msg) noexcept : AsException(msg) {}
  explicit AsModelException(const std::string& err_str) noexcept
      : AsException(err_str.c_str()) {}
};

class AsModelIOException : public AsModelException {
 public:
  explicit AsModelIOException(const char* msg) noexcept
      : AsModelException(msg) {}
  explicit AsModelIOException(const std::string& err_str) noexcept
      : AsModelException(err_str.c_str()) {}
};

#define AS_CHECK(status)                                        \
  do {                                                          \
    allspark::AsStatus err_status = status;                     \
    if (err_status != allspark::AsStatus::ALLSPARK_SUCCESS and  \
        err_status != allspark::AsStatus::ALLSPARK_STREAMING) { \
      printf("Failed: %s:%d '%s'\n", __FILE__, __LINE__,        \
             AsGetErrorByCode(err_status).c_str());             \
      throw AsException(AsGetErrorByCode(err_status).c_str());  \
    }                                                           \
  } while (0);

#define AS_THROW(status)                                   \
  do {                                                     \
    if (status != allspark::AsStatus::ALLSPARK_SUCCESS) {  \
      throw AsException(AsGetErrorByCode(status).c_str()); \
    }                                                      \
  } while (0);

namespace allspark {

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
  ALLSPARK_CHUNK_PREFILL = 14,    // 长请求首包分割进行
  ALLSPARK_DEPRECATED = 20,       // 触发过时接口
  ALLSPARK_STREAMING = 200,       // 流式返回
};

const std::string AsGetErrorByCode(AsStatus error_code);
void AsSaveError(const std::string& e);
const std::string AsConcatErrors();
void AsClearErrors();

}  // namespace allspark
