/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    common.h
 */

#pragma once

#include <allspark.pb.h>
#include <glog/logging.h>
#include <interface/allspark.h>
#include <utility/check.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace allspark {

#define PERSISTENT_TENSOR_PREFIX "PERSISTENT_TENSOR:"
#define BFC_SPACE_NAME "__BFC_SPACE__:"

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete
#endif

#ifndef DISABLE_MOVE_COPY_ASSIGN
#define DISABLE_MOVE_COPY_ASSIGN(classname)        \
  classname(classname const&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(classname const&) = delete; \
  classname& operator=(classname&&) = delete
#endif

// Enable this config will generate all free and malloc log, enable if want to
// debug related issue.
// #define CONFIG_MEM_DEBUG .

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) +
                1;  // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1);  // We don't want the '\0' inside
}

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

inline std::string DataTypeToString(DataType data_type) {
  switch (data_type) {
    case DataType::DATATYPE_UNDEFINED:
      return "DataType::DATATYPE_UNDEFINED";
    case DataType::FLOAT32:
      return "DataType::FLOAT32";
    case DataType::FLOAT16:
      return "DataType::FLOAT16";
    case DataType::INT8:
      return "DataType::INT8";
    case DataType::INT16:
      return "DataType::INT16";
    case DataType::INT32:
      return "DataType::INT32";
    case DataType::INT64:
      return "DataType::INT64";
    case DataType::STRING:
      return "DataType::STRING";
    case DataType::BOOL:
      return "DataType::BOOL";
    case DataType::BFLOAT16:
      return "DataType::BFLOAT16";
    case DataType::UINT8:
      return "DataType::UINT8";
    case DataType::POINTER:
      return "DataType::POINTER";
    default:
      return "UNKNOWN";
  }
}

inline std::string DeviceTypeToString(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::DEVICETYPE_UNDEFINED:
      return "DEVICETYPE_UNDEFINED";
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES:
      return "COMPILE_TIME_MAX_DEVICE_TYPES";
    case DeviceType::CPU_PINNED:
      return "CPU_PINNED";
    default:
      return "UNKNOWN";
  }
}

inline std::string AsStatusToString(AsStatus status) {
  switch (status) {
    case AsStatus::ALLSPARK_SUCCESS:
      return "ALLSPARK_SUCCESS";
    case AsStatus::ALLSPARK_UNKNOWN_ERROR:
      return "ALLSPARK_UNKNOWN_ERROR";
    case AsStatus::ALLSPARK_PARAM_ERROR:
      return "ALLSPARK_PARAM_ERROR";
    case AsStatus::ALLSPARK_IO_ERROR:
      return "ALLSPARK_IO_ERROR";
    case AsStatus::ALLSPARK_MEMORY_ERROR:
      return "ALLSPARK_MEMORY_ERROR";
    case AsStatus::ALLSPARK_RUNTIME_ERROR:
      return "ALLSPARK_RUNTIME_ERROR";
    case AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR:
      return "ALLSPARK_EXCEED_LIMIT_ERROR";
    case AsStatus::ALLSPARK_INVALID_CALL_ERROR:
      return "ALLSPARK_INVALID_CALL_ERROR";
    case AsStatus::ALLSPARK_EMPTY_REQUEST:
      return "ALLSPARK_EMPTY_REQUEST";
    case AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID:
      return "ALLSPARK_ILLEGAL_REQUEST_ID";
    case AsStatus::ALLSPARK_CACHE_MEMORY_OUT:
      return "ALLSPARK_CACHE_MEMORY_OUT";
    case AsStatus::ALLSPARK_REQUEST_DENIED:
      return "ALLSPARK_REQUEST_DENIED";
    case AsStatus::ALLSPARK_CHUNK_PREFILL:
      return "ALLSPARK_CHUNK_PREFILL";
    case AsStatus::ALLSPARK_DEPRECATED:
      return "ALLSPARK_DEPRECATED";
    case AsStatus::ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR:
      return "ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR";
    case AsStatus::ALLSPARK_LORA_RANK_EXCEED_LIMIT_ERROR:
      return "ALLSPARK_LORA_RANK_EXCEED_LIMIT_ERROR";
    case AsStatus::ALLSPARK_LORA_NOT_FOUND:
      return "ALLSPARK_LORA_NOT_FOUND";
    case AsStatus::ALLSPARK_LORA_ALREADY_LOADED:
      return "ALLSPARK_LORA_ALREADY_LOADED";
    case AsStatus::ALLSPARK_LORA_IN_USE:
      return "ALLSPARK_LORA_IN_USE";
    case AsStatus::ALLSPARK_STREAMING:
      return "ALLSPARK_STREAMING";
    default:
      return "UNKNOWN";
  }
}

inline bool DataModeIsSparse(DataMode mode) {
  if (mode == DataMode::CSC || mode == DataMode::ELL) return true;
  return false;
}

struct RankInfo {
  RankInfo() : rank_id(0), rank_size(1) {}
  RankInfo(int rank_id_p, int rank_size_p)
      : rank_id(rank_id_p), rank_size(rank_size_p) {}
  int rank_id;
  int rank_size;

  inline bool operator==(const RankInfo& other) const {
    return (this->rank_size == other.rank_size) &&
           (this->rank_id == other.rank_id);
  }
};

inline std::ostream& operator<<(std::ostream& os, const RankInfo& rk) {
  os << "RankInfo[" << rk.rank_id << "/" << rk.rank_size << "]";
  return os;
}

inline bool operator<(const RankInfo& lhs, const RankInfo& rhs) {
  return lhs.rank_size * lhs.rank_size + lhs.rank_id <
         rhs.rank_size * rhs.rank_size + rhs.rank_id;
}

struct AsDevice {
  AsDevice() = delete;
  explicit AsDevice(const DeviceType dev_type, const int dev_id)
      : device_type(dev_type), device_id(dev_id) {}
  bool operator<(const AsDevice other_dev) const {
    return (device_type == other_dev.device_type)
               ? (device_id < other_dev.device_id)
               : (device_type < other_dev.device_type);
  }
  bool operator==(const AsDevice other_dev) const {
    return (device_type == other_dev.device_type) &&
           (device_id == other_dev.device_id);
  }
  bool operator>(const AsDevice other_dev) const {
    return (device_type == other_dev.device_type)
               ? (device_id > other_dev.device_id)
               : (device_type > other_dev.device_type);
  }

  DeviceType device_type{DeviceType::CPU};
  int device_id{0};
};

inline std::ostream& operator<<(std::ostream& os, DeviceType device_type) {
  switch (device_type) {
    case allspark::DeviceType::CUDA:
      return os << "CUDA";
      break;
    case allspark::DeviceType::CPU:
      return os << "CPU";
      break;
    default:
      // If the error value isn't found (shouldn't happen)
      return os << "Unkown Device";
  }
}

inline int get_layer_num(std::string str) {
  std::stringstream ss(str);
  std::string temp;
  while (std::getline(ss, temp, '.')) {
    bool flag = true;
    for (char c : temp) {
      if (!std::isdigit(c)) /* 如果不是数字，返回 false */ {
        flag = false;
        break;
      }
    }
    if (flag) {
      return std::stoi(temp);
    }
  }
  return -1;
}

// deprecated api declear
#if __cplusplus >= 201402L  // c++14

#if defined(__GNUC__) && !defined(__clang__)
#define AS_DEPRECATED(s) [[deprecated(s)]]
#endif

#ifdef __clang__
// clang does not support deprecate keyword.
#define AS_DEPRECATED(s)
#endif

#else
#endif
struct BatchGencfg {
  int batch_size = 0;
  void* repetition_penalty_list = nullptr;
  void* presence_penalty_list = nullptr;
  void* frequency_penalty_list = nullptr;
  void* no_repeat_ngram_size_list = nullptr;
  void* min_length_list = nullptr;
  void* eos_token_id_list = nullptr;
  void* cur_len_list = nullptr;
  void* input_len_list = nullptr;
  void* suppress_repetition_in_generation_list = nullptr;
};
}  // namespace allspark

namespace std {
template <>
struct hash<allspark::RankInfo> {
  inline size_t operator()(const allspark::RankInfo& x) const {
    return x.rank_size * 10000L + x.rank_id;
  }
};

}  // namespace std
