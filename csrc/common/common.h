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
    case allspark::DeviceType::CPU:
      return os << "CPU";
      break;
    default:
      // If the error value isn't found (shouldn't happen)
      return os << "Unkown Device";
  }
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

}  // namespace allspark

namespace std {
template <>
struct hash<allspark::RankInfo> {
  inline size_t operator()(const allspark::RankInfo& x) const {
    return x.rank_size * 10000L + x.rank_id;
  }
};
}  // namespace std
