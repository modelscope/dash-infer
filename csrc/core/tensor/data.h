/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    data.h
 */

#pragma once
#include <common/allocator.h>

#include <memory>
#include <string>

namespace allspark {

enum AsDataFlags {
  empty_flag = 0x0,
};

/*!
 * @brief Data base class
 */
class Data {
 public:
  explicit Data(const std::string& name,
                DeviceType device_type = DeviceType::CPU,
                int32_t flags = AsDataFlags::empty_flag);
  virtual ~Data() = default;
  void* GetRawData() const;
  std::string GetName() const { return name_; };
  DeviceType GetDeviceType() const { return device_type_; };

 protected:
  void* raw_data_ = nullptr;
  std::shared_ptr<Allocator> allocator_;
  std::string name_;
  int32_t flags_ = AsDataFlags::empty_flag;

 private:
  const DeviceType device_type_;
};

/*!
 * @brief DenseData
    feature below:
 *      > support construct from external data
 *      > support resize
 */
using deleter_t = std::function<void(void*)>;
class DenseData : public Data {
 public:
  // construct data from inner allocator
  explicit DenseData(const std::string& name, int64_t nbytes = 0,
                     DeviceType device_type = DeviceType::CPU,
                     int32_t flags = AsDataFlags::empty_flag);
  // construct data from external data
  explicit DenseData(const std::string& name, int64_t nbytes,
                     DeviceType device_type, void* raw_data, deleter_t deleter);
  explicit DenseData(const std::string& name, int64_t nbytes,
                     DeviceType device_type, deleter_t deleter);

  ~DenseData();
  AsStatus Resize(int64_t nbytes);
  int64_t GetSize() const;

 private:
  int64_t nbytes_ = 0;
  deleter_t deleter_;
};

/*!
 * @brief CSCData
 */
class CSCData : public Data {
 public:
  explicit CSCData(const std::string& name, int nnz, int cols,
                   DeviceType device_type, int type_size);
  ~CSCData();
  void* GetRowIndices() const;
  void* GetColOffsets() const;
  int GetNNZ() const;

 private:
  int nnz_;
  void* row_indices_;
  void* col_offsets_;
};

class ELLData : public Data {
 public:
  explicit ELLData(const std::string& name, int nnz, int cols,
                   DeviceType device_type, int type_size);
  ~ELLData();
  void* GetRowIndices() const;
  int GetNNZ() const;

 private:
  int nnz_;
  void* row_indices_;
};
}  // namespace allspark
