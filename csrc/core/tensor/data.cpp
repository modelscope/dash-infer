/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    data.cpp
 */

#include "data.h"  // NOLINT

#include <cpu/cpu_allocator.h>

namespace allspark {

// Data --------------------------- //
Data::Data(const std::string& name, DeviceType device_type, int32_t flags)
    : name_(name),
      device_type_(device_type),
      raw_data_(nullptr),
      flags_(flags) {
  switch (device_type) {
    case DeviceType::CPU: {
      allocator_ = std::make_shared<CPUAllocator>();
      break;
    }
    default: {
      LOG(ERROR) << "DeviceType::" << device_type
                 << " is not supported. Please check build option."
                 << std::endl;
      throw AsException("ALLSPARK_PARAM_ERROR");
    }
  }
}

void* Data::GetRawData() const { return raw_data_; }

// DenseData ---------------------- //
DenseData::DenseData(const std::string& name, int64_t nbytes,
                     DeviceType device_type, int32_t flags)
    : Data(name, device_type, flags), nbytes_(nbytes), deleter_(nullptr) {
  if (nbytes) {
    auto ret = allocator_->Alloc(&raw_data_, nbytes, name);
    AS_CHECK(ret);
  }
}

DenseData::DenseData(const std::string& name, int64_t nbytes,
                     DeviceType device_type, deleter_t deleter)
    : Data(name, device_type), nbytes_(nbytes), deleter_(deleter) {
  if (nbytes) {
    AS_CHECK(allocator_->Alloc(&raw_data_, nbytes, name));
  }
}

DenseData::DenseData(const std::string& name, int64_t nbytes,
                     DeviceType device_type, void* raw_data, deleter_t deleter)
    : Data(name, device_type), nbytes_(nbytes), deleter_(deleter) {
  raw_data_ = raw_data;
}

DenseData::~DenseData() {
  if (raw_data_) {
    if (deleter_ != nullptr) {
      deleter_(raw_data_);
      deleter_ = nullptr;
    } else {
      allocator_->Free(raw_data_);
    }
  }
}

AsStatus DenseData::Resize(int64_t nbytes) {
  if (nbytes <= nbytes_) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (raw_data_) {
    if (deleter_ != nullptr) {
      deleter_(raw_data_);
      deleter_ = nullptr;
    } else {
      AS_CHECK_STATUS(allocator_->Free(raw_data_));
    }
  }
  AS_CHECK_STATUS(allocator_->Alloc(&raw_data_, nbytes, name_));
  nbytes_ = nbytes;
  return AsStatus::ALLSPARK_SUCCESS;
}

int64_t DenseData::GetSize() const { return nbytes_; }

// CSCData ------------------------ //
CSCData::CSCData(const std::string& name, int nnz, int cols,
                 DeviceType device_type, int type_size)
    : Data(name, device_type),
      nnz_(nnz),
      col_offsets_(nullptr),
      row_indices_(nullptr) {
  if (nnz) {
    AS_CHECK(allocator_->Alloc(&raw_data_, nnz * type_size, name));
    AS_CHECK(allocator_->Alloc(&col_offsets_, (cols + 1) * sizeof(int), name));
    AS_CHECK(allocator_->Alloc(&row_indices_, nnz * sizeof(int), name));
  }
}
CSCData::~CSCData() {
  if (row_indices_) {
    allocator_->Free(raw_data_);
    allocator_->Free(row_indices_);
    allocator_->Free(col_offsets_);
  }
}
void* CSCData::GetRowIndices() const { return row_indices_; }
void* CSCData::GetColOffsets() const { return col_offsets_; }
int CSCData::GetNNZ() const { return nnz_; }

// ELLData------------------------- //
ELLData::ELLData(const std::string& name, int nnz, int cols,
                 DeviceType device_type, int type_size)
    : Data(name, device_type), nnz_(nnz), row_indices_(nullptr) {
  if (nnz) {
    AS_CHECK(allocator_->Alloc(&raw_data_, nnz * type_size, name));
    AS_CHECK(
        allocator_->Alloc(&row_indices_, nnz * sizeof(unsigned short), name));
  }
}
ELLData::~ELLData() {
  if (row_indices_) {
    allocator_->Free(raw_data_);
    allocator_->Free(row_indices_);
  }
}
void* ELLData::GetRowIndices() const { return row_indices_; }
int ELLData::GetNNZ() const { return nnz_; }
}  // namespace allspark
