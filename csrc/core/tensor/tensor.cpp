/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tensor.cpp
 */

#include "tensor.h"  // NOLINT

#include <utility/cnpy.h>
#include <weight/weight_loader.h>

#include <algorithm>
#include <as_param_check.hpp>
#include <iterator>
#include <sstream>
#include <utility>
#include <vector>

#ifdef ENABLE_FP16
#include <common/float16.h>
#endif
#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

namespace allspark {
void CopyData(void* dst_data, DeviceType dst_device, const void* src_data,
              DeviceType src_device, int64_t nbytes,
              const DeviceContext* device_context) {
  if (nbytes == 0) {
    return;
  }
  if (src_device == DeviceType::CPU && dst_device == DeviceType::CPU) {
    memcpy(dst_data, src_data, nbytes);
  }
}

// allocate memory
AsTensor::AsTensor(const std::string& name, const DeviceType backend,
                   const DataType dtype, const DataMode mode,
                   const Shape& shape, int32_t flags)
    : name_(name),
      backend_(backend),
      dtype_(dtype),
      mode_(mode),
      shape_(shape),
      flags_(flags) {
  switch (mode_) {
    case DataMode::DENSE: {
      int64_t nbytes = shape_.Count() * SizeofType(dtype);
      int32_t data_flags = static_cast<int32_t>(AsDataFlags::empty_flag);
      data_ = std::make_shared<DenseData>(name, nbytes, backend, data_flags);
      break;
    }
    case DataMode::CSC:
      // data_ = std::make_shared<CSCData>();
      break;
    case DataMode::ELL:
      // data_ = std::make_shared<ELLData>();
      break;
    default:
      LOG(ERROR) << "Unspported DataMode:" << int(mode_) << std::endl;
      break;
  }
}

// allocate memory and fill AsTensor by copying tensor_proto.data (as char
// buffer), only used in parse io tensors from model_ir
AsTensor::AsTensor(const TensorProto& tensor_proto, DeviceType backend)
    : name_(tensor_proto.name()), backend_(backend) {
  if (!tensor_proto.data().empty()) {
    const std::string& tensor_string = tensor_proto.data();
    WeightFileParser weight_parser;
    TensorInfo info = weight_parser.ParseTensorInfo(tensor_string.c_str(),
                                                    tensor_string.size());
    this->SetDataType(info.dtype);
    this->SetDataMode(info.mode);
    this->SetShape(Shape(info.shape));

    int64_t nbytes = shape_.Count() * SizeofType(dtype_);
    data_ = std::make_shared<DenseData>(name_, nbytes, backend_);

    // only accept input & output tensor by this constructor.
    // don't accept have weight protobuf.
  } else {
    dtype_ = DataType::DATATYPE_UNDEFINED;
    mode_ = DataMode::DENSE;
    shape_ = {};
    data_ = std::make_shared<DenseData>(name_, 0, backend);
  }
}

AsTensor::AsTensor(const AsTensor& src_tensor, DeviceType backend)
    : name_(src_tensor.GetName()),
      dtype_(src_tensor.GetDataType()),
      mode_(src_tensor.GetDataMode()),
      shape_(src_tensor.GetShape()),
      backend_(backend) {
  if (src_tensor.GetDeviceType() == backend) {
    LOG(ERROR) << "AsTensor: tensors should be on different devices!"
               << std::endl;
    throw AsException("AsTensor: tensors should be on different devices!");
  }

  if (src_tensor.GetShape().Count() != shape_.Count()) {
    LOG(ERROR) << "AsTensor: Tensor copy should have same shape.";
    throw AsException("AsTensor: Copy In Different Shape");
  }

  if (dtype_ != src_tensor.GetDataType()) {
    LOG(ERROR) << "AsTensor: Tensor Copy should have same data type.";
    throw AsException("AsTensor: Copy in Different Type");
  }

  switch (mode_) {
    case DataMode::DENSE: {
      int64_t nbytes = shape_.Count() * SizeofType(dtype_);
      data_ =
          std::make_shared<DenseData>(src_tensor.GetName(), nbytes, backend_);
      CopyDataFrom(
          src_tensor.GetDataPtr(),
          src_tensor.GetShape().Count() * SizeofType(src_tensor.dtype_),
          src_tensor.GetDeviceType());

      break;
    }
    case DataMode::CSC:
      // data_ = std::make_shared<CSCData>();
      throw AsException("AsTensor: Copy sparse tensor not supported.");
      break;
    case DataMode::ELL:
      throw AsException("AsTensor: Copy sparse tensor not supported.");
      // data_ = std::make_shared<ELLData>();
      break;

    default:
      LOG(ERROR) << "Unspported DataMode:" << int(mode_) << std::endl;
      break;
  }
}

AsTensor::AsTensor(std::string new_name, const AsTensor& src_tensor)
    : name_(std::move(new_name)),
      dtype_(src_tensor.GetDataType()),
      mode_(src_tensor.GetDataMode()),
      shape_(src_tensor.GetShape()),
      backend_(src_tensor.GetDeviceType()) {
  if (src_tensor.GetName() == name_) {
    LOG(ERROR) << "AsTensor: tensors should have different names!" << std::endl;
    throw AsException("AsTensor: tensors should have different names!");
    return;
  }
  switch (mode_) {
    case DataMode::DENSE: {
      int64_t nbytes = shape_.Count() * SizeofType(dtype_);
      data_ = std::make_shared<DenseData>(name_, nbytes, backend_);
      CopyDataFrom(
          src_tensor.GetDataPtr(),
          src_tensor.GetShape().Count() * SizeofType(src_tensor.dtype_),
          src_tensor.GetDeviceType());

      return;
      break;
    }
    case DataMode::CSC:
      // data_ = std::make_shared<CSCData>();
      break;
    case DataMode::ELL:
      // data_ = std::make_shared<ELLData>();
      break;
    default:
      LOG(ERROR) << "Unspported DataMode:" << int(mode_) << std::endl;
      break;
  }

  throw AsException("AsTensor: Copy sparse tensor not supported.");
}

void AsTensor::SetMutable(bool is_mutable) { mutable_ = true; }

bool AsTensor::GetMutable() { return mutable_; }

// --------------- AsTensor <-> DLTensor --------------- //
struct TensorExchangeResource {
  // actual raw data in it
  std::shared_ptr<AsTensor> astensor_ptr;
  // a DLPack wrapper that refers to AsTensor's raw_data
  DLManagedTensor wrapper{};
};

// map AsTensor -> DLTensor, keep device type unchanged
DLManagedTensor* AsTensor::ToDLPack(DeviceContext* device_ctx,
                                    bool do_duplicate) const {
  auto* resource(new TensorExchangeResource);
  if (do_duplicate) {
    resource->astensor_ptr = std::make_shared<AsTensor>(
        PERSISTENT_TENSOR_PREFIX + std::string(name_), *this);
  } else {
    resource->astensor_ptr = std::make_shared<AsTensor>(*this);
  }

  resource->wrapper.manager_ctx = resource;
  // to be implicitly called by pytorch when as_out not used any more in
  // python
  resource->wrapper.deleter = [](DLManagedTensor* resource) {
    delete static_cast<TensorExchangeResource*>(resource->manager_ctx);
  };
  resource->wrapper.dl_tensor.data = resource->astensor_ptr->GetDataPtr();
  resource->wrapper.dl_tensor.ndim = shape_.Size();
  resource->wrapper.dl_tensor.shape =
      const_cast<int64_t*>(resource->astensor_ptr->GetShape().DataPtr());
  switch (backend_) {
    case DeviceType::CPU:
      resource->wrapper.dl_tensor.device.device_type = kDLCPU;
      resource->wrapper.dl_tensor.device.device_id = 0;
      break;
    default:
      LOG(ERROR) << "Unsupported DLDevice" << std::endl;
  }
  resource->wrapper.dl_tensor.strides = nullptr;
  resource->wrapper.dl_tensor.dtype.lanes = 1;
  resource->wrapper.dl_tensor.byte_offset = 0;
  switch (dtype_) {
    case DataType::FLOAT32: {
      resource->wrapper.dl_tensor.dtype.code = kDLFloat;
      resource->wrapper.dl_tensor.dtype.bits = 32;
      break;
    }
    case DataType::FLOAT16: {
      resource->wrapper.dl_tensor.dtype.code = kDLFloat;
      resource->wrapper.dl_tensor.dtype.bits = 16;
      break;
    }
    case DataType::INT64: {
      resource->wrapper.dl_tensor.dtype.code = kDLInt;
      resource->wrapper.dl_tensor.dtype.bits = 64;
      break;
    }
    case DataType::INT32: {
      resource->wrapper.dl_tensor.dtype.code = kDLInt;
      resource->wrapper.dl_tensor.dtype.bits = 32;
      break;
    }
    case DataType::INT16: {
      resource->wrapper.dl_tensor.dtype.code = kDLInt;
      resource->wrapper.dl_tensor.dtype.bits = 16;
      break;
    }
    case DataType::INT8: {
      resource->wrapper.dl_tensor.dtype.code = kDLInt;
      resource->wrapper.dl_tensor.dtype.bits = 8;
      break;
    }
    case DataType::UINT8: {
      resource->wrapper.dl_tensor.dtype.code = kDLUInt;
      resource->wrapper.dl_tensor.dtype.bits = 8;
      break;
    }
    case DataType::BOOL: {
      resource->wrapper.dl_tensor.dtype.code = kDLUInt;
      resource->wrapper.dl_tensor.dtype.bits = 1;
    }
    default:
      break;
  }
  return &(resource->wrapper);
}

std::vector<char> AsTensor::ToNumpy(std::string save_file_path) {
  auto nbytes = GetSizeInByte();
  std::vector<size_t> shape;
  for (auto i = 0; i < shape_.Size(); i++) {
    shape.push_back(shape_[i]);
  }
  void* p = GetDataPtr();
  std::shared_ptr<AsTensor> tp_cpu;
  if (backend_ != DeviceType::CPU && backend_ != DeviceType::CPU_PINNED) {
    tp_cpu = std::make_shared<AsTensor>(*this, DeviceType::CPU);
    p = tp_cpu->GetDataPtr();
  }

  if (p == nullptr) {
    return cnpy::to_npy_or_save(save_file_path, (char*)p,
                                std::vector<size_t>{0});
  }

  switch (dtype_) {
    case DataType::FLOAT32:
      return cnpy::to_npy_or_save(save_file_path, (float*)p, shape);
#ifdef ENABLE_FP16
    case DataType::FLOAT16:
      return cnpy::to_npy_or_save(save_file_path, (half*)p, shape);
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16:
      return cnpy::to_npy_or_save(save_file_path, (hie::bfloat16*)p, shape);
#endif
    case DataType::UINT8:
      return cnpy::to_npy_or_save(save_file_path, (unsigned char*)p, shape);
    case DataType::INT8:
      return cnpy::to_npy_or_save(save_file_path, (char*)p, shape);
    case DataType::INT16:
      return cnpy::to_npy_or_save(save_file_path, (short*)p, shape);
    case DataType::INT32:
      return cnpy::to_npy_or_save(save_file_path, (int*)p, shape);
    case DataType::INT64:
      return cnpy::to_npy_or_save(save_file_path, (long long*)p, shape);

    default: {
      LOG(ERROR) << "AsTensor::ToNumpy(): unsupported datatype "
                 << DataType_Name(dtype_);
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }

  return std::vector<char>();
}

void AsTensor::BuildFromDLTensor(const std::string& name,
                                 const DLManagedTensor* managed_dltensor,
                                 const DeviceType new_tensor_device_type) {
  const DLTensor& dltensor = managed_dltensor->dl_tensor;
  DeviceType dltensor_device_type = DeviceType::CPU;
  dltensor_device_type =
      DLDeviceTypeToAsDeviceType(dltensor.device.device_type);
  dtype_ = DataType::DATATYPE_UNDEFINED;
  switch (dltensor.dtype.code) {
    case kDLFloat:
      switch (dltensor.dtype.bits) {
        case 16:
          dtype_ = DataType::FLOAT16;
          break;
        case 32:
          dtype_ = DataType::FLOAT32;
          break;
      }
      break;
    case kDLUInt:
      switch (dltensor.dtype.bits) {
        case 1:
          dtype_ = DataType::BOOL;
          break;
      }
      break;
    case kDLInt:
      switch (dltensor.dtype.bits) {
        case 8:
          dtype_ = DataType::INT8;
          break;
        case 16:
          dtype_ = DataType::INT16;
          break;
        case 32:
          dtype_ = DataType::INT32;
          break;
        case 64:
          dtype_ = DataType::INT64;
          break;
      }
      break;
    default:
      LOG(ERROR) << "Unsupported DLDataType" << std::endl;
      dtype_ = DataType::DATATYPE_UNDEFINED;
  }
  shape_ = std::move(Shape(dltensor.ndim, dltensor.shape));
  int nbytes = SizeofType(dtype_) * shape_.Count();
  // we are not moving the ownership into engine, so we don't need to
  // delete the dltensor
  // NOTE: call the dltensor's deletor require a GIL
  data_ = std::make_shared<DenseData>(name, nbytes, new_tensor_device_type);
  this->CopyDataFrom(dltensor.data, nbytes, dltensor_device_type);
}

// map DLTensor -> AsTensor, keep same device
// use manager tensor's device type.
AsTensor::AsTensor(const std::string& name,
                   const DLManagedTensor* managed_dltensor)
    : name_(name), mode_(DataMode::DENSE) {
  if (managed_dltensor == nullptr) {
    LOG(ERROR) << "Invalid DLTensor : " << name << std::endl;
    exit(-1);
  } else {
    BuildFromDLTensor(name, managed_dltensor,
                      DLDeviceTypeToAsDeviceType(
                          managed_dltensor->dl_tensor.device.device_type));
  }
}

// solid copy DLTensor -> AsTensor, no matter same device or not
AsTensor::AsTensor(const std::string& name,
                   const DLManagedTensor* managed_dltensor,
                   const DeviceType as_backend)
    : name_(name), mode_(DataMode::DENSE), backend_(as_backend) {
  if (managed_dltensor == nullptr) {
    LOG(ERROR) << "Invalid DLTensor : " << name << std::endl;
    exit(-1);
  } else {
    BuildFromDLTensor(name, managed_dltensor, as_backend);
  }
}

DeviceType DLDeviceTypeToAsDeviceType(const DLDeviceType dltensor_device_type) {
  DeviceType as_tensor_device_type;
  switch (dltensor_device_type) {
    case kDLCPU:
      as_tensor_device_type = CPU;
      break;
    default:
      LOG(ERROR) << "Unsupported DLDevice" << dltensor_device_type << std::endl;
      as_tensor_device_type = DEVICETYPE_UNDEFINED;
  }
  return as_tensor_device_type;
}

AsTensor::AsTensor(const std::string& name,
                   const std::vector<std::vector<int64_t>>& input,
                   const DeviceType as_backend)
    : name_(name),
      mode_(DataMode::DENSE),
      backend_(as_backend),
      dtype_(DataType::INT64) {
  if (input.empty()) {
    LOG(ERROR) << "Invalid vector<vector> : " << name << std::endl;
    exit(-1);
  } else {
    shape_ = std::move(Shape(
        {static_cast<long>(input.size()), static_cast<long>(input[0].size())}));
    int nbytes = SizeofType(dtype_) * shape_.Count();
    std::vector<int64_t> cpu_src_mem;
    for (auto v : input)
      std::copy(v.begin(), v.end(), std::back_inserter(cpu_src_mem));
    data_ = std::make_shared<DenseData>(name, nbytes, backend_);
    this->CopyDataFrom(cpu_src_mem.data(), cpu_src_mem.size() * sizeof(int64_t),
                       DeviceType::CPU);
  }
}

AsTensor::AsTensor(
    const std::string& name,
    const std::pair<std::vector<float>, std::vector<int64_t>>& input,
    const DeviceType as_backend)
    : name_(name),
      mode_(DataMode::DENSE),
      backend_(as_backend),
      dtype_(DataType::FLOAT32) {
  shape_ = std::move(Shape(input.second));
  int nbytes = SizeofType(dtype_) * shape_.Count();
  data_ = std::make_shared<DenseData>(name, nbytes, backend_);
  this->CopyDataFrom(input.first.data(), nbytes, DeviceType::CPU);
}
// --------------- AsTensor <-> DLTensor --------------- //

const std::string& AsTensor::GetName() const { return name_; }
void AsTensor::SetName(const std::string& name) { name_ = name; }
const Shape& AsTensor::GetShape() const { return shape_; }
DataType AsTensor::GetDataType() const { return dtype_; }
DeviceType AsTensor::GetDeviceType() const { return backend_; }
DataMode AsTensor::GetDataMode() const { return mode_; }
void* AsTensor::GetDataPtr() const {
  if (block_ != nullptr) {
    return block_->RawData();
  } else {
    return data_->GetRawData();
  }
}

Data* AsTensor::GetData() const { return data_.get(); }

void AsTensor::CopyDataFrom(const void* src_data, const size_t src_bytes,
                            DeviceType src_device,
                            const DeviceContext* device_ctx) {
  if (mode_ != DataMode::DENSE) {
    throw std::runtime_error("currently not support copy sparse data.");
  }

  int64_t nbytes = shape_.Count() * SizeofType(dtype_);

  if (nbytes == 0) {
    return;
  }

  if (nbytes > src_bytes) {
    // NOTE(liyifei): this should not be considered a bug, so I change it to
    // a debug warning. (20231103)
    DLOG(WARNING) << "CopyDataFrom: this tensor is larger than src_data, "
                     "only src_bytes are copied, src_bytes: "
                  << src_bytes << " nbytes: " << nbytes;
  }
  nbytes = src_bytes < nbytes ? src_bytes : nbytes;

  void* dst_data = GetDataPtr();
  if (src_device == DeviceType::CPU && backend_ == DeviceType::CPU) {
    memcpy(dst_data, src_data, nbytes);
  } else {
    LOG(ERROR) << "Not support copy data between "
               << DeviceType_Name(src_device) << " and "
               << DeviceType_Name(backend_) << std::endl;
    assert(-1);
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
}

void AsTensor::CopyDataTo(void* dst_data, size_t dst_byte,
                          DeviceType dst_device,
                          const DeviceContext* device_ctx) const {
  void* src_data = GetDataPtr();

  int64_t nbytes = shape_.Count() * SizeofType(dtype_);

  if (dst_device == DeviceType::CPU && backend_ == DeviceType::CPU) {
    memcpy(dst_data, src_data, nbytes);
  } else {
    LOG(ERROR) << "Not support copy data between "
               << DeviceType_Name(dst_device) << " and "
               << DeviceType_Name(backend_) << std::endl;
    assert(-1);
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
}

void AsTensor::ShareData(AsTensor& rhs) {
  if (this->mode_ != rhs.mode_ && rhs.mode_ != DataMode::DENSE) {
    LOG(ERROR) << "not same mode: dst: " << (int)this->mode_
               << " src: " << (int)rhs.mode_;
    throw std::invalid_argument(
        "deep copy require same mode, and mode should be dense.");
  }

  if (this->shape_ != rhs.shape_) {
    LOG(ERROR) << "not same shape: dst: " << this->shape_.ToString()
               << " src: " << rhs.shape_.ToString();
    throw std::invalid_argument("deep copy require same shape");
  }

  if (this->dtype_ != rhs.dtype_) {
    LOG(ERROR) << "not same data type: dst: " << (int)rhs.dtype_
               << " src: " << (int)this->dtype_;
    throw std::invalid_argument("deep copy require same data type");
  }

  if (this->backend_ != rhs.backend_) {
    LOG(ERROR) << "not same backend type: dst: " << (int)rhs.backend_
               << " src: " << (int)this->backend_;
    throw std::invalid_argument("deep copy require same device type");
  }
  this->data_ = rhs.data_;
  this->block_ = rhs.block_;
}

void AsTensor::SwapData(AsTensor& rhs) {
  if (this->mode_ != rhs.mode_ && rhs.mode_ != DataMode::DENSE) {
    LOG(ERROR) << "not same mode: dst: " << (int)this->mode_
               << " src: " << (int)rhs.mode_;
    throw std::invalid_argument(
        "deep copy require same mode, and mode should be dense.");
  }

  if (this->shape_ != rhs.shape_) {
    LOG(ERROR) << "not same shape: dst: " << this->shape_.ToString()
               << " src: " << rhs.shape_.ToString();
    throw std::invalid_argument("deep copy require same shape");
  }

  if (this->dtype_ != rhs.dtype_) {
    LOG(ERROR) << "not same data type: dst: " << (int)rhs.dtype_
               << " src: " << (int)this->dtype_;
    throw std::invalid_argument("deep copy require same data type");
  }
  if (this->backend_ != rhs.backend_) {
    LOG(ERROR) << "not same backend type: dst: " << (int)rhs.backend_
               << " src: " << (int)this->backend_;
    throw std::invalid_argument("deep copy require same device type");
  }
  swap(rhs.data_, this->data_);
  swap(rhs.block_, this->block_);
}

static size_t SparseDataGetNNZ(const AsTensor& tensor) {
  assert(tensor.GetDataMode() != DataMode::DENSE);
  switch (tensor.GetDataMode()) {
    case DataMode::CSC:
      return ((size_t)((CSCData*)(tensor.GetData()))->GetNNZ());
    case DataMode::ELL:
      return ((size_t)((ELLData*)(tensor.GetData()))->GetNNZ());
    default:
      assert(-1);
      return 0;
  }
  return 0;
}

size_t AsTensor::GetStrideInByte() const {
  // TODO: support alignment in tensor allocation.
  if (this->shape_.Count() >= 2) {
    return this->shape_[1] * SizeofType(this->dtype_);
  } else {
    return this->shape_[0] * SizeofType(this->dtype_);
  }
}

size_t AsTensor::GetSizeInByte() const {
  if (this->mode_ == DataMode::DENSE) {
    // return current size or underline data size.
    size_t min_size = this->shape_.Count() * SizeofType(this->dtype_);
    if (block_ != nullptr)
      return std::max(static_cast<int64_t>(min_size),
                      static_cast<int64_t>(block_->Size()));
    else {
      DenseData* dense_data = static_cast<DenseData*>(data_.get());

      return std::max(static_cast<int64_t>(min_size),
                      static_cast<int64_t>(dense_data->GetSize()));
    }
  } else if (DataModeIsSparse(this->mode_)) {
    return SparseDataGetNNZ(*this) * SizeofType(this->dtype_);
  } else {
    assert(-1);
    return 0;
  }
}

AsStatus AsTensor::SetShape(Shape&& shape) {
  int64_t nbytes = shape.Count() * SizeofType(dtype_);
  if (!GetMutable()) {
    // the tensor is not mutable, for just, we can just print some log.
    LOG(ERROR) << "Warn: Tensor is set mutable, but user still try to "
                  "change the shape. "
               << name_;
    // abort();
  }
  if (block_ != nullptr) {
    block_->Resize(nbytes);
  } else {
    auto* dense_data = dynamic_cast<DenseData*>(data_.get());
    if (dense_data) {
      auto ret = dense_data->Resize(nbytes);
      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "Tensor Resize failed, trying to allocate nbytes "
                   << nbytes << "shape: " << shape;
        return ret;
      }
    }
  }

  shape_ = shape;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsTensor::Free() {
  if (block_ != nullptr) {
    block_->Free();
  } else {
    DenseData* dense_data = dynamic_cast<DenseData*>(data_.get());
    if (dense_data) {
      data_ = std::make_shared<DenseData>(name_, 1, backend_);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsTensor::SetDataType(DataType dtype) {
  dtype_ = dtype;
  if (!GetMutable()) {
    // the tensor is not mutable, for just, we can just print some log.
    LOG(ERROR) << "Warn: Tensor is set mutable, but user still try to "
                  "change the dtype. "
               << name_;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsTensor::SetDataMode(DataMode mode) {
  mode_ = mode;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsTensor::SetData(std::shared_ptr<Data> data) {
  data_ = data;
  if (!GetMutable()) {
    // the tensor is not mutable, for just, we can just print some log.
    LOG(ERROR) << "Warn: Tensor is set mutable, but user still try to "
                  "change the data. "
               << name_;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
std::string AsTensor::ToString() const {
  if (mode_ == DataMode::DENSE) {
    return string_format(
        "{ name: %s, device: %s, dtype: %s, shape: %s, addr: %p, val: %s }",
        name_.c_str(), DeviceType_Name(backend_).c_str(),
        DataType_Name(dtype_).c_str(), shape_.ToString().c_str(),
        ((data_ == nullptr && block_ == nullptr) || GetDataPtr() == nullptr)
            ? nullptr
            : GetDataPtr(),
        GetDataString().c_str());
  } else {
    return string_format(
        "{ name: %s, device: %s, dtype: %s, shape: %s, val: %s) "
        "sparse_type:%d}",
        name_.c_str(), DeviceType_Name(backend_).c_str(),
        DataType_Name(dtype_).c_str(), shape_.ToString().c_str(),
        GetDataString().c_str(), mode_);
  }
}

std::string AsTensor::ToStringAll() const {
  if (mode_ == DataMode::DENSE) {
    return string_format(
        "{ name: %s, device: %s, dtype: %s, shape: %s, val: %s }",
        name_.c_str(), DeviceType_Name(backend_).c_str(),
        DataType_Name(dtype_).c_str(), shape_.ToString().c_str(),
        GetDataStringAll().c_str());
  } else {
    return string_format(
        "{ name: %s, device: %s, dtype: %s, shape: %s, val: %s) "
        "sparse_type:%d}",
        name_.c_str(), DeviceType_Name(backend_).c_str(),
        DataType_Name(dtype_).c_str(), shape_.ToString().c_str(),
        GetDataStringAll().c_str(), mode_);
  }
}

std::string AsTensor::GetDataString() const {
  std::stringstream ss;
  if ((data_ == nullptr && block_ == nullptr) || GetDataPtr() == nullptr) {
    return "(null)";
  }
  int64_t count = shape_.Count() * SizeofType(dtype_);
  if (mode_ != DataMode::DENSE) {
    count = ((CSCData*)data_.get())->GetNNZ() * SizeofType(dtype_);
  }
  void* data_ptr = GetDataPtr();
  int print_len = 8;
  switch (dtype_) {
    case DataType::FLOAT32: {
      count /= 4;
      float* ptr = static_cast<float*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << ptr[count - 1 - i];
        }
      }
      break;
    }
    case DataType::FLOAT16: {
#ifdef ENABLE_FP16
      count /= 2;
      half* ptr = static_cast<half*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << (float)ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << (float)ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << (float)ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << (float)ptr[count - 1 - i];
        }
      }
#else
      LOG(INFO) << "Float16 support not compiled";
#endif
      break;
    }
    case DataType::BFLOAT16: {
#ifdef ENABLE_BF16
      count /= 2;
      hie::bfloat16* ptr = static_cast<hie::bfloat16*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << (float)ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << (float)ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << (float)ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << (float)ptr[count - 1 - i];
        }
      }
#else
      LOG(INFO) << "BFloat16 support not compiled";
#endif
      break;
    }
    case DataType::INT64: {
      count /= 8;
      int64_t* ptr = static_cast<int64_t*>(data_ptr);
      ss << ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << ptr[count - 1 - i];
        }
      }
      break;
    }
    case DataType::INT32: {
      count /= 4;
      int* ptr = static_cast<int*>(data_ptr);
      ss << ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << ptr[count - 1 - i];
        }
      }
      break;
    }
    case DataType::INT8: {
      count /= 1;
      int8_t* ptr = static_cast<int8_t*>(data_ptr);
      ss << (int)ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << (int)ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << (int)ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << (int)ptr[count - 1 - i];
        }
      }
      break;
    }
    case DataType::UINT8: {
      count /= 1;
      uint8_t* ptr = static_cast<uint8_t*>(data_ptr);
      ss << (int)ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << (int)ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << (int)ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << (int)ptr[count - 1 - i];
        }
      }
      break;
    }
    default:
      LOG(ERROR) << "Currently not support to dump this data type" << std::endl;
      return "(dump error)";
  }
  return ss.str();
}

std::string AsTensor::GetDataStringAll() const {
  std::stringstream ss;
  if ((data_ == nullptr && block_ == nullptr) || GetDataPtr() == nullptr) {
    return "(null)";
  }
  int64_t count = shape_.Count() * SizeofType(dtype_);
  if (mode_ != DataMode::DENSE) {
    count = ((CSCData*)data_.get())->GetNNZ() * SizeofType(dtype_);
  }
  void* data_ptr = GetDataPtr();
  int print_len = 8;
  switch (dtype_) {
    case DataType::FLOAT32: {
      count /= 4;
      float* ptr = static_cast<float*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << ptr[i];
      }
      break;
    }
    case DataType::FLOAT16: {
#ifdef ENABLE_FP16
      int print_len = 6;
      count /= 2;
      half* ptr = static_cast<half*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << (float)ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << (float)ptr[i];
      }
#else
      LOG(INFO) << "Float16 support not compiled";
#endif
      break;
    }
    case DataType::BFLOAT16: {
#ifdef ENABLE_BF16
      int print_len = 6;
      count /= 2;
      hie::bfloat16* ptr = static_cast<hie::bfloat16*>(data_ptr);
      ss.precision(6);
      ss.flags(std::ios_base::fixed);
      ss << (float)ptr[0];
      if (count <= print_len) {
        for (int i = 1; i < count; ++i) {
          ss << "," << (float)ptr[i];
        }
      } else {
        for (int i = 1; i < print_len / 2; ++i) {
          ss << "," << (float)ptr[i];
        }
        ss << ", ... ";
        for (int i = print_len / 2 - 1; i >= 0; --i) {
          ss << "," << (float)ptr[count - 1 - i];
        }
      }
#else
      LOG(INFO) << "BFloat16 support not compiled";
#endif
      break;
    }
    case DataType::INT64: {
      count /= 8;
      int64_t* ptr = static_cast<int64_t*>(data_ptr);
      ss << ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << ptr[i];
      }
      break;
    }
    case DataType::INT32: {
      count /= 4;
      int* ptr = static_cast<int*>(data_ptr);
      ss << ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << ptr[i];
      }
      break;
    }
    case DataType::INT8: {
      count /= 1;
      int8_t* ptr = static_cast<int8_t*>(data_ptr);
      ss << (int)ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << (int)ptr[i];
      }
      break;
    }
    case DataType::UINT8: {
      count /= 1;
      uint8_t* ptr = static_cast<uint8_t*>(data_ptr);
      ss << (int)ptr[0];
      for (int i = 1; i < count; ++i) {
        ss << "," << (int)ptr[i];
      }
      break;
    }
    default:
      LOG(ERROR) << "Currently not support to dump this data type" << std::endl;
      return "(dump error)";
  }
  return ss.str();
}

std::string AsTensor::GetMD5Sum() {
  byte md5_buffer[MD5_DIGEST_LENGTH];
  if (mode_ != DataMode::DENSE) {
    return "";
  }
  auto md5_byte2str = [](byte(&md5)[MD5_DIGEST_LENGTH]) -> std::string {
    static const char hex_index[16 + 1] = "0123456789abcdef";
    char convert_str[2 * MD5_DIGEST_LENGTH + 1];
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
      convert_str[i * 2] = hex_index[(md5[i] >> 4) & 0xF];
      convert_str[i * 2 + 1] = hex_index[(md5[i]) & 0xF];
    }
    convert_str[MD5_DIGEST_LENGTH * 2] = '\0';
    return std::string(convert_str);
  };
  void* host_ptr = nullptr;
  // copy to host if it's a device memory.

  int64_t count = shape_.Count() * SizeofType(dtype_);
  if (backend_ == DeviceType::CPU) {
    host_ptr = GetDataPtr();
  }

  md5_basic::MD5Impl()((byte*)host_ptr, count,
                       md5_buffer);  // same as openssl/md5 MD5();
  std::string md5_string = md5_byte2str(md5_buffer);
  return md5_string;
}

}  // namespace allspark
